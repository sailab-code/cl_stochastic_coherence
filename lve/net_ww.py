import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lve.encoder import EncoderFactory


class UnsupervisedClassifier(nn.Module):

    def __init__(self, options):
        super(UnsupervisedClassifier, self).__init__()
        self.options = options
        self.unsupervised_projection = nn.Sequential(
            nn.Linear(in_features=options['num_what'], out_features=100, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=options['unsupervised_categories'], bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.unsupervised_projection(x)


class BaseOpenSetClassifier(nn.Module):

    def __init__(self, options, sup_buffer, device, distance='euclidean'):
        super(BaseOpenSetClassifier, self).__init__()
        self.w = options["w"]
        self.h = options["h"]
        self.sup_buffer = sup_buffer
        self.options = options
        self.device = device
        self.distance = distance

    def forward(self, frame_embeddings):
        pass

    def forward_and_compute_supervised_loss(self, data, labels):
        pass

    def predict(self, frame_embeddings):
        pass

    def compute_mask_of_pixels_to_predict(self, frame_embeddings):
        template_list = self.sup_buffer.get_embeddings()  # list of embedding vector (each of them dim x 1)
        _, template_classes = self.sup_buffer.get_embeddings_labels()
        #   frame_embeddings = [ num-pixels, num_what,]
        if len(template_list) > 0:
            dists_from_templates = [None] * len(template_list)
            for i in range(len(template_list)):
                template = template_list[i]  # [1, num_what ]
                if self.distance == 'cosine':
                    dists_from_templates[i] = 1. - torch.sum(frame_embeddings * template, dim=1)
                    # dists_from_templates[i] = 2. - (torch.inner(frame_embeddings, template)).squeeze()
                elif self.distance == 'euclidean':
                    dists_from_templates[i] = torch.sum(torch.pow(frame_embeddings - template, 2.0),
                                                        dim=1)  # num-pixels
                else:
                    raise NotImplementedError

            dists_from_templates = torch.stack(dists_from_templates, dim=0)  # [num-templates, num-pixels]
            min_dists_from_templates, min_indexes_from_template = torch.min(dists_from_templates,
                                                                            dim=0)  # num-pixels

            mask_list = []
            for thresh in self.options["dist_threshold"]:
                mask_list.append(min_dists_from_templates <= thresh)

            return mask_list, min_dists_from_templates, \
                   template_classes[min_indexes_from_template].to(frame_embeddings.device)  # num-pixels

        else:
            return [torch.zeros((1, self.h * self.w), dtype=torch.bool, device=frame_embeddings.device)] * len(
                self.options["dist_threshold"]), None, None


class NNOpenSetClassifier(BaseOpenSetClassifier):

    def __init__(self, options, sup_buffer, device, distance='euclidean'):
        super(NNOpenSetClassifier, self).__init__(options, sup_buffer, device, distance)

    def predict(self, frame_embeddings):
        # frame_embeddings [num_pixel, num_what]
        mask_list, min_dists_from_templates, neigh_classes = self.compute_mask_of_pixels_to_predict(frame_embeddings)

        if len(self.sup_buffer.get_embeddings()) == 0:
            return [torch.zeros((self.h * self.w, self.options["supervised_categories"]),
                                device=frame_embeddings.device)] * len(self.options["dist_threshold"]), \
                   mask_list, \
                   [(self.options["supervised_categories"] - 1) * torch.ones((self.h * self.w),
                                                                             device=frame_embeddings.device)] * len(
                       self.options["dist_threshold"])
        else:
            one_hot_classes = \
                torch.nn.functional.one_hot(neigh_classes,
                                            num_classes=self.options["supervised_categories"]).to(torch.float)

            masked_pred_list = []
            neigh_classes_list = []
            for mask in mask_list:
                neigh_classes_temp = neigh_classes.detach().clone()
                masked_pred_list.append(one_hot_classes * mask.view(self.w * self.h, 1))
                neigh_classes_temp[mask == False] = self.options["supervised_categories"] - 1
                neigh_classes_list.append(neigh_classes_temp)

            return masked_pred_list, mask_list, neigh_classes_list

    def forward_and_compute_supervised_loss(self, data, labels):
        return torch.tensor(0.0, device=self.device), 0.0


class NeuralOpenSetClassifier(BaseOpenSetClassifier):

    def __init__(self, options, sup_buffer, device):
        super(NeuralOpenSetClassifier, self).__init__(options, sup_buffer, device)

        self.supervised_projection = nn.Sequential(
            nn.Linear(in_features=options['num_what'], out_features=100, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=options['supervised_categories'], bias=True))

    def forward(self, frame_embeddings):
        return self.supervised_projection(frame_embeddings) if frame_embeddings is not None else None

    def forward_and_compute_supervised_loss(self, embeddings, labels):
        logits = self.forward(embeddings)
        if logits is None:
            return torch.tensor(0., device=self.device), 0.

        lambda_l = self.options['lambda_l']
        loss_sup = F.cross_entropy(logits, labels, reduction='mean')

        loss = lambda_l * loss_sup
        return loss, loss_sup.item()

    def predict(self, frame_embeddings):
        logits = self.supervised_projection(frame_embeddings)
        mask, _, _ = self.compute_mask_of_pixels_to_predict(frame_embeddings)
        supervised_probs = torch.softmax(logits, dim=1) * mask.view(self.w * self.h, 1)

        return supervised_probs, mask, torch.argmax(supervised_probs, dim=1)


class NetWW(nn.Module):

    def __init__(self, options, device, sup_buffer):
        super(NetWW, self).__init__()

        # keeping track of the network options
        self.options = options
        self.device = device

        # encoding splitting
        self.num_what = options['num_what']
        self.num_where = options['num_what']
        self.w = options["w"]
        self.h = options["h"]

        # encoder
        self.encoder = EncoderFactory.createEncoder(options)

        # # decoder
        # self.decoder = DecoderFactory.createDecoder(options)

        # unsupervised module
        self.unsupervised_classifier = UnsupervisedClassifier(options)

        # supervised module
        self.classifier = None
        if options["classifier"] == "neural":
            self.classifier = NeuralOpenSetClassifier(options, sup_buffer, self.device)
        elif options["classifier"] == "NN":
            self.classifier = NNOpenSetClassifier(options, sup_buffer, self.device)
        else:
            raise NotImplementedError

    def forward(self, frame, piggyback_frames):

        # stacking data
        if piggyback_frames is not None:
            frames = torch.cat([frame, piggyback_frames], dim=0)
        else:
            frames = frame

        # encoding data
        encoded_frames, _ = self.encoder(frames)

        # selecting where and what stuff
        whats = encoded_frames[:, :, :, :]

        if self.options['normalize']:
            whats = whats / (torch.norm(whats, dim=1, keepdim=True) + 1e-12)

        # what is done below only consider the current frame and not the piggyback ones
        unsupervised_probs = \
            self.unsupervised_classifier(whats[0, :, :, :].detach().view(self.num_what, -1).t())
        masked_supervised_probs, mask, prediction_idx = \
            self.classifier.predict(whats[0, :, :, :].view(self.num_what, -1).t())

        return whats, unsupervised_probs, masked_supervised_probs, mask, prediction_idx

    def compute_unsupervised_loss(self, whats, activations_foa, activations_foa_prev,
                                  unsupervised_probs, avg_unsupervised_probs,
                                  unsupervised_probs_foa_blob, unsupervised_probs_foa, foa_blob, foa_row_col):

        lambda_c = self.options['lambda_c']
        lambda_e = self.options['lambda_e']
        lambda_s_in = self.options['lambda_s_in']
        lambda_s_out = self.options['lambda_s_out']
        lambda_t = self.options['lambda_t']

        # spatial coherence loss
        loss_s_in, loss_s_out, points_indices_in, points_indices_out = \
            self.__compute_spatial_coherence(whats, foa_blob,
                                             foa_row_col, contrastive=lambda_s_out > 0.,
                                             normalized_data=self.options['normalize'])

        # temporal coherence loss
        if activations_foa is not None and activations_foa_prev is not None:
            if self.options['normalize']:
                loss_t = (1. - torch.sum(activations_foa * activations_foa_prev)) \
                    if activations_foa_prev is not None else torch.tensor(0., device=self.device)
            else:
                loss_t = torch.sum((activations_foa - activations_foa_prev) ** 2) \
                    if activations_foa_prev is not None else torch.tensor(0., device=self.device)
        else:
            loss_t = torch.tensor(0.0, device=self.device)

        # mutual information (squared approximation)
        if unsupervised_probs is not None:
            cond_entropy = -torch.sum(torch.pow(unsupervised_probs, 2)).div(unsupervised_probs.shape[0])
            entropy = -torch.sum(torch.pow(avg_unsupervised_probs, 2))
            loss_mi = -(lambda_e * entropy - lambda_c * cond_entropy)

            # spatial coherence loss on the unsupervised symbols
            if unsupervised_probs_foa_blob is not None and unsupervised_probs_foa is not None:
                loss_smi = torch.sum((unsupervised_probs_foa_blob - unsupervised_probs_foa) ** 2)  # broadcast
            else:
                loss_smi = torch.tensor(0.0, device=self.device)

            # mutual information: useful values to be returned
            pm = unsupervised_probs.shape[1]
            mi = (entropy.item() - cond_entropy.item()) * pm / (pm - 1.)  # [0,1]
            cond_entropy = (cond_entropy.item() + 1.) * (pm / (pm - 1.))  # [0,1]
            entropy = (entropy.item() + 1.) * (pm / (pm - 1.))  # [0,1]
        else:
            cond_entropy = entropy = mi = -1.0
            loss_mi = torch.tensor(0.0, device=self.device)
            loss_smi = torch.tensor(0.0, device=self.device)

        # final loss
        loss = lambda_s_in * loss_s_in + lambda_s_out * loss_s_out + lambda_t * loss_t \
               + loss_mi + lambda_s_in * loss_smi
        return loss, (loss_s_in.item(), loss_s_out.item(), loss_t.item(), loss_mi.item(), cond_entropy, entropy, mi,
                      points_indices_in, points_indices_out)

    def compute_supervised_loss(self, embeddings, labels):
        return self.classifier.forward_and_compute_supervised_loss(embeddings, labels)

    def zero_grad(self):
        for param in self.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param.grad.zero_()

    def print_parameters(self):
        params = list(self.parameters())
        print("Number of tensor params: " + str(len(params)))
        for i in range(0, len(params)):
            p = params[i]
            print("   Tensor size: " + str(p.size()) + " (req. grad = " + str(p.requires_grad) + ")")

    def __sample_points_in_foa_blob(self, n_in, foa_blob, foa_row_col, ensure_foa_is_there=True):
        assert n_in >= 1, "At least two points must be sampled."

        all_indices = torch.arange(self.w * self.h, device=foa_blob.device)
        foa_blob_indices = all_indices[foa_blob.view(-1)]

        if foa_blob_indices.shape[0] > 0:
            perm = torch.randperm(foa_blob_indices.shape[0])
            sampled_indices = foa_blob_indices[perm[:n_in]]

            if ensure_foa_is_there:
                foa_index = foa_row_col[0] * self.w + foa_row_col[1]
                if foa_index not in sampled_indices:
                    sampled_indices[0] = foa_index

            return sampled_indices
        else:
            return None

    def __sample_points_out_of_foa_blob(self, n_out, foa_blob, foa_row_col, spread_factor):
        assert n_out >= 1, "At least one point must be sampled."

        # generating gaussian distribution around focus of attention
        sqrt_area = math.sqrt(torch.sum(foa_blob))
        gaussian_foa = torch.distributions.normal.Normal(torch.tensor([float(foa_row_col[0]), float(foa_row_col[1])]),
                                                         torch.tensor([spread_factor * sqrt_area,
                                                                       spread_factor * sqrt_area]))

        # sampling "n_out" points out of the foa blob
        got_n_out = 0
        points_indices_per_attempt = []

        attempts = 0
        max_attempts = 10

        while got_n_out < n_out and attempts < max_attempts:

            # sampling
            points_row_col = gaussian_foa.sample(torch.Size([n_out - got_n_out])).to(torch.long)

            # avoiding out-of-bound points
            points_row_col = points_row_col[torch.logical_and(
                torch.logical_and(points_row_col[:, 0] >= 0, points_row_col[:, 0] < self.h),
                torch.logical_and(points_row_col[:, 1] >= 0, points_row_col[:, 1] < self.w)), :]

            if points_row_col.shape[0] > 0:
                foa_blob = foa_blob.view(-1)

                # converting to indices
                points_indices = (points_row_col[:, 0] * self.w + points_row_col[:, 1]).to(foa_blob.device)

                # keeping only indices out of the foa_blob
                points_indices = points_indices[~foa_blob[points_indices]]

                # appending the found points to an ad-hoc list
                if points_indices.shape[0] > 0:
                    points_indices_per_attempt.append(points_indices)
                    got_n_out += points_indices.shape[0]

            # counting the number of sampling attempts done so far
            attempts += 1

        # from list to tensor
        if len(points_indices_per_attempt) > 0:
            points_indices_out = torch.cat(points_indices_per_attempt)
        else:
            points_indices_out = None

        return points_indices_out

    def __compute_spatial_coherence(self, full_frame_activations, foa_blob, foa_row_col,
                                    contrastive=True, normalized_data=False):

        num_pairs = self.options['num_pairs']
        if self.options['num_pairs'] < 0:
            n_in = torch.sum(foa_blob)
            if n_in == 0: n_in = 1
            points_indices_in = torch.arange(end=foa_blob.nelement())[foa_blob.flatten()]
        else:
            # in order to get "num_pairs" pairs insider the foa blob,
            # we need to sample (1 + sqrt(1 + 8*num_pairs)) * 0.5 nodes
            n_in = int((1. + math.sqrt(1 + 8 * num_pairs)) * 0.5)  # inside the foa blob (it will yield "num_pairs" edges)

            # sampling
            points_indices_in = self.__sample_points_in_foa_blob(n_in, foa_blob, foa_row_col, ensure_foa_is_there=True)

            # checking

        if points_indices_in is None or points_indices_in.nelement() == 0:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), None, None
        # getting activations
        full_frame_activations = full_frame_activations.view(full_frame_activations.shape[0],
                                                             full_frame_activations.shape[1],
                                                             -1)
        A_in = full_frame_activations[0, :, points_indices_in].t()

        # computing loss
        if not normalized_data:
            A_in_diff = A_in[:, None] - A_in  # difference between all the pairs

            loss_s_in = torch.sum(A_in_diff * A_in_diff) / (n_in * (n_in - 1))
        else:
            loss_s_in = 1.0 - ((torch.sum(torch.matmul(A_in, A_in.t())) - n_in) / (n_in * (n_in - 1)))

        if contrastive:
            if self.options['num_pairs'] < 0:
                points_indices_out = torch.arange(end=foa_blob.nelement())[~foa_blob.flatten()]
            else:
                # in order to get "num_pairs" pairs composed of one element inside the blob and one outside,
                # we need to sample num_pairs / (num_points_inside_foa_blob + 1) points outside the foa blob
                n_out = int(num_pairs / (n_in + 1.))  # outside the foa blob (it will yield "num_pairs" edges)

                # sampling
                points_indices_out = self.__sample_points_out_of_foa_blob(n_out, foa_blob, foa_row_col,
                                                                          spread_factor=self.options["spread_factor"])

                # checking
                if points_indices_out is None:
                    return loss_s_in, torch.tensor(0.0, device=self.device), points_indices_in, None

            # getting activations
            A_out = full_frame_activations[0, :, points_indices_out]

            # computing loss
            if not normalized_data:
                loss_s_out = 1.0 / (torch.mean(torch.sum(A_in * A_in, dim=1, keepdim=True)
                                               - 2.0 * torch.matmul(A_in, A_out)
                                               + torch.sum(A_out * A_out, dim=0, keepdim=True)) + 1e-20)
            else:
                loss_s_out = 1.0 + torch.mean(torch.matmul(A_in, A_out))

            return loss_s_in, loss_s_out, points_indices_in, points_indices_out
        else:
            return loss_s_in, torch.tensor(0., device=self.device), points_indices_in, None
