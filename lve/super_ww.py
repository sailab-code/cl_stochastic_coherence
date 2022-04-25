import torch


class SuperWW:

    def __init__(self, device = torch.device("cpu")):

        self.device = device

        # frame buffer: each element down here is a list of "b" elements - ("b": the number of supervised frames)
        self.frames = []  # each element of this list is 1 x c x h x w
        self.targets = []  # each element is the vector of integer labels associated to labeled pixels of a frame
        self.indices = []  # each element is the vector of indices of the labeled pixels of a frame

        # embedding buffer: each element down here is a list of "n" elements ("n": total number of supervised pixels)
        self.embeddings = []  # each element is the vector of features of a pixel, 1 x e
        self.embeddings_labels = []  # each element is the label (integer) associated to an embedding
        self.embeddings_labels_tensor = []  # tensor-based version of the previous list

        # maps: from frame to embedding and vice-versa
        self.embedding_to_frame = []  # map from embedding ID to frame ID
        self.frame_to_embeddings = []  # map from frame ID to embeddings IDs (each frame can have multiple supervisions)

        # random-shuffle-related stuff
        self.frames_order = None
        self.frames_order_last_id = -1  # it must start at -1 (it is incremented before being used)
        self.embeddings_order = None
        self.embeddings_order_last_id = -1  # it must start at -1 (it is incremented before being used)

        # status of frame sampler
        self.__last_sampled_frames = None  # indices
        self.__batches_including_last_supervised_frame = 0

        # status of embedding sampler
        self.__batches_including_last_supervised_embeddings = 0  # number of batches that included the last received supervisions
        self.__supervised_embeddings_last_id = 0  # index

    def add(self, frame, targets, indices, all_pixels_embeddings, frame_was_already_added=False):
        """Add the supervision-data associated to a certain frame to the supervision buffer.

        Args:
            frame (torch.Tensor): the considered frame (1 x c x h x w).
            targets (torch.LongTensor): the vector with the class labels (>= 0 and up to 255 - excluded) associated to
                the supervised pixes (it can be None).
            indices (torch.LongTensor): the vector with the indices of the supervised pixels (pixels are indexed from 0
                to hw, row-wise) (it can be None).
            all_pixels_embeddings (torch.FloatTensor): the embeddings of all the pixels of the frame (1 x e x h x w).
            frame_was_already_added (bool): flag to indicate that we already added some supervisions for the current
                frame, and we are simply adding new supervisions we want to merge with the already given ones.

        Returns:
            True, if supervisions were added to the buffer (False, otherwise).
        """
        if targets is None or indices is None:
            return False

        # if we are adding this frame for the first time, not much to do...
        # ...otherwise, we have to merge the existing supervisions with the newly provided ones
        if not frame_was_already_added:
            self.frames.append(frame.to(self.device))  # device
            self.targets.append(targets.to(self.device))  # device
            self.indices.append(indices.to(self.device))  # device
        else:
            wh = frame.shape[2] * frame.shape[3]
            targets_mask = torch.ones(wh, dtype=torch.long, device=self.device) * 255  # device (assuming 255 to be a no-class label)
            targets_mask[self.indices[-1]] = self.targets[-1]
            targets_mask[indices] = targets
            valid_targets = targets_mask < 255
            self.targets[-1] = targets_mask[valid_targets]
            self.indices[-1] = torch.arange(wh)[valid_targets]

            # altering some elements to force a full restore of all the embeddings of this frame
            targets = self.targets[-1]
            indices = self.indices[-1]
            embeddings_to_remove = len(self.frame_to_embeddings.pop())
            for i in range(0, embeddings_to_remove):
                self.embeddings.pop()
                self.embeddings_labels.pop()

        # storing embeddings
        num_sup_last_frame = torch.numel(indices)
        e = all_pixels_embeddings.shape[1]  # length of each embedding
        last_frame_frame_id = len(self.frames) - 1
        self.frame_to_embeddings.append([])

        for i in range(0, num_sup_last_frame):
            self.embeddings.append(all_pixels_embeddings.view(1, e, -1)[:, :, indices[i]].to(self.device))  # device
            self.embeddings_labels.append(targets[i].to(self.device))  # device

            last_embedding_id = len(self.embeddings) - 1
            self.embedding_to_frame.append(last_frame_frame_id)
            self.frame_to_embeddings[last_frame_frame_id].append(last_embedding_id)

        # updating the tensor-based version of the label list
        self.embeddings_labels_tensor = torch.stack(self.embeddings_labels)

        # random permutation, not considering the last added frame/supervisions
        self.__shuffle_frames(len(self.frames))
        self.__shuffle_embeddings(len(self.embeddings))

        # resetting the state of the frame sampler
        self.__batches_including_last_supervised_frame = 0

        # resetting the state of the embedding sampler
        self.__batches_including_last_supervised_embeddings = 0
        self.__supervised_embeddings_last_id = 0
        return True

    def sample_embeddings_batch(self, batch_size, last_supervision_persistence):
        n = len(self.embeddings)
        if n == 0:
            return None, None

        # fixing
        if batch_size > n:
            batch_size = n

        # here we decide if we are going to force the sampling of the last supervisions or not
        last_supervised_embedding_ids = self.frame_to_embeddings[-1]
        if self.__batches_including_last_supervised_embeddings < last_supervision_persistence:
            include_last_supervised_embeddings = True
            self.__batches_including_last_supervised_embeddings += 1
        else:
            include_last_supervised_embeddings = False
        got_enough = False

        # packing mini-batch (always adding the supervisions of the last frame)
        batch_x = [None] * batch_size
        batch_y = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # device

        for i in range(0, batch_size):

            # adding the supervisions of the last supervised frame
            if include_last_supervised_embeddings and not got_enough:
                j = last_supervised_embedding_ids[self.__supervised_embeddings_last_id]

                self.__supervised_embeddings_last_id += 1
                if self.__supervised_embeddings_last_id >= len(last_supervised_embedding_ids):
                    self.__supervised_embeddings_last_id = 0
                    got_enough = True
            else:

                # the following elements of the batch are randomly sampled
                j = self.embeddings_order[self.embeddings_order_last_id]
                self.embeddings_order_last_id += 1
                if self.embeddings_order_last_id >= n:
                    self.__shuffle_embeddings(n)  # re-shuffle and restart

                # if we already added the supervisions of the last frame in the mini-batch, we have to skip them now!
                if include_last_supervised_embeddings:
                    while j in last_supervised_embedding_ids:
                        j = self.embeddings_order[self.embeddings_order_last_id]
                        self.embeddings_order_last_id += 1
                        if self.embeddings_order_last_id >= n:
                            self.__shuffle_embeddings(n)

            batch_x[i] = self.embeddings[j]  # randomly selected supervisions
            batch_y[i] = self.embeddings_labels[j]

        batch_x = torch.cat(batch_x, dim=0)
        return batch_x, batch_y

    def sample_frames(self, batch_size, last_supervised_frame_persistence):
        b = len(self.frames)
        if b == 0:
            return None

        # fixing
        if batch_size > b:
            batch_size = b

        # here we decide if we are going to force the sampling of the last supervised frame or not
        last_supervised_frame_id = len(self.frames) - 1
        if self.__batches_including_last_supervised_frame < last_supervised_frame_persistence:
            include_last_supervised_frame = True
            self.__batches_including_last_supervised_frame += 1
        else:
            include_last_supervised_frame = False
        got_enough = False

        # sampling
        self.__last_sampled_frames = [None] * batch_size
        batch = [None] * batch_size
        for i in range(0, batch_size):

            # the first element of the batch is the last supervised frame (if needed)
            if include_last_supervised_frame and not got_enough:
                j = last_supervised_frame_id
                got_enough = True
            else:

                # the following elements of the batch are randomly sampled
                j = self.frames_order[self.frames_order_last_id]
                self.frames_order_last_id += 1
                if self.frames_order_last_id >= b:
                    self.__shuffle_frames(b)  # re-shuffle and restart

                # if we already added the last supervised frame in the mini-batch, we have to skip it now!
                if include_last_supervised_frame:
                    while j == last_supervised_frame_id:
                        j = self.frames_order[self.frames_order_last_id]
                        self.frames_order_last_id += 1
                        if self.frames_order_last_id >= b:
                            self.__shuffle_frames(b)

            batch[i] = self.frames[j]
            self.__last_sampled_frames[i] = j

        # packing
        return torch.cat(batch, dim=0)

    def update_embeddings_of_sampled_frames(self, all_pixels_embeddings):
        if all_pixels_embeddings is None:
            return

        e = all_pixels_embeddings.shape[1]  # number of features in a single embedding

        k = 0  # k: frame index 0,1,...,n
        for j in self.__last_sampled_frames:  # j: frame index 7,2,...,4
            embeddings_ids = self.frame_to_embeddings[j]
            indices = self.indices[j]

            for i in range(0, len(embeddings_ids)):
                self.embeddings[embeddings_ids[i]] = \
                    all_pixels_embeddings[k, :, :, :].view(e, -1)[:, indices[i]].unsqueeze(0).to(self.device)  # device

            k += 1

    def get_last_frame_targets(self):
        return self.targets[-1]

    def get_last_frame_indices(self):
        return self.indices[-1]

    def get_embeddings(self):
        return self.embeddings

    def get_embeddings_labels(self):
        return self.embeddings_labels, self.embeddings_labels_tensor

    def detach_last_frame_supervisions(self):
        ids = self.frame_to_embeddings[-1]
        for _id in ids:
            self.embeddings[_id] = self.embeddings[_id].detach()

    def __shuffle_frames(self, size):
        self.frames_order = torch.randperm(size)
        self.frames_order_last_id = 0

    def __shuffle_embeddings(self, size):
        self.embeddings_order = torch.randperm(size)
        self.embeddings_order_last_id = 0
