import os
import numpy as np
from random import randint, uniform, randrange
import lve
import torch
import cv2
import time

from collections import OrderedDict


class WorkerWW(lve.Worker):

    def __init__(self, w, h, c, fps, options):
        super().__init__(w, h, c, fps, options)  # do not forget this

        # if the device name ends with 'b' (e.g., 'cpub'), the torch benchmark mode is activated (usually keep it off)
        if options["device"][-1] == 'b':
            self.device = torch.device(options["device"][0:-1])
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device(options["device"])

        # enforcing a deterministic behaviour, when possible
        # torch.set_deterministic(True)

        # setting up seeds for random number generators
        seed = int(time.time()) if options["seed"] < 0 else int(options["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)

        # registering supported commands (commands that are triggered by the external visualizer/interface)
        self.register_command("reset_foa", self.__handle_command_reset_foa)
        self.register_command("supervise", self.__handle_command_supervise)

        # saving a shortcut to the neural network options
        self.net_options = self.options["net"]
        self.net_options["w"] = self.w
        self.net_options["h"] = self.h

        # defining processors
        self.optical_flow = lve.OpticalFlowCV()
        self.geymol = lve.GEymol(self.options["foa"], self.device) if self.options["foa"] is not None else None
        self.sup_buffer = lve.SuperWW(device=self.device)
        self.net = lve.NetWW(self.net_options, self.device, self.sup_buffer).to(self.device)

        # neural network optimizer
        self.__lr = self.net_options["step_size"]
        if self.__lr < 0.:  # hack
            self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=-self.__lr)
        else:
            self.net_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.__lr)
        self.__freeze = self.net_options["freeze"]
        if self.__freeze:
            self.net.requires_grad_(False)

        # setting up initial supervision map (if any)
        self.augment_supervision_map(self.options["supervision_map"], self.net_options["supervised_categories"])

        # misc
        self.__ids = torch.arange(self.w * self.h, device=self.device)
        self.__activations_foa_prev = None
        self.__avg_unsupervised_probs = None
        self.__sup_added = False
        self.__sup_new_category_added = False
        self.__frame = None
        self.__frame_embeddings = None
        self.__motion_needed = (self.geymol is not None and self.geymol.parameters["alpha_of"] > 0.)
        self.__unsup_loss = torch.tensor(-1.).to(self.device)
        self.__sup_loss = torch.tensor(-1.).to(self.device)
        self.__stats = OrderedDict([('loss', -1.), ('loss_s_in', -1.), ('loss_s_out', -1.), ('loss_t', -1.),
                                    ('loss_mi', -1.), ('loss_sup', -1.), ('cond_entropy', -1.), ('entropy', -1.),
                                    ('mi', -1.), ('foax', -1.), ('foay', -1.),
                                    ('foavx', -1.), ('foavy', -1.), ('saccade', -1)])  # placeholders
        self.__what = None
        self.__foa = None

    def process_frame(self, frame, of=None, supervisions=None, foa=None):

        # data returned by the call below and their types are:
        # frame: input frame (torch tensor, 1 x c x h x w, better keep the dummy batch dimension here),
        # motion: optical flow (torch tensor, 1 x 2 x h x w, better keep the dummy batch dimension here),
        # foa_row_col: focus of attention - integer coordinates (torch tensor, 2 elements)
        # saccade_bool: flag that tell if the movement of the focus of attention is a saccade (bool)
        # foa: focus of attention - real x, y, velocity_x, velocity_y (list of 4 elements)
        # sups: a pair of torch tensors (targets, indices), or (None, None)
        frame, motion, foa_row_col, saccade_bool, foa, sup_targets, sup_indices = \
            self.__compute_missing_data_and_convert(frame, of, foa, supervisions)

        # we override saccades with a flag that is only about movement
        saccade_bool = torch.norm(motion[0, :, foa_row_col[0], foa_row_col[1]]) == 0

        # normalization (customizable)
        frame = (frame - 0.5) / 0.25

        # sampling a/some buffered frame(s) (associated to previous supervisions - if any)
        buff_frames = self.sup_buffer.sample_frames(self.options['piggyback_frames'],
                                                    self.options['sup_persistence'])
        buff_frames = buff_frames.to(self.device) if buff_frames is not None else None

        # inference (full inference on the current frame, encoding only on the buffered frames - if any)
        # reconstruction, unsupervised_probs, supervised_logits: they are only about the current frame
        # what, where: they are about the current frame and the buffered frames (if any),
        #   the current frame and the buffered frames constitute a batch of frames (index 0: current frame)
        whats, unsupervised_probs, supervised_probs_masked_list, prediction_mask_list, prediction_idx_list = \
            self.net(frame, buff_frames)

        # # TODO move from here
        # supervised_probs_masked_tensor = supervised_probs_masked_list
        # prediction_mask_tensor = prediction_mask_list
        # prediction_idx_tensor = prediction_idx_list

        # data to be buffered or updated on the buffers
        piggyback_frame_embeddings = whats[1:, :, :, :]
        self.__frame_embeddings = whats[0, None]

        # updating embeddings of the supervised pixel(s) of the buffered frames (if any)
        self.sup_buffer.update_embeddings_of_sampled_frames(piggyback_frame_embeddings.detach()
                                                            if piggyback_frame_embeddings.shape[0] > 0 else None)

        # storing supervisions associated to the current frame, provided by the input stream (if any)
        self.__sup_added = self.sup_buffer.add(frame, sup_targets, sup_indices, self.__frame_embeddings)

        # updating the worker-level information about supervision (worker has its own counts, exposed to the visualizer)
        self.__update_internal_supervision_counts()

        # computing the motion-based 'blob' around the focus of attention
        foa_blob = self.__compute_foa_moving_blob(motion, foa_row_col,
                                                  motion_threshold=self.options['motion_threshold'])

        # getting unsupervised probabilities associated to the focus-of-attention-blob
        unsupervised_probs_foa_blob = unsupervised_probs[None, foa_blob.view(-1), :]

        # getting the activation associated to the focus-of-attention-coordinates at every layer
        activations_foa = whats[None, 0, :, foa_row_col[0], foa_row_col[1], None]

        # take unsup_prob on the foa for the current frame
        unsupervised_probs_foa = unsupervised_probs[None, foa_row_col[0] * self.w + foa_row_col[1], :]

        # previous activations on the foa coordinates (if available)
        activations_foa_prev = \
            self.__activations_foa_prev if self.__activations_foa_prev is not None else activations_foa.detach()

        # updating the average probabilities of the unsupervised categories (over time - moving average)
        if not saccade_bool:
            avg_unsupervised_probs = self.__update_avg_unsupervised_probs_over_time(unsupervised_probs_foa)

        self.__unsup_loss, unsup_details = \
            self.net.compute_unsupervised_loss(self.__frame_embeddings,
                                               activations_foa if not saccade_bool else None,
                                               activations_foa_prev if not saccade_bool else None,
                                               unsupervised_probs_foa if not saccade_bool else None,
                                               avg_unsupervised_probs if not saccade_bool else None,
                                               unsupervised_probs_foa_blob if not saccade_bool else None,
                                               unsupervised_probs_foa if not saccade_bool else None,
                                               foa_blob, foa_row_col)

        # sampling supervised data (if any)
        embeddings, labels = self.sup_buffer.sample_embeddings_batch(self.options['sup_batch'],
                                                                     self.options['sup_persistence'])
        embeddings = embeddings.to(self.device) if embeddings is not None else None
        labels = labels.to(self.device) if labels is not None else None

        # loss (supervised)
        self.__sup_loss, sup_details = self.net.compute_supervised_loss(embeddings, labels)

        # saving output data related to the current frame
        foa_blob = lve.utils.torch_float_01_to_np_uint8(foa_blob.to(torch.float))

        unsupervised_probs_idx = torch.argmax(unsupervised_probs, dim=1)
        unsupervised_probs = unsupervised_probs.detach().t(). \
            view(1, unsupervised_probs.shape[1], self.h, self.w).cpu().numpy()

        # TODO we pass only the first prediction from the threshold list -- to revise
        supervised_probs_masked_list = [el.detach().t(). \
            view(1, el.shape[1], self.h, self.w).cpu().numpy() for el in supervised_probs_masked_list]

        prediction_idx_list_detached = [el.detach().cpu().numpy() for el in prediction_idx_list]

        prediction_mask = lve.utils.torch_float_01_to_np_uint8(prediction_mask_list[0].view(self.h, self.w).to(torch.float))
        sup_targets = self.sup_buffer.get_last_frame_targets().cpu().numpy().astype(
            np.uint32) if self.__sup_added else None
        sup_indices = self.sup_buffer.get_last_frame_indices().cpu().numpy().astype(
            np.uint32) if self.__sup_added else None

        sampled_points = np.zeros((self.w * self.h, 3), dtype=np.uint8)
        points_indices_in = unsup_details[7]
        points_indices_out = unsup_details[8]
        if points_indices_in is not None:
            points_indices_in = points_indices_in.cpu().numpy()
            sampled_points[points_indices_in, 0] = 255
            sampled_points[points_indices_in, 1] = 0
            sampled_points[points_indices_in, 2] = 0
        if points_indices_out is not None:
            points_indices_out = points_indices_out.cpu().numpy()
            sampled_points[points_indices_out, 0] = 0
            sampled_points[points_indices_out, 1] = 0
            sampled_points[points_indices_out, 2] = 255
        sampled_points = sampled_points.reshape(self.h, self.w, 3)
        sampled_points = cv2.blur(sampled_points, (5, 5))
        sampled_points = sampled_points.reshape(self.h * self.w, 3)
        sampled_points[sampled_points[:, 0] > 0, 0] = 255
        sampled_points[sampled_points[:, 1] > 0, 1] = 255
        sampled_points[sampled_points[:, 2] > 0, 2] = 255
        sampled_points = sampled_points.reshape(self.h, self.w, 3)

        self.__stats.update({"loss": self.__unsup_loss.item() + self.__sup_loss.item(),
                             "loss_s_in": unsup_details[0],
                             "loss_s_out": unsup_details[1],
                             "loss_t": unsup_details[2],
                             "loss_mi": unsup_details[3],
                             "loss_sup": sup_details,
                             "cond_entropy": unsup_details[4],
                             "entropy": unsup_details[5],
                             "mi": unsup_details[6],
                             "foax": foa[0], "foay": foa[1],
                             "foavx": foa[2], "foavy": foa[3],
                             "saccade": int(saccade_bool)})

        what_numpy = whats[0, None].detach().cpu().numpy()
        self.add_outputs({"motion": of[0],  # bin
                          "what": what_numpy,  # bin
                          "blob": foa_blob,  # image
                          "sampled-points": sampled_points,  # image
                          "unsup-probs": unsupervised_probs,  # bin
                          "unsup-probs_idx": unsupervised_probs_idx.detach().cpu().numpy(),
                          "sup-probs": supervised_probs_masked_list[0],  # bin
                          "sup-probs-list": supervised_probs_masked_list,  # bin
                          "prediction_idx": prediction_idx_list_detached[0],
                          "prediction_idx-list": prediction_idx_list_detached,
                          "pred-mask": prediction_mask,  # bin
                          "sup.targets": sup_targets,  # bin
                          "sup.indices": sup_indices,  # bin
                          "sup.map": self.get_supervision_map() if self.__sup_new_category_added else None,  # JSON
                          "stats.worker": self.__stats,  # dictionary
                          "logs.worker": list(self.__stats.values()),  # CSV log
                          "tb.worker": self.__stats}, batch_index=0)  # tensorboard

        # storing data to be used in the next frame or needed to handle a supervision given through the visualizer
        self.__activations_foa_prev = activations_foa.detach()
        if not saccade_bool:
            self.__avg_unsupervised_probs = avg_unsupervised_probs.detach()
        self.__sup_new_category_added = False
        self.__frame = frame
        self.__what = what_numpy
        self.__foa = (foa[0], foa[1])

    def update_model_parameters(self):

        # check if the freeze option was changed while the agent was running (and react to such change)
        if self.__freeze != self.net_options['freeze']:
            self.__freeze = self.net_options['freeze']
            if self.__freeze:
                self.net.requires_grad_(False)
            else:
                self.net.requires_grad_(True)  # we still have not computed gradients, better 'return'
            return

        # if frozen, nothing to do here
        if self.__freeze:
            return

        # completing the loss function
        loss = self.__unsup_loss + self.__sup_loss

        if loss > 0:

            # computing gradients
            loss.backward()

            # update step
            self.net_optimizer.step()
            self.net.zero_grad()

        # detaching last frame supervisions (if any)
        if self.__sup_added:
            self.sup_buffer.detach_last_frame_supervisions()

        # check if learning rate was changed (hot)
        if self.__lr != self.net_options['step_size']:
            self.__lr = self.net_options['step_size']
            if self.__lr < 0.:
                self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=-self.__lr)
            else:
                self.net_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.__lr)

    def load(self, model_folder):
        worker_model_folder = model_folder + os.sep

        # loading neural network weights
        self.net.load_state_dict(torch.load(worker_model_folder + "net.pth", map_location=self.device))

        # loading worker-status related tensors
        worker_status = torch.load(worker_model_folder + "worker.pth", map_location=self.device)
        self.__activations_foa_prev = worker_status["what_for_prev"]
        self.__avg_unsupervised_probs = worker_status["avg_unsupervised_probs"]
        self.sup_buffer = worker_status["supervision_buffer"]

        # loading worker-status parameters from a readable JSON
        params = lve.utils.load_json(worker_model_folder + "worker.json")

        # setting up the internal elements using the JSON-loaded parameters
        if self.geymol is not None:
            self.geymol.reset(params["foa_y"], params["foa_t"])
            self.geymol.first_call = False
            self.geymol.IOR_matrix = worker_status["foa_ior_matrix"]

        self.augment_supervision_map(params["supervision_map"],
                                     self.net_options["supervised_categories"],
                                     counts=params["supervision_count"])

        if self.options["supervision_map"] is not None and len(self.options["supervision_map"]) > 0:
            print("WARNING: the provided supervision map will be overwritten by the one loaded from disk!")

    def save(self, model_folder):
        worker_model_folder = model_folder + os.sep
        if not os.path.exists(worker_model_folder):
            os.makedirs(worker_model_folder)

        # saving neural network weights
        torch.save(self.net.state_dict(), worker_model_folder + "net.pth")

        # saving worker-status related tensors
        torch.save({"what_for_prev": self.__activations_foa_prev,
                    "avg_unsupervised_probs": self.__avg_unsupervised_probs,
                    "supervision_buffer": self.sup_buffer,
                    "foa_ior_matrix": self.geymol.IOR_matrix if self.geymol is not None else torch.zeros(
                        (self.h, self.w), dtype=torch.float32, device=self.device)},
                   worker_model_folder + "worker.pth")

        # saving other parameters
        lve.utils.save_json(worker_model_folder + "worker.json",
                            {"foa_y": list(self.geymol.y) if self.geymol is not None else [0., 0., 0., 0.],
                             "foa_t": self.geymol.t if self.geymol is not None else 0,
                             "supervision_map": self.get_supervision_map(),
                             "supervision_count": self.get_supervision_count()})

    def get_output_types(self):
        output_types = {  # the output element "frames" is already registered by default
            "motion": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "what": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "blob": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "sampled-points": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "unsup-probs": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "unsup-probs_idx": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup-probs": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup-probs-list": {'data_type': lve.OutputType.PRIVATE, 'per_frame': True},
            "prediction_idx": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "prediction_idx-list": {'data_type': lve.OutputType.PRIVATE, 'per_frame': True},
            "pred-mask": {'data_type': lve.OutputType.IMAGE, 'per_frame': True},
            "sup.indices": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup.targets": {'data_type': lve.OutputType.BINARY, 'per_frame': True},
            "sup.map": {'data_type': lve.OutputType.JSON, 'per_frame': False},
            "stats.worker": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "logs.worker": {'data_type': lve.OutputType.TEXT, 'per_frame': False},
            "logs.worker__header": ['frame'] + list(self.__stats.keys())  # first line of CSV
        }

        return output_types

    def print_info(self):
        s = "   worker {"
        i = 0
        for k, v in self.__stats.items():
            s += (k + (": {0:.3e}".format(v) if abs(v) >= 1000 else ": {0:.3f}".format(v)))
            if (i + 1) % 7 == 0:
                if i < len(self.__stats) - 1:
                    s += ",\n           "
                else:
                    s += "}"
            else:
                if i < len(self.__stats) - 1:
                    s += ", "
                else:
                    s += "}"
            i += 1

        print(s)

    def __handle_command_reset_foa(self, command_value, batch_index=0):
        if batch_index:
            pass
        if self.geymol is not None:
            self.geymol.reset([command_value['y'], command_value['x'],
                               2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1)),
                               2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))])
            return True
        else:
            return False

    def __handle_command_supervise(self, command_value, batch_index=0):

        # here is the thing: this function accepts supervision given on a bounding box,
        # where (row, col) is the upper-left coordinate of the box of size (n_rows, n_cols).
        # of course, when a single pixel is supervised, we have (n_rows = 1, n_cols = 1)
        class_name = command_value['class'].lower().strip()
        row = command_value['y']
        col = command_value['x']
        n_cols = command_value['w']
        n_rows = command_value['h']
        n_sup = n_cols * n_rows  # total number of supervised pixels

        # from class name to numerical index (it will also create a new category - for never-seen-before class names)
        target, new_class_added = self.get_target(class_name, self.net_options["supervised_categories"])

        # checking if the system was unable to add any more classes (maximum number of categories has been reached)
        if target is None:
            return False

        # marking the flag that tells that a new category was added (then we will save the updated class map to disk)
        self.__sup_new_category_added = self.__sup_new_category_added or new_class_added

        # preparing supervision data: targets, indices
        # here we have to get the indices of the pixels in the supervised box
        # boxes are expected to be relatively small, so this procedure will be fast
        if n_sup > 1:
            list_of_indices_on_each_row = [None] * n_rows
            for r in range(0, n_rows):
                list_of_indices_on_each_row[r] = torch.arange(n_cols, dtype=torch.long) + (row + r) * self.w + col
            indices = torch.cat(list_of_indices_on_each_row)  # CPU
            targets = torch.ones(n_sup, dtype=torch.long) * target  # CPU
        else:
            indices = torch.ones(1, dtype=torch.long) * row * self.w + col  # CPU
            targets = torch.ones(1, dtype=torch.long) * target  # CPU

        # flag to mark whether this frame has already received supervisions (from input stream of from the visualizer)
        frame_was_already_supervised_before = self.__sup_added

        # buffering (and merging/intersecting) supervisions (warning: supervisions can be overwritten)
        self.__sup_added = self.sup_buffer.add(self.__frame, targets, indices, self.__frame_embeddings,
                                               frame_was_already_added=frame_was_already_supervised_before)

        # saving/updating internal worker stats (after having buffered/intersected them)
        self.__update_internal_supervision_counts(frame_was_already_considered=frame_was_already_supervised_before)

        # sampling or re-sampling (it the batch was already sampled in other methods on in another call to this method)
        embeddings, labels = self.sup_buffer.sample_embeddings_batch(self.options['sup_batch'],
                                                                     self.options['sup_persistence'])
        embeddings = embeddings.to(self.device) if embeddings is not None else None
        labels = labels.to(self.device) if labels is not None else None

        # loss (supervised)
        self.__sup_loss, sup_details = self.net.compute_supervised_loss(embeddings, labels)

        # adding/updating supervision-related output
        sup_targets = self.sup_buffer.get_last_frame_targets().cpu().numpy().astype(np.uint32) if self.__sup_added else None
        sup_indices = self.sup_buffer.get_last_frame_indices().cpu().numpy().astype(np.uint32) if self.__sup_added else None

        self.__stats.update({"loss": self.__unsup_loss.item() + self.__sup_loss.item(),
                             "loss_sup": sup_details})

        self.add_outputs({"sup.targets": sup_targets,  # bin
                          "sup.indices": sup_indices,  # bin
                          "sup.map": self.get_supervision_map() if self.__sup_new_category_added else None,  # JSON
                          "stats.worker": self.__stats})  # dictionary

    def __compute_missing_data_and_convert(self, batch_frames_np_uint8, batch_motion_np_float32,
                                           batch_foa_np_float32, batch_sup_np):

        # assumption: data are stored in batches of size 1, i.e., one frame at each time instant
        assert len(batch_frames_np_uint8) == 1, "We are assuming to deal with batches of size 1, " \
                                                "and it does not seem to be the case here!"

        # convert to tensor
        frame_np_uint8 = batch_frames_np_uint8[0]
        frame = lve.utils.np_uint8_to_torch_float_01(frame_np_uint8, device=self.device)

        # grayscale-instance of the input frame
        if not self.frame_is_gray_scale:
            frame_gray_np_uint8 = cv2.cvtColor(frame_np_uint8, cv2.COLOR_BGR2GRAY).reshape(self.h, self.w, 1)
            frame_gray = lve.utils.np_uint8_to_torch_float_01(frame_gray_np_uint8, device=self.device)
        else:
            frame_gray_np_uint8 = frame_np_uint8
            frame_gray = frame

        # optical flow
        if batch_motion_np_float32 is None or batch_motion_np_float32[0] is None:
            if self.__motion_needed:
                motion_np_float32 = self.optical_flow(frame_gray_np_uint8)  # it returns np.float32, h x w x 2
            else:
                motion_np_float32 = np.zeros((self.h, self.w, 2), dtype=np.float32)
            motion = lve.utils.np_float32_to_torch_float(motion_np_float32, device=self.device)  # 1 x 2 x h x w

            if batch_motion_np_float32 is not None:
                batch_motion_np_float32[0] = motion_np_float32  # updating, it might be used out of this function
        else:
            motion_np_float32 = batch_motion_np_float32[0]  # h x w x 2
            motion = lve.utils.np_float32_to_torch_float(motion_np_float32, device=self.device)  # 1 x 2 x h x w

        # focus of attention (a negative dissipation triggers a random trajectory, useful in some experiments)
        if batch_foa_np_float32 is None or batch_foa_np_float32[0] is None:
            if self.geymol is not None:
                if self.options["foa"]["dissipation"] >= 0.0:
                    foa, saccade_bool = self.geymol.next_location(frame_gray, motion,
                                                                  frame_gray_uint8_cpu=frame_gray_np_uint8)
                else:
                    foa = np.array([randrange(0, self.h - 1), randrange(0, self.w - 1), 0., 0.], dtype=np.float64)
                    saccade_bool = False
            else:
                foa = np.array([0., 0., 0., 0.], dtype=np.float64)
                saccade_bool = False
        else:
            foa = batch_foa_np_float32[0][0:4]
            saccade_bool = bool(batch_foa_np_float32[0][-1])

        # getting the two integer coordinates of the focus of attention (discarding velocity)
        foa_row_col = torch.from_numpy(foa[0:2].astype(np.long)).to(self.device).view(2).to(torch.long)
        # a pair of (targets, indices)
        if batch_sup_np is not None and batch_sup_np[0] is not None:
            sup_targets, sup_indices = torch.tensor(batch_sup_np[0][0], dtype=torch.int64), torch.tensor(
                batch_sup_np[0][1])  # CPU
        else:
            sup_targets, sup_indices = None, None

        return frame, motion, foa_row_col, saccade_bool, foa, sup_targets, sup_indices

    def __compute_foa_moving_blob(self, motion, foa_row_col, motion_threshold=0.1):

        if motion_threshold < 0:
            moving_bool = torch.sum(motion ** 2, dim=1).view(self.h, self.w) > 0
            if moving_bool[foa_row_col[0], foa_row_col[1]]:
                return moving_bool
            else:
                return torch.zeros((self.h, self.w), dtype=torch.bool, device=motion.device)

        # boolean mask of those pixels that are moving
        moving_bool = torch.sum(motion ** 2, dim=1).view(-1) > (motion_threshold * motion_threshold)

        # boolean mask (flat) that, at the end of this function, will indicate the pixels belonging to the blob
        blob_bool = torch.zeros(self.h * self.w, dtype=torch.bool, device=motion.device)

        if not moving_bool[foa_row_col[0] * self.w + foa_row_col[1]]:
            return blob_bool.view(self.h, self.w)

        # first of all, we mark the current position of the focus of attention as belonging to the blob
        frontier = foa_row_col[0] * self.w + foa_row_col[1]
        blob_bool[frontier] = True

        while torch.numel(frontier) > 0:
            # expanding the frontier
            frontier_neighs = self.__get_neighbors(frontier).view(-1)

            # removing repeated elements
            frontier_neighs = torch.unique(frontier_neighs)

            # keeping only new elements that are moving
            frontier = torch.masked_select(frontier_neighs,
                                           (blob_bool[frontier_neighs] == 0) * (moving_bool[frontier_neighs] != 0))

            # expanding the blob
            blob_bool[frontier] = True

        # reshaping
        return blob_bool.view(self.h, self.w)

    def __get_neighbors(self, _pixel_ids):
        _w = self.w
        _h = self.h

        # boolean masks of pixels for which it is not valid to get the pixel on top, bottom, left, right, ...
        no_top = _pixel_ids < _w
        no_bottom = _pixel_ids >= (_w * (_h - 1))
        no_left = _pixel_ids % _w == 0
        no_right = (_pixel_ids + 1) % _w == 0
        no_top_left = no_top + no_left
        no_top_right = no_top + no_right
        no_bottom_left = no_bottom + no_left
        no_bottom_right = no_bottom + no_right

        # 8-by-n matrix, where 'n' is the number of pixels identifiers
        neighbors = torch.stack([_pixel_ids - _w - 1,
                                 _pixel_ids - _w,
                                 _pixel_ids - _w + 1,
                                 _pixel_ids - 1,
                                 _pixel_ids + 1,
                                 _pixel_ids + _w - 1,
                                 _pixel_ids + _w,
                                 _pixel_ids + _w + 1])

        # fixing borders (replacing invalid neighbors with the identifier of the original pixels)
        neighbors[0, no_top_left] = _pixel_ids[no_top_left]
        neighbors[1, no_top] = _pixel_ids[no_top]
        neighbors[2, no_top_right] = _pixel_ids[no_top_right]
        neighbors[3, no_left] = _pixel_ids[no_left]
        neighbors[4, no_right] = _pixel_ids[no_right]
        neighbors[5, no_bottom_left] = _pixel_ids[no_bottom_left]
        neighbors[6, no_bottom] = _pixel_ids[no_bottom]
        neighbors[7, no_bottom_right] = _pixel_ids[no_bottom_right]

        return neighbors

    def __update_internal_supervision_counts(self, frame_was_already_considered=False):
        if self.__sup_added:
            if not frame_was_already_considered:
                self.create_supervision_count_checkpoint()
            else:
                self.restore_supervision_count_checkpoint()

            found_targets, targets_count = torch.unique(self.sup_buffer.get_last_frame_targets(), return_counts=True)
            for k in range(0, torch.numel(found_targets)):
                target = found_targets[k].item()
                assert self.get_class_name(target) is not None, \
                    ("ERROR: Target " + str(target) + " is not associated to any class names!")
                self.increment_supervision_count(target, num_sup=targets_count[k].item())

    def __update_avg_unsupervised_probs_over_time(self, unsupervised_probs):
        zeta = self.options['mi_history_weight']

        # moving average, weighed by 'zeta'
        if self.__avg_unsupervised_probs is not None and 0. <= zeta < 1.:
            avg_unsupervised_probs = zeta * torch.mean(unsupervised_probs, dim=0) \
                                     + (1.0 - zeta) * self.__avg_unsupervised_probs
        else:
            avg_unsupervised_probs = torch.mean(unsupervised_probs, dim=0)

        return avg_unsupervised_probs
