import os
import numpy as np
import random
from random import randint, uniform
import lve
import torch
import cv2


class WorkerBasic(lve.Worker):

    def __init__(self, w, h, c, fps, options):
        super().__init__(w, h, c, fps, options)  # do not forget this
        self.device = torch.device(options["device"] if "device" in options else "cpu")  # device

        # registering supported commands
        self.register_command("reset_foa", self.__handle_command_reset_foa)

        # model parameters
        self.dummy_weights = np.array([2.0*np.random.rand(4,self.c,9).astype(np.float32)-1.0,
                                       2.0*np.random.rand(3,4,25).astype(np.float32)-1.0], dtype=object)  # random
        self.rho = self.options["rho"]

        # processors
        self.blur_processor = lve.BlurCV(self.w, self.h, self.c, self.device)
        self.optical_flow_processor = lve.OpticalFlowCV()
        self.foa_processor = lve.GEymol(self.options["foa"], self.device)

    def process_frame(self, frame_numpy_uint8, of=None, supervisions=None):

        # blurring
        frame_numpy_uint8 = frame_numpy_uint8[0]  # batched data
        frame_numpy_uint8 = self.blur_processor(frame_numpy_uint8, blur_factor=1.0 - self.rho)  # it returns np.float32
        frame_numpy_uint8 = frame_numpy_uint8.astype(np.uint8)  # keeping np.uint8 format

        # grayscale-instance of the (blurred) input frame
        if frame_numpy_uint8.ndim == 3 and frame_numpy_uint8.shape[2] == 3:
            frame_gray_numpy_uint8 = cv2.cvtColor(frame_numpy_uint8, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray_numpy_uint8 = frame_numpy_uint8

        # optical flow
        if of is None or of[0] is None:
            motion_numpy_float32 = self.optical_flow_processor(frame_gray_numpy_uint8)  # it returns np.float32
        else:
            motion_numpy_float32 = of[0]

        # moving data into the torch environment: blurred frame, grayscale instance of it, motion
        # converting 3d tensors (h x w x depth) to 4d tensor float32 (1 x depth x h x w), in [0,1]
        frame = lve.utils.np_uint8_to_torch_float_01(frame_numpy_uint8, device=self.device)
        if frame.shape[1] == 3:
            frame_gray = lve.utils.np_uint8_to_torch_float_01(frame_gray_numpy_uint8, device=self.device)
        else:
            frame_gray = frame
        motion = lve.utils.np_float32_to_torch_float(motion_numpy_float32, device=self.device)

        # focus of attention
        foa, next_will_be_fixation = \
            self.foa_processor.next_location(frame_gray, motion, frame_gray_uint8_cpu=frame_gray_numpy_uint8)

        # computing random features (2 layers)
        features = [np.random.rand(self.h, self.w, 3).astype(np.float32),
                    np.random.rand(self.h, self.w, 4).astype(np.float32)]

        # adding numerical details
        feature_extractor_details = {'obj': random.uniform(0.0, 10.0), 'mi': random.uniform(0.0, 1.0)}

        # saving output data related to the current frame
        self.add_outputs({"motion": motion_numpy_float32,  # binary
                          "blurred": frame_numpy_uint8,  # PNG image
                          "others.foa": {"x": foa[0], "y": foa[1], "vx": foa[2], "vy": foa[3]},  # JSON
                          "others.fe": feature_extractor_details,  # JSON
                          "others.blur": {'rho': self.rho},  # JSON
                          "logs": [feature_extractor_details['obj'], feature_extractor_details['mi']],  # CSV log
                          "tb.rho": self.rho,  # tensorboard
                          "tb.fe_details": feature_extractor_details})  # tensorboard

        for i in range(0,2):
            self.add_output("probabilities." + str(i), features[i])  # binary
            self.add_output("filters." + str(i), self.dummy_weights[i])  # binary

    def update_model_parameters(self):
        self.dummy_weights = np.array([2.0*np.random.rand(4,self.c,9).astype(np.float32)-1.0,
                                       2.0*np.random.rand(3,4,25).astype(np.float32)-1.0], dtype=object)

        # blurring factor update rule
        if self.rho < 1.0:
            diff_rho = 1.0 - self.rho
            self.rho = self.rho + self.options["eta"] * diff_rho  # eta: hot-changeable option
            if self.rho > 0.99:
                self.rho = 1.0

    def load(self, model_folder):
        worker_model_folder = model_folder + os.sep

        # loading weights
        self.dummy_weights = np.load(worker_model_folder + "dummy_weights.npz", allow_pickle=True)['arr_0']

        # loading other parameters
        params = lve.utils.load_json(worker_model_folder + "params.json")

        # setting up the internal elements using the loaded parameters
        self.rho = params["rho"]
        self.foa_processor.reset(params["foa_y"], params["foa_t"])
        self.foa_processor.first_call = False

    def save(self, model_folder):
        worker_model_folder = model_folder + os.sep
        if not os.path.exists(worker_model_folder):
            os.makedirs(worker_model_folder)

        # saving weights
        np.savez_compressed(worker_model_folder + "dummy_weights.npz", self.dummy_weights)

        # saving other parameters
        lve.utils.save_json(worker_model_folder + "params.json", {"rho": self.rho,
                                                                  "foa_y": list(self.foa_processor.y),
                                                                  "foa_t": self.foa_processor.t})

    def get_output_types(self):
        output_types = { # the output element "frames" is already registered by default
            "motion": {"data_type": lve.OutputType.BINARY, "per_frame": True},
            "blurred": {"data_type": lve.OutputType.IMAGE, "per_frame": True},
            "others.foa": {"data_type": lve.OutputType.JSON, "per_frame": True},
            "others.fe": {"data_type": lve.OutputType.JSON, "per_frame": True},
            "others.blur": {"data_type": lve.OutputType.JSON, "per_frame": True},
            "logs": {"data_type": lve.OutputType.TEXT, "per_frame": False},
            "logs__header": ['frame', 'obj', 'mi']  # the special suffix "__header" will create the first line of CSV
        }

        for i in range(0, 2):
            output_types.update({
                "probabilities." + str(i): {"data_type": lve.OutputType.BINARY, "per_frame": True},
                "filters." + str(i): {"data_type": lve.OutputType.BINARY, "per_frame": True}
            })

        return output_types

    def print_info(self):
        print("   {rho: " + str(self.rho) + ", eta: " + str(self.options["eta"]) + "}")

    def __handle_command_reset_foa(self, command_value):
        self.foa_processor.reset([command_value['y'], command_value['x'],
                                  2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1)),
                                  2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))])

