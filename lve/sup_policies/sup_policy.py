import abc
import math
import os

import lve.utils
import lve.sup_policies


class SupervisionPolicy(metaclass=abc.ABCMeta):
    JSON_FILENAME = "sup_policy.json"

    def __init__(self, video_frames, options):
        self.options = options
        self.video_frames = video_frames
        self.sup_history = {}  # key => class id, value {count: int, last_frame: int}

    @staticmethod
    def create_from_options(video_frames, options):
        if options['type'] == 'only_moving_objects':
            return lve.sup_policies.OnlyMovingObjectsPolicy(video_frames, options)
        else:
            raise ValueError("Invalid supervision policy type")

    def get_current_repetition(self, frame_idx):
        return math.ceil(frame_idx / self.video_frames)

    def apply(self, frame_idx, batch_frame, batch_of, batch_supervisions, batch_foa):
        current_repetition = self.get_current_repetition(frame_idx)

        if current_repetition >= self.options['min_repetitions'] \
                and ('max_repetitions' not in self.options or current_repetition <=  self.options['max_repetitions']):
            batch_supervisions = self._apply(batch_frame, batch_of, batch_supervisions, batch_foa)
        else:
            batch_supervisions = None

        if batch_supervisions is not None:
            batch_supervisions = self.filter_by_history(batch_supervisions, frame_idx)

            # update history with remaining supervisions
            self.update_sup_history(batch_supervisions, frame_idx)

        if batch_supervisions is not None and len(batch_supervisions) == 0:
            batch_supervisions = None

        return batch_supervisions

    def max_supervisions_condition(self, class_history):
        return class_history['count'] >= self.options['max_supervisions']

    def wait_for_frames_condition(self, class_history, frame_idx):
        return (
            class_history['count'] != 0 and
            class_history['last_frame'] + self.options['wait_for_frames'] >= frame_idx
        )

    def filter_by_history(self, batch_supervisions, frame_idx):
        filtered_supervisions = []
        for supervisions in batch_supervisions:
            targets, indices = supervisions

            filtered_targets, filtered_indices = [], []

            for idx, target in enumerate(targets):
                class_history = self.sup_history.get(str(target), {"count": 0, "last_frame": 0})

                if not (
                    self.max_supervisions_condition(class_history) or
                    self.wait_for_frames_condition(class_history, frame_idx)
                ):
                    filtered_targets.append(target)
                    filtered_indices.append(indices[idx])

            if len(filtered_targets) != 0:
                filtered_supervisions.append((filtered_targets, filtered_indices))

        return filtered_supervisions

    def update_sup_history(self, batch_supervisions, frame_idx):
        filtered_supervisions = [*batch_supervisions]
        for supervisions in filtered_supervisions:
            targets, indices = supervisions

            for target in targets:
                class_history = self.sup_history.get(str(target), {"count": 0, "last_frame": 0})
                class_history['count'] += 1
                class_history['last_frame'] = frame_idx
                self.sup_history[str(target)] = class_history

    def save(self, model_folder):
        json_path = model_folder + os.sep + self.JSON_FILENAME
        lve.utils.save_json(json_path, self.sup_history)

    def load(self, model_folder):
        json_path = model_folder + os.sep + self.JSON_FILENAME
        self.sup_history = lve.utils.load_json(json_path)

    @abc.abstractmethod
    def _apply(self, frame, of, supervisions, foa):
        pass
