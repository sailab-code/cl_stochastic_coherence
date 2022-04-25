import numpy
import numpy as np
import collections
from collections import OrderedDict
import lve
import os
import torch
import copy


class MetricsContainer:
    def __init__(self, num_classes, num_clusters, w, num_what, output_stream, options, threshold_list):
        self.metrics_list = [
            SupervisedMetrics(num_classes, num_clusters, w, num_what, output_stream, options, thresh_idx, thresh)
            for thresh_idx, thresh in enumerate(threshold_list)]

        self.unsup_metrics = Metrics(num_classes, num_clusters, w, num_what, output_stream, options)

        self.output_stream = output_stream

        self.num_classes = num_classes  # background class is already counted
        self.num_clusters = num_clusters
        self.w = w
        self.output_stream = output_stream
        self.window_size = options['window']  # these are the metrics-related options only
        self.trash_class = options['trash_class']  # these are the metrics-related options only
        self.num_what = num_what
        self.threshold_list = threshold_list

        # only for structure

        self.__stats = OrderedDict({
            'whole_frame':
                {
                    'running':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            'purity': [None] * (self.num_clusters + 1),
                            'rand_index': [None],
                            "coherence_loss_t": None,
                            "coherence_loss_s_in": None,
                            "coherence_loss_s_out": None
                        },
                    'window':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            'purity': [None] * (self.num_clusters + 1),
                            'rand_index': [None],
                            "coherence_loss_t": None,
                            "coherence_loss_s_in": None,
                            "coherence_loss_s_out": None,
                            "best_threshold": None
                        },
                },
            'foa':
                {
                    'running':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            'purity': [None] * (self.num_clusters + 1),
                            'rand_index': [None]
                        },
                    'window':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            'purity': [None] * (self.num_clusters + 1),
                            'rand_index': [None]
                        },
                },
            'foa_moving':
                {
                    'running':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            'purity': [None] * (self.num_clusters + 1),
                            'rand_index': [None]
                        },
                    'window':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            'purity': [None] * (self.num_clusters + 1),
                            'rand_index': [None]
                        }
                }
        })

        self.output_stream.register_output_elements(self.get_output_types())


    def update(self, full_sup):
        # update metrics for the various thresholds
        for i in self.metrics_list:
            i.update(full_sup)
        self.unsup_metrics.update(full_sup)

    def compute(self):
        # TODO put in self__stats the best

        for obj in self.metrics_list:
            obj.compute()
        self.unsup_metrics.compute()
        # update output stream

        best_supervised_stats, best_threshold_idx = self.pick_best_sup_stats()
        unsup_stats = self.unsup_metrics.get_stats()


        dict_stats = OrderedDict()
        for area, value_area in best_supervised_stats.items():
            dict_stats[area] = {}
            for setting, value_setting in value_area.items():
                d1 = copy.deepcopy(value_setting)
                d1.update(unsup_stats[area][setting])
                dict_stats[area][setting] = d1

        self.__stats = dict_stats
        self.__stats["whole_frame"]["window"]["best_threshold"] = self.threshold_list[best_threshold_idx]

        # "prediction_idx": prediction_idx_tensor_detached[0],
        best_preds = self.output_stream.get_output_elements()["prediction_idx-list"]["data"][best_threshold_idx]
        best_sup_probs = self.output_stream.get_output_elements()["sup-probs-list"]["data"][best_threshold_idx]
        self.output_stream.get_output_elements()["prediction_idx"]["data"] = best_preds
        self.output_stream.get_output_elements()["sup-probs"]["data"] = best_sup_probs

        self.output_stream.save_elements({"stats.metrics": self.__stats,  # dictionary
                                          "logs.metrics": self.__convert_stats_values_to_list(),  # CSV log
                                          "tb.metrics": self.__stats}, prev_frame=True)

    def pick_best_sup_stats(self):
        best_metric = -1.
        best_threshold_idx = None

        for obj in self.metrics_list:
            metric = obj.get_stats()["whole_frame"]["window"]["f1"][-1]
            if metric > best_metric:
                best_threshold_idx = obj.thresh_idx
                best_metric = metric

        return self.metrics_list[best_threshold_idx].get_stats(), best_threshold_idx

    def save(self, model_folder):
        for obj in self.metrics_list:
            obj.save(model_folder)
        self.unsup_metrics.save(model_folder)

    def load(self, model_folder):
        for obj in self.metrics_list:
            obj.load(model_folder)
        self.unsup_metrics.load(model_folder)

    def get_output_types(self):
        output_types = {
            "stats.metrics": {'data_type': lve.OutputType.JSON, 'per_frame': True},
            "logs.metrics": {'data_type': lve.OutputType.TEXT, 'per_frame': False},
            "logs.metrics__header": ['frame'] + self.__convert_stats_keys_to_list()
        }
        return output_types

    def __convert_stats_values_to_list(self):
        stats_list = []
        for area, area_d in self.__stats.items():
            for setting, setting_d in area_d.items():
                for metric, metric_v in setting_d.items():
                    if isinstance(metric_v, list):
                        for m_v in metric_v:
                            stats_list.append(m_v)
                    else:
                        stats_list.append(metric_v)
        return stats_list

    def __convert_stats_keys_to_list(self):
        stats_list = []
        for area, area_d in self.__stats.items():
            for setting, setting_d in area_d.items():
                for metric, metric_v in setting_d.items():
                    if isinstance(metric_v, list):
                        ml = len(metric_v)
                        for k in range(0, ml - 1):
                            stats_list.append(metric + '_c' + str(k))
                        stats_list.append(metric + '_glob')
                    else:
                        stats_list.append(metric)
        return stats_list


class Confusion:
    def __init__(self, labels, predictions):
        self.cm = numpy.zeros((labels, predictions))

    def get_cm(self):
        return self.cm


class SupervisedMetrics:
    def __init__(self, num_classes, num_clusters, w, num_what, output_stream, options, thresh_idx, thresh):
        # options
        self.num_classes = num_classes  # background class is already counted
        self.num_clusters = num_clusters
        self.w = w
        self.output_stream = output_stream
        self.window_size = options['window']  # these are the metrics-related options only
        self.trash_class = options['trash_class']  # these are the metrics-related options only
        self.num_what = num_what
        self.thresh_idx = thresh_idx
        self.thresh = thresh

        # references to the confusion and contingency matrices
        self.running_confusion_whole_frame = Confusion(num_classes, num_classes)
        self.running_confusion_foa = Confusion(num_classes, num_classes)
        self.running_confusion_foa_moving = Confusion(num_classes, num_classes)

        self.window_confusion_whole_frame = collections.deque(maxlen=self.window_size)
        self.window_confusion_foa = collections.deque(maxlen=self.window_size)
        self.window_confusion_foa_moving = collections.deque(maxlen=self.window_size)
        # add initial dummy one
        self.window_confusion_foa_moving.appendleft(Confusion(num_classes, num_classes).get_cm())

        self.__stats = OrderedDict({
            'whole_frame':
                {
                    'running':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),

                        },
                    'window':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                        },
                },
            'foa':
                {
                    'running':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),

                        },
                    'window':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),
                        },
                },
            'foa_moving':
                {
                    'running':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),

                        },
                    'window':
                        {
                            'acc': [None] * (self.num_classes + 1),
                            'f1': [None] * (self.num_classes + 1),

                        }
                }
        })

    def get_stats(self):
        return self.__stats

    def save(self, model_folder):
        metrics_model_folder = model_folder + os.sep + "metrics_supervised" + os.sep + str(self.thresh_idx) + os.sep
        if not os.path.exists(metrics_model_folder):
            os.makedirs(metrics_model_folder)

        # saving metrics-status related tensors
        torch.save({"running_confusion_whole_frame": self.running_confusion_whole_frame,
                    "running_confusion_foa": self.running_confusion_foa,
                    "running_confusion_foa_moving": self.running_confusion_foa_moving,
                    "window_confusion_whole_frame": self.window_confusion_whole_frame,
                    "window_confusion_foa": self.window_confusion_foa,
                    "window_confusion_foa_moving": self.window_confusion_foa_moving,
                    },
                   metrics_model_folder + "metrics.pth")

    def load(self, model_folder):
        metrics_model_folder = model_folder + os.sep + "metrics_supervised" + os.sep + str(self.thresh_idx) + os.sep

        # loading metrics-status related tensors
        if os.path.exists(metrics_model_folder + "metrics.pth"):
            metrics_status = torch.load(metrics_model_folder + "metrics.pth")

            self.running_confusion_whole_frame = metrics_status["running_confusion_whole_frame"]
            self.running_confusion_foa = metrics_status["running_confusion_foa"]
            self.running_confusion_foa_moving = metrics_status["running_confusion_foa_moving"]
            self.window_confusion_whole_frame = metrics_status["window_confusion_whole_frame"]
            self.window_confusion_foa = metrics_status["window_confusion_foa"]
            self.window_confusion_foa_moving = metrics_status["window_confusion_foa_moving"]

    def compute_confusion(self, y, y_pred):
        indices = self.num_classes * y.to(torch.int64) + y_pred.to(torch.int64)
        m = torch.bincount(indices,
                           minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return m

    @staticmethod
    def __compute_matrices_and_update_running(pred, target, compute, running_cm, running_cm_window):
        current_cm = compute(y_pred=torch.as_tensor(pred), y=torch.as_tensor(target)).numpy()
        running_cm.cm = running_cm.cm + current_cm

        # windowed confusion matrix update
        running_cm_window.appendleft(current_cm)

    def update(self, full_sup):
        # reset the movement flag

        # get the current frame targets
        targets, indices = full_sup

        # getting model predictions, gather predictions from output stream, they are already in numpy!
        pred_idx = self.output_stream.get_output_elements()["prediction_idx-list"]["data"][
            self.thresh_idx]  # get the current frame
        motion = self.output_stream.get_output_elements()["motion"]["data"]  # get the current optical flow

        # computing confusion and contingency matrices
        SupervisedMetrics.__compute_matrices_and_update_running(pred=pred_idx, target=targets,
                                                                compute=self.compute_confusion,
                                                                running_cm=self.running_confusion_whole_frame,
                                                                running_cm_window=self.window_confusion_whole_frame)

        # restricting to the FOA
        foax = self.output_stream.get_output_elements()["stats.worker"]["data"]["foax"].astype(np.long)
        foay = self.output_stream.get_output_elements()["stats.worker"]["data"]["foay"].astype(np.long)

        pred_foa = torch.tensor([pred_idx[foax * self.w + foay]])
        target_foa = torch.tensor([targets[foax * self.w + foay]])

        # computing confusion and contingency matrices (FOA only)
        SupervisedMetrics.__compute_matrices_and_update_running(pred=pred_foa, target=target_foa,
                                                                compute=self.compute_confusion,
                                                                running_cm=self.running_confusion_foa,
                                                                running_cm_window=self.window_confusion_foa)

        # computing confusion and contingency matrices (FOA moving only)
        if np.linalg.norm(motion[foax, foay, :]) > 0.:
            # set the movement flag
            # self.there_is_movement_flag = True
            SupervisedMetrics.__compute_matrices_and_update_running(pred=pred_foa, target=target_foa,
                                                                    compute=self.compute_confusion,
                                                                    running_cm=self.running_confusion_foa_moving,
                                                                    running_cm_window=self.window_confusion_foa_moving)

    @staticmethod
    def __compute__all__supervised_metrics(confusion_mat):
        per_class_accuracy, global_accuracy = Metrics.accuracy(confusion_mat)
        per_class_f1, global_f1 = Metrics.f1(confusion_mat)

        return {'acc': np.append(per_class_accuracy, global_accuracy),
                'f1': np.append(per_class_f1, global_f1),
                }

    def compute(self):
        """
        Computes the metric based on it's accumulated state.

        """

        # computing all the metrics, given the pre-computed matrices
        metrics_whole_frame = \
            SupervisedMetrics.__compute__all__supervised_metrics(self.running_confusion_whole_frame.cm)
        metrics_foa = \
            SupervisedMetrics.__compute__all__supervised_metrics(self.running_confusion_foa.cm)

        metrics_foa_moving = \
            SupervisedMetrics.__compute__all__supervised_metrics(self.running_confusion_foa_moving.cm)

        metrics_whole_frame_window = \
            SupervisedMetrics.__compute__all__supervised_metrics(np.sum(self.window_confusion_whole_frame, axis=0))
        metrics_foa_window = \
            SupervisedMetrics.__compute__all__supervised_metrics(np.sum(self.window_confusion_foa, axis=0))
        metrics_foa_moving_window = \
            SupervisedMetrics.__compute__all__supervised_metrics(np.sum(self.window_confusion_foa_moving, axis=0))

        self.__stats.update({
            'whole_frame':
                {
                    'running':
                        {
                            'acc': metrics_whole_frame['acc'].tolist(),
                            'f1': metrics_whole_frame['f1'].tolist(),

                        },
                    'window':
                        {
                            'acc': metrics_whole_frame_window['acc'].tolist(),
                            'f1': metrics_whole_frame_window['f1'].tolist(),

                        },
                },
            'foa':
                {
                    'running':
                        {
                            'acc': metrics_foa['acc'].tolist(),
                            'f1': metrics_foa['f1'].tolist(),

                        },
                    'window':
                        {
                            'acc': metrics_foa_window['acc'].tolist(),
                            'f1': metrics_foa_window['f1'].tolist(),

                        },
                },
            'foa_moving':
                {
                    'running':
                        {
                            'acc': metrics_foa_moving['acc'].tolist(),
                            'f1': metrics_foa_moving['f1'].tolist(),

                        },
                    'window':
                        {
                            'acc': metrics_foa_moving_window['acc'].tolist(),
                            'f1': metrics_foa_moving_window['f1'].tolist(),

                        }
                }
        })


class Metrics:

    def __init__(self, num_classes, num_clusters, w, num_what, output_stream, options):

        # options
        self.num_classes = num_classes  # background class is already counted
        self.num_clusters = num_clusters
        self.w = w
        self.output_stream = output_stream
        self.window_size = options['window']  # these are the metrics-related options only
        self.trash_class = options['trash_class']  # these are the metrics-related options only
        self.num_what = num_what

        self.movement_counter_coherence_t = 0
        self.running_coherence_t = 0.0
        self.coherence_t_window = collections.deque(maxlen=self.window_size)
        self.coherence_t_window.appendleft(0.0)

        self.blob_counter_coherence_s_in = 0
        self.running_coherence_s_in = 0.0
        self.coherence_s_in_window = collections.deque(maxlen=self.window_size)
        self.coherence_s_in_window.appendleft(0.0)

        self.blob_counter_coherence_s_out = 0
        self.running_coherence_s_out = 0.0
        self.coherence_s_out_window = collections.deque(maxlen=self.window_size)
        self.coherence_s_out_window.appendleft(0.0)

        # references to the confusion and contingency matrices

        self.running_confusion_self_whole_frame = Confusion(num_classes, num_classes)  #
        self.running_confusion_self_foa = Confusion(num_classes, num_classes)
        self.running_confusion_self_foa_moving = Confusion(num_classes, num_classes)

        self.running_contingency_whole_frame = Confusion(num_classes, num_clusters)
        self.running_contingency_foa = Confusion(num_classes, num_clusters)
        self.running_contingency_foa_moving = Confusion(num_classes, num_clusters)

        self.window_confusion_self_whole_frame = collections.deque(maxlen=self.window_size)
        self.window_confusion_self_foa = collections.deque(maxlen=self.window_size)
        self.window_confusion_self_foa_moving = collections.deque(maxlen=self.window_size)
        # add initial dummy one
        self.window_confusion_self_foa_moving.appendleft(Confusion(num_classes, num_classes).get_cm())

        self.window_contingency_whole_frame = collections.deque(maxlen=self.window_size)
        self.window_contingency_foa = collections.deque(maxlen=self.window_size)
        self.window_contingency_foa_moving = collections.deque(maxlen=self.window_size)
        # add initial dummy one
        self.window_contingency_foa_moving.appendleft(Confusion(num_classes, num_clusters).get_cm())

        self.__stats = OrderedDict({
            'whole_frame':
                {
                    'running':
                        {
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            'purity': [None] * (self.num_clusters + 1),
                            'rand_index': [None],
                            "coherence_loss_t": None,
                            "coherence_loss_s_in": None,
                            "coherence_loss_s_out": None
                        },
                    'window':
                        {
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            'purity': [None] * (self.num_clusters + 1),
                            'rand_index': [None],
                            "coherence_loss_t": None,
                            "coherence_loss_s_in": None,
                            "coherence_loss_s_out": None
                        },
                },
            'foa':
                {
                    'running':
                        {
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            'purity': [None] * (self.num_clusters + 1),
                            'rand_index': [None]
                        },
                    'window':
                        {
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            'purity': [None] * (self.num_clusters + 1),
                            'rand_index': [None]
                        },
                },
            'foa_moving':
                {
                    'running':
                        {
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            'purity': [None] * (self.num_clusters + 1),
                            'rand_index': [None]
                        },
                    'window':
                        {
                            'acc_self': [None] * (self.num_classes + 1),
                            'f1_self': [None] * (self.num_classes + 1),
                            'purity': [None] * (self.num_clusters + 1),
                            'rand_index': [None]
                        }
                }
        })

    def get_stats(self):
        return self.__stats

    def compute(self):
        """
        Computes the metric based on it's accumulated state.

        """

        # computing all the metrics, given the pre-computed matrices
        metrics_whole_frame = \
            Metrics.__compute_all_unsup_metrics(self.running_contingency_whole_frame.cm,
                                                self.trash_class)
        metrics_foa = \
            Metrics.__compute_all_unsup_metrics(self.running_contingency_foa.cm,
                                                self.trash_class)

        metrics_foa_moving = \
            Metrics.__compute_all_unsup_metrics(self.running_contingency_foa_moving.cm,
                                                self.trash_class)

        metrics_whole_frame_window = \
            Metrics.__compute_all_unsup_metrics(np.sum(self.window_contingency_whole_frame, axis=0),
                                                self.trash_class)
        metrics_foa_window = \
            Metrics.__compute_all_unsup_metrics(np.sum(self.window_contingency_foa, axis=0),
                                                self.trash_class)
        metrics_foa_moving_window = \
            Metrics.__compute_all_unsup_metrics(np.sum(self.window_contingency_foa_moving, axis=0),
                                                self.trash_class)

        # coherence_t
        # running coherence_t whole life
        whole_running_coherence_t = self.running_coherence_t / (
                self.movement_counter_coherence_t * self.num_what + 1e-20)
        # window coherence_t
        window_coherence_t = np.sum(self.coherence_t_window) / (len(self.coherence_t_window) * self.num_what)

        # coherence_s_in
        whole_running_coherence_s_in = self.running_coherence_s_in / (
                self.blob_counter_coherence_s_in * self.num_what + 1e-20)
        window_coherence_s_in = np.sum(self.coherence_s_in_window) / (len(self.coherence_s_in_window) * self.num_what)

        # coherence_s_out
        whole_running_coherence_s_out = self.running_coherence_s_out / (
                self.blob_counter_coherence_s_out * self.num_what + 1e-20)
        window_coherence_s_out = np.sum(self.coherence_s_out_window) / (
                len(self.coherence_s_out_window) * self.num_what)

        self.__stats.update({
            'whole_frame':
                {
                    'running':
                        {
                            'acc_self': metrics_whole_frame['acc_self'].tolist(),
                            'f1_self': metrics_whole_frame['f1_self'].tolist(),
                            'purity': metrics_whole_frame['purity'].tolist(),
                            'rand_index': metrics_whole_frame['rand_index'].tolist(),
                            'coherence_loss_t': whole_running_coherence_t,
                            'coherence_loss_s_in': whole_running_coherence_s_in,
                            'coherence_loss_s_out': whole_running_coherence_s_out
                        },
                    'window':
                        {
                            'acc_self': metrics_whole_frame_window['acc_self'].tolist(),
                            'f1_self': metrics_whole_frame_window['f1_self'].tolist(),
                            'purity': metrics_whole_frame_window['purity'].tolist(),
                            'rand_index': metrics_whole_frame_window['rand_index'].tolist(),
                            'coherence_loss_t': window_coherence_t,
                            'coherence_loss_s_in': window_coherence_s_in,
                            'coherence_loss_s_out': window_coherence_s_out
                        },
                },
            'foa':
                {
                    'running':
                        {

                            'acc_self': metrics_foa['acc_self'].tolist(),
                            'f1_self': metrics_foa['f1_self'].tolist(),
                            'purity': metrics_foa['purity'].tolist(),
                            'rand_index': metrics_foa['rand_index'].tolist()
                        },
                    'window':
                        {

                            'acc_self': metrics_foa_window['acc_self'].tolist(),
                            'f1_self': metrics_foa_window['f1_self'].tolist(),
                            'purity': metrics_foa_window['purity'].tolist(),
                            'rand_index': metrics_foa_window['rand_index'].tolist()
                        },
                },
            'foa_moving':
                {
                    'running':
                        {

                            'acc_self': metrics_foa_moving['acc_self'].tolist(),
                            'f1_self': metrics_foa_moving['f1_self'].tolist(),
                            'purity': metrics_foa_moving['purity'].tolist(),
                            'rand_index': metrics_foa_moving['rand_index'].tolist()
                        },
                    'window':
                        {

                            'acc_self': metrics_foa_moving_window['acc_self'].tolist(),
                            'f1_self': metrics_foa_moving_window['f1_self'].tolist(),
                            'purity': metrics_foa_moving_window['purity'].tolist(),
                            'rand_index': metrics_foa_moving_window['rand_index'].tolist()
                        }
                }
        })

    def update(self, full_sup):

        # reset the movement flag

        # self.there_is_movement_flag = False

        # get the current frame targets
        targets, indices = full_sup

        motion = self.output_stream.get_output_elements()["motion"]["data"]  # get the current optical flow

        # getting unsupervised outputs from output_stream
        unsup_pred_idx = self.output_stream.get_output_elements()["unsup-probs_idx"]["data"]  # get the current frame

        Metrics.__compute_matrices_and_update_running(pred=unsup_pred_idx, target=targets,
                                                      compute=self.compute_contingency,
                                                      running_cm=self.running_contingency_whole_frame,
                                                      running_cm_window=self.window_contingency_whole_frame)

        # restricting to the FOA
        foax = self.output_stream.get_output_elements()["stats.worker"]["data"]["foax"].astype(np.long)
        foay = self.output_stream.get_output_elements()["stats.worker"]["data"]["foay"].astype(np.long)

        target_foa = torch.tensor([targets[foax * self.w + foay]])

        # unsupervised foa predictions
        unsup_pred_foa = torch.tensor([unsup_pred_idx[foax * self.w + foay]])

        Metrics.__compute_matrices_and_update_running(pred=unsup_pred_foa, target=target_foa,
                                                      compute=self.compute_contingency,
                                                      running_cm=self.running_contingency_foa,
                                                      running_cm_window=self.window_contingency_foa)

        # computing confusion and contingency matrices (FOA moving only)
        if np.linalg.norm(motion[foax, foay, :]) > 0.:
            Metrics.__compute_matrices_and_update_running(pred=unsup_pred_foa, target=target_foa,
                                                          compute=self.compute_contingency,
                                                          running_cm=self.running_contingency_foa_moving,
                                                          running_cm_window=self.window_contingency_foa_moving)

            # coherence_t
            coherence_t = self.output_stream.get_output_elements()["stats.worker"]["data"]["loss_t"]
            self.update_coherence_t_values(coherence_t=coherence_t)

        # coherence_s
        blob = self.output_stream.get_output_elements()["blob"]["data"]  # get the current blob - note that it has
        # a rgb mask of values 0 or  255
        blob_area = np.sum(blob)  # area of the blob
        if blob_area > 1:
            coherence_s_in = self.output_stream.get_output_elements()["stats.worker"]["data"]["loss_s_in"]
            coherence_s_out = self.output_stream.get_output_elements()["stats.worker"]["data"]["loss_s_out"]
            self.update_coherence_s_in_values(coherence_s=coherence_s_in)
            self.update_coherence_s_out_values(coherence_s=coherence_s_out)

    def load(self, model_folder):
        metrics_model_folder = model_folder + os.sep + "metrics_unsupervised" + os.sep

        # loading metrics-status related tensors
        if os.path.exists(metrics_model_folder + "metrics.pth"):
            metrics_status = torch.load(metrics_model_folder + "metrics.pth")

            self.running_contingency_whole_frame = metrics_status["running_contingency_whole_frame"]
            self.running_contingency_foa = metrics_status["running_contingency_foa"]
            self.running_contingency_foa_moving = metrics_status["running_contingency_foa_moving"]
            self.window_contingency_whole_frame = metrics_status["window_contingency_whole_frame"]
            self.window_contingency_foa = metrics_status["window_contingency_foa"]
            self.window_contingency_foa_moving = metrics_status["window_contingency_foa_moving"]
            self.running_coherence_t = metrics_status["running_coherence_t"]
            self.movement_counter_coherence_t = metrics_status["movement_counter_coherence_t"]
            self.coherence_t_window = metrics_status["window_coherence_t"]
            self.running_coherence_s_in = metrics_status["running_coherence_s_in"]
            self.blob_counter_coherence_s_in = metrics_status["blob_counter_coherence_s_in"]
            self.coherence_s_in_window = metrics_status["window_coherence_s_in"]
            self.running_coherence_s_out = metrics_status["running_coherence_s_out"]
            self.blob_counter_coherence_s_out = metrics_status["blob_counter_coherence_s_out"]
            self.coherence_s_out_window = metrics_status["window_coherence_s_out"]

    def save(self, model_folder):
        metrics_model_folder = model_folder + os.sep + "metrics_unsupervised" + os.sep
        if not os.path.exists(metrics_model_folder):
            os.makedirs(metrics_model_folder)

        # saving metrics-status related tensors
        torch.save({
            "running_contingency_whole_frame": self.running_contingency_whole_frame,
            "running_contingency_foa": self.running_contingency_foa,
            "running_contingency_foa_moving": self.running_contingency_foa_moving,
            "window_contingency_whole_frame": self.window_contingency_whole_frame,
            "window_contingency_foa": self.window_contingency_foa,
            "window_contingency_foa_moving": self.window_contingency_foa_moving,
            "running_coherence_t": self.running_coherence_t,
            "movement_counter_coherence_t": self.movement_counter_coherence_t,
            "window_coherence_t": self.coherence_t_window,
            "window_coherence_t": self.coherence_t_window,
            "running_coherence_s_in": self.running_coherence_s_in,
            "blob_counter_coherence_s_in": self.blob_counter_coherence_s_in,
            "window_coherence_s_in": self.coherence_s_in_window,
            "running_coherence_s_out": self.running_coherence_s_out,
            "blob_counter_coherence_s_out": self.blob_counter_coherence_s_out,
            "window_coherence_s_out": self.coherence_s_out_window
        },
            metrics_model_folder + "metrics.pth")

    def print_info(self):
        s = "   metrics {"
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

    def compute_confusion(self, y, y_pred):
        indices = self.num_classes * y.to(torch.int64) + y_pred.to(torch.int64)
        m = torch.bincount(indices,
                           minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return m

    def compute_contingency(self, y, y_pred):
        indices = self.num_clusters * y.to(torch.int64) + y_pred.to(torch.int64)
        m = torch.bincount(indices,
                           minlength=self.num_classes * self.num_clusters).reshape(self.num_classes, self.num_clusters)
        return m

    @staticmethod
    def __compute_matrices_and_update_running(pred, target, compute, running_cm, running_cm_window):
        current_cm = compute(y_pred=torch.as_tensor(pred), y=torch.as_tensor(target)).numpy()
        running_cm.cm = running_cm.cm + current_cm

        # windowed confusion matrix update
        running_cm_window.appendleft(current_cm)

    @staticmethod
    def __compute_all_unsup_metrics(contingency_mat, trash_class):
        running_confusion_self = Metrics.__compute_self_supervised_confusion_matrix(contingency_mat, trash_class)

        per_class_accuracy_self, global_accuracy_self = Metrics.accuracy(running_confusion_self)
        per_class_f1_self, global_f1_self = Metrics.f1(running_confusion_self)

        per_cluster_purity, global_purity = Metrics.purity(contingency_mat)
        global_ari = Metrics.adjusted_rand_index(contingency_mat)

        return {
            'acc_self': np.append(per_class_accuracy_self, global_accuracy_self),
            'f1_self': np.append(per_class_f1_self, global_f1_self),
            'purity': np.append(per_cluster_purity, global_purity),
            'rand_index': np.array([global_ari])}

    @staticmethod
    def __compute_self_supervised_confusion_matrix(contingency_matrix, trash_class):
        c = contingency_matrix.shape[0]
        k = contingency_matrix.shape[1]
        cont_copy = np.array(contingency_matrix, copy=True)  # deep copy of the contingency matrix (needed)
        cont_copy_sum_cols = np.sum(cont_copy, axis=1, keepdims=True)
        cont_copy_sum_cols[cont_copy_sum_cols == 0] = 1
        cont_copy = cont_copy / cont_copy_sum_cols
        cont_copy[trash_class, :] = -1

        # computing a customized homogeneity (purity) and completeness scores, limited to the largest clusters
        classes_to_clusters = np.zeros(c, dtype=np.int64)  # from class ID to associated cluster ID
        unassociated_clusters = np.ones(k, dtype=np.bool)

        for i in range(0, c):  # warning: the cluster associated to the trash class is basically the first one left

            # computing the (currently) largest cluster and its associated class
            largest_id = cont_copy.argmax()
            largest_cluster_id = largest_id - (largest_id // k) * k
            largest_cluster_associated_class_id = largest_id // k

            # saving association
            classes_to_clusters[largest_cluster_associated_class_id] = largest_cluster_id

            # virtually purging row and column of the (currently) largest cluster from the contingency matrix
            cont_copy[largest_cluster_associated_class_id, :] = -1
            cont_copy[:, largest_cluster_id] = -1

            # marking
            unassociated_clusters[largest_cluster_id] = False

        # reducing the contingency matrix
        cont_restricted = contingency_matrix[:, classes_to_clusters]

        # accumulating in the background class all the other clusters
        for j in range(0, k):
            if unassociated_clusters[j]:
                cont_restricted[:, trash_class] = cont_restricted[:, trash_class] + contingency_matrix[:, j]

        return cont_restricted

    @staticmethod
    def accuracy(cm):
        acc_det = cm.sum(axis=1)
        acc_det[acc_det == 0] = 1
        per_class_accuracy = cm.diagonal() / acc_det
        global_accuracy = np.mean(per_class_accuracy)  # macro
        return per_class_accuracy, global_accuracy

    @staticmethod
    def f1(cm):
        num_classes = cm.shape[0]
        per_class_f1 = np.zeros(num_classes)

        for c in range(0, num_classes):
            tp = cm[c, c]
            fn = np.sum(cm[c, :]) - tp
            fp = np.sum(cm[:, c]) - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.
            per_class_f1[c] = (2. * p * r) / (p + r) if (p + r) > 0 else 0.

        global_f1 = np.mean(per_class_f1)  # macro
        return per_class_f1, global_f1

    @staticmethod
    def purity(cm):
        most_represented_class_per_cluster = np.amax(cm, axis=0)
        number_of_points_per_cluster = np.sum(cm, axis=0)
        number_of_points_per_cluster[number_of_points_per_cluster == 0] = 1.0
        per_cluster_purity = most_represented_class_per_cluster / number_of_points_per_cluster
        cm_sum = np.sum(cm)
        global_purity = np.sum(most_represented_class_per_cluster) / (cm_sum if cm_sum != 0 else 1)
        return per_cluster_purity, global_purity

    @staticmethod
    def adjusted_rand_index(cm):
        a = np.sum(cm, axis=1)
        b = np.sum(cm, axis=0)
        n = np.sum(cm)

        n_bin = (cm * (cm - 1)) / 2.
        a_bin = (a * (a - 1)) / 2.
        b_bin = (b * (b - 1)) / 2.

        a_bin_sum = np.sum(a_bin)
        b_bin_sum = np.sum(b_bin)
        mixed_term_det = (n * (n - 1) / 2.)
        mixed_term = (a_bin_sum * b_bin_sum) / mixed_term_det if mixed_term_det != 0. else 0.

        ari_num = np.sum(n_bin) - mixed_term
        ari_den = 0.5 * (a_bin_sum + b_bin_sum) - mixed_term

        return ari_num / ari_den if ari_den != 0. else 0.

    def update_coherence_t_values(self, coherence_t):

        self.running_coherence_t += coherence_t
        self.movement_counter_coherence_t += 1

        # window
        self.coherence_t_window.appendleft(coherence_t)

    def update_coherence_s_in_values(self, coherence_s):

        self.running_coherence_s_in += coherence_s
        self.blob_counter_coherence_s_in += 1

        # window
        self.coherence_s_in_window.appendleft(coherence_s)

    def update_coherence_s_out_values(self, coherence_s):

        self.running_coherence_s_out += coherence_s
        self.blob_counter_coherence_s_out += 1

        # window
        self.coherence_s_out_window.appendleft(coherence_s)
