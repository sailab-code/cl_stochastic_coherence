import time
import traceback
import json
import lve
import argparse
import os

from settings import sailenv_settings, pretrained_segmentation_archs_dim

start_time = time.time()

# resuming
resume = False


def run_exp(args_cmd):
    args_cmd.num_what = args_cmd.num_what if args_cmd.num_what > 0 else pretrained_segmentation_archs_dim[args_cmd.arch]

    # creating streams
    if args_cmd.dataset == "livingroom":
        supervised_categories = 5
        foa_file = "data/livingroom/foa_log_alpha_c0.1__alpha_of_1.0__alpha_fm_0.0__" + \
                   "max_distance_257__dissipation_0.05__fixation_threshold_speed_25.foa"
    elif args_cmd.dataset == "emptyspace":
        supervised_categories = 5
        foa_file = "data/emptyspace/empty_space_bench_foa_long.foa"
    elif args_cmd.dataset == "solid":
        supervised_categories = 4
        foa_file = "data/solid/foa_new_solid_bench_long.foa"

    repetitions = args_cmd.laps_unsup + args_cmd.laps_sup + args_cmd.laps_metrics
    ins = lve.InputStream("data/" + args_cmd.dataset, w=-1, h=-1, fps=None, max_frames=None,
                          repetitions=repetitions, force_gray=args_cmd.force_gray == "yes", foa_file=foa_file,
                          unity_settings=sailenv_settings)

    if args_cmd.dataset == "solid" and args_cmd.force_gray == "no":
        raise NotImplementedError

    output_settings = {
        'folder': "output_folder",
        'fps': ins.fps,
        'virtual_save': True,
        'tensorboard': False,
        'save_per_frame_data': True,
        'purge_existing_data': not resume
    }

    #### OPTIONS
    general_options = {
        "device": args_cmd.device,  # "cuda:0",  # cpu, cuda:0, cuda:1, ...
        "seed": args_cmd.seed,  # if smaller than zero, current time is used
        'motion_threshold': -1.0,  # if negative, the whole set of moving pixels are taken
        'mi_history_weight': 0.1,  # the contribution of the last frame in the MI-related frequency counts
        'sup_batch': 16,
        'sup_persistence': 5,
        'piggyback_frames': supervised_categories * args_cmd.max_supervisions if args_cmd.train == 'yes' else 1,
        "supervision_map": ins.sup_map
    }

    sup_policy_options = {
        'type': 'only_moving_objects',
        'min_repetitions': args_cmd.laps_unsup + 1,
        # first repetitions which receives supervisions (the one after unsup reps)
        'max_repetitions': args_cmd.laps_unsup + args_cmd.laps_sup,
        # last repetition which receives supervisions
        'wait_for_frames': 100,  # frames passed before giving a supervision again
        'max_supervisions': args_cmd.max_supervisions  # max supervisions per object
    }

    foa_options = {'alpha_c': 0.1,
                   'alpha_of': 1.0,
                   'alpha_fm': 0.0,
                   'alpha_virtual': 0.0,
                   'max_distance': int(0.5 * (ins.w + ins.h)) if int(0.5 * (ins.w + ins.h)) % 2 == 1 else int(
                       0.5 * (ins.w + ins.h)) + 1,
                   'dissipation': 0.1,
                   'fps': ins.fps,
                   'w': ins.w,
                   'h': ins.h,
                   'y': None,
                   'is_online': ins.input_type == lve.InputType.WEB_CAM or ins.input_type == lve.InputType.UNITY,
                   'fixation_threshold_speed': int(0.1 * 0.5 * (ins.w + ins.h))}

    if args_cmd.normalize == "yes":
        dist_threshold = [0.000001, 0.0005, 0.0003, 0.0002, 0.0007, 0.001, 0.01, 0.1, 0.25, 0.5, 0.7, 1.0]
    else:
        dist_threshold = [0.1, 2, 10, 18, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600]

    net_options = {'c': ins.c,
                   'step_size': args_cmd.step_size,  # a negative value triggers Adam
                   'num_what': args_cmd.num_what,
                   'num_where': 0,
                   'unsupervised_categories': 2,
                   'supervised_categories': supervised_categories,
                   "classifier": "NN",  # 'NN', 'neural'
                   'dist_threshold': dist_threshold,
                   'lambda_c': 0.0,
                   'lambda_e': 0.0,
                   'num_pairs': args_cmd.num_pairs,
                   'spread_factor': args_cmd.spread_factor,
                   'lambda_t': args_cmd.lambda_t,
                   'lambda_s_in': args_cmd.lambda_s_in,
                   'lambda_s_out': args_cmd.lambda_s_out,
                   'lambda_l': 0.1,
                   'freeze': args_cmd.train == "no",
                   'training_max_repetitions': repetitions - args_cmd.laps_metrics,
                   "blob_reconstruction": True,
                   'architecture': args_cmd.arch,
                   'normalize': args_cmd.normalize == "yes"}

    metrics_options = {'window': ins.effective_video_frames,
                       'min_repetitions': args_cmd.laps_unsup + 1,  # we save a bit of time
                       'trash_class': ins.sup_map['background']}

    # creating worker
    worker = lve.WorkerWW(ins.w, ins.h, ins.c, ins.fps, options={
        **general_options,
        "sup_policy": sup_policy_options,
        "foa": foa_options,
        "net": net_options,
        "metrics": metrics_options
    })

    # logger
    log_dict = {'element': 'stats.metrics', 'log_last_only': True, 'logged': []}

    log_opts = {'': general_options,
                'sup_policy': sup_policy_options,
                'net': net_options,
                'metrics': metrics_options,
                '': {
                    'repetitions_unsup': args_cmd.laps_unsup,
                    'repetitions_sup': args_cmd.laps_sup,
                    'force_gray': args_cmd.force_gray,
                    'dataset': args_cmd.dataset,
                    'ref_run': args_cmd.ref_run,
                    'notes': args_cmd.notes
                }
                }
    total_options = {}
    for prefix, dic in log_opts.items():
        for key, val in dic.items():
            total_options[prefix + "_" + key] = val

    # processing stream
    target_port = 8080

    port = target_port
    while port - target_port < 25:
        try:
            print('Starting VProcessor with visualizer at port ' + str(port) + '..')
            outs = lve.OutputStream(**output_settings)
            lve.VProcessor(ins, outs, worker, "model_folder",
                           visualization_port=port, resume=resume).process_video(log_dict=log_dict)

            break
        except OSError:
            traceback.print_exc()
            port += 1

    elapsed_time = time.time() - start_time

    # closing streams
    ins.close()
    outs.close()

    print("")
    print("Elapsed: " + str(elapsed_time) + " seconds")

    final_stats = log_dict['logged'][-1]

    # final evaluation
    print("F1 Global (whole-frame, window, all classes + global):")
    print(final_stats['whole_frame']['window']['f1'][:])

    mapping_dict = {"whole_frame": "whole", "foa_moving": "foam", "foa": "foa"}

    metric_dict = {}
    for metric in ["f1", "acc"]:
        for area in ["whole_frame", "foa", "foa_moving"]:
            for setting in ["window", "running"]:
                key = f"{metric}_{setting}_{mapping_dict[area]}"
                value = final_stats[area][setting][metric]
                for i in range(len(value) - 1):
                    metric_dict[key + "_" + str(i)] = value[i]
                metric_dict[key + "_global"] = value[-1]
    # dump metrics dict to file
    with open(os.path.join("model_folder", 'results.json'), 'w') as fp:
        json.dump(metric_dict, fp)


def get_runner_parser():
    parser = argparse.ArgumentParser(description='Sthocastic coherence experiments for settings')
    parser.add_argument('--laps_unsup', type=int, default=1)
    parser.add_argument('--laps_sup', type=int, default=1)
    parser.add_argument('--laps_metrics', type=int, default=1)
    parser.add_argument('--step_size', type=float, default=-0.001)
    parser.add_argument('--lambda_s_in', type=float, default=0.00001)
    parser.add_argument('--lambda_s_out', type=float, default=0.00001)
    parser.add_argument('--spread_factor', type=float, default=1.5)
    parser.add_argument('--lambda_t', type=float, default=0.00001)
    parser.add_argument('--num_what', type=int, default=-1)
    parser.add_argument('--num_pairs', type=int, default=100)
    parser.add_argument('--max_supervisions', type=int, default=3)
    parser.add_argument('--notes', type=str, default=None)
    parser.add_argument('--force_gray', type=str, default="yes", choices=["yes", "no"])
    parser.add_argument('--normalize', type=str, default="yes", choices=["yes", "no"])
    parser.add_argument('--train', type=str, default="yes", choices=["yes", "no"])
    parser.add_argument('--arch', type=str, default="standard",
                        choices=[ "resnetu", "larger_standard", "resnetunobn", "resnetunoskip",
                                 "resnetunolastskip"]
                                + list(pretrained_segmentation_archs_dim.keys()))
    parser.add_argument('--dataset', type=str, default="emptyspace",
                        choices=["livingroom", "emptyspace", "solid"])
    parser.add_argument('--device', type=str, default="cpu",
                        choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--ref_run', type=str, default=None)
    return parser


if __name__ == "__main__":
    parser = get_runner_parser()
    args_cmd = parser.parse_args()
    run_exp(args_cmd)
