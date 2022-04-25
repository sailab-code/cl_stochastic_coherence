arch_groups = {
    'dpt_classifier_group':
        {'name': 'DPT-C', 'list': ['dpt_classifier']},
    'dpt_backbone_group':
        {'name': 'DPT-B', 'list': ['dpt_backbone']},
    'deeplab_classifier_group':
        {'name': 'DeepLabV3-C',
         'list': ['deeplab_resnet101_classifier']},
    'deeplab_backbone_group':
        {'name': 'DeepLabV3-B',
         'list': ['deeplab_resnet101_backbone']},
    'identity_group':
        {'name': 'Identity (baseline)', 'list': ['identity']},
    'lve_hourglass_group':
        {'name': 'HourGlass', 'multirun':True,
         'list': ['resnetu', 'resnetunobn', 'resnetunolastskip', 'resnetunoskip']},
    'lve_fully_connected_group':
        {'multirun':True, 'name': 'FCN-ND', 'list': ['larger_standard']}
}

key_metric = "f1_window_foa_global"
datasets = ["emptyspace", "solid", "livingroom"]

dataslugs_name_mapping = {"empty_space_bench_BW": 'EmptySpace (BW)',
                          "empty_space_bench_RGB": 'EmptySpace (RGB)',
                          "solid_benchmark_BW": 'Solid (BW)', "toy_bench_BW": 'LivingRoom (BW)',
                          "toy_bench_RGB": 'LivingRoom (RGB)'}
metrics_name_mapping = {"f1_window_foa_global": "F1 FOA", "f1_window_whole_global": "F1 frame", "acc_window_whole_global": "Acc. frame"}
params_name_mapping = {'max_supervisions': 'Supervisions', 'repetitions_unsup': 'Unsupervised repetitions', 'num_pairs': 'Pairs (attention graph)'}

sailenv_settings = {
    "depth_frame_active": False,
    "flow_frame_active": True,
    "object_frame_active": False,
    "main_frame_active": True,
    "category_frame_active": False,
    "gzip": False,
    "scene": 2
}
pretrained_segmentation_archs_dim = {
    "deeplab_resnet101_backbone": 2048,
    "deeplab_resnet101_classifier": 256,
    "dpt_backbone": 256,
    "dpt_classifier": 256,
    "identity": 3
}