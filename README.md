# Stochastic Coherence Over Attention Trajectory For Continuous Learning In Video Streams

This repository contains the code, data and selected hyperparameters for the paper **Stochastic Coherence Over Attention Trajectory For Continuous Learning In Video Streams**,
accepted for publication at the IJCAI-ECAI2022 conference. 

*Authors:*  [Matteo Tiezzi](https://mtiezzi.github.io/), Simone Marullo, Lapo Faggi,  Enrico Meloni, Alessandro Betti, Stefano Melacci

You can find [ here the Arxiv pre-print!](https://arxiv.org/abs/2204.12193)


_Notice that reproducibility is not guaranteed by PyTorch across different releases, platforms, hardware. Moreover,
determinism cannot be enforced due to use of PyTorch operations for which deterministic implementations do not exist
(e.g. bilinear upsampling)._

REPOSITORY DESCRIPTION
----------------------

Here we describe the structure of the code repository. We provide in folder `3denv` the new scenes we created for the 3D
Virtual Environment SAILenv, with the corresponding loading instructions. Moreover, we provide the visual streams that
we exploited in our paper in folder data, which were obtained from the rendering of the aforementioned scenes.
The folder structure is the following:

    3denv :                 folder containing the new scenes we produced for the 3D Virtual Environment SAILenv 
    data :                  folder containing the rendered streams
    dpt_intel :             folder of the DPT model (competitor)
    lve :                   main source folder
    exp_runner.py :         experiments runner
    reproduce_runs.txt :    text file containing the command lines (and parameters) to reproduce the main results


HOW TO REPRODUCE THE MAIN RESULTS
---------------------------------

**Note:** In order to use the DPT model, you have to run the "downloader.sh" script  (folder dpt_intel)
in order to download the model weights into the `weights` folder.

**HW requirements:** It is recommended to run experiments on a GPU with large memory (24GB).

Make sure to have Python dependencies (except PyTorch 1.7.1) by running:

```
pip install -r requirements.txt
```

In the `reproduce_runs.txt` file there are the command lines (hence, the experiments parameters) required to reproduce
the experiments of the main results (Table 1).

HOW TO RUN AN EXPERIMENT
------------------------
The script exp_runner.py allows you to run an experiment on one of the three visual streams presented in the paper,
which can be specified by the argument --dataset.

The PyTorch device is chosen through the `--device` argument (`cpu`, `cuda:0`,
`cuda:1`, etc.).

    usage: exp_runner.py [-h] [--laps_unsup LAPS_UNSUP] [--laps_sup LAPS_SUP] [--laps_metrics LAPS_METRICS]
    [--step_size STEP_SIZE] [--lambda_s_in LAMBDA_S_IN] [--lambda_s_out LAMBDA_S_OUT] [--spread_factor SPREAD_FACTOR]
                         [--lambda_t LAMBDA_T] [--num_what NUM_WHAT] [--num_pairs NUM_PAIRS]
                         [--max_supervisions MAX_SUPERVISIONS] [--notes NOTES] [--force_gray {yes,no}]
                         [--normalize {yes,no}] [--train {yes,no}]
                         [--arch {resnetu,larger_standard,resnetunobn,resnetunoskip,resnetunolastskip,
                         deeplab_resnet101_backbone,deeplab_resnet101_classifier,dpt_backbone,dpt_classifier,identity}]
                         [--dataset {livingroom,emptyspace,solid}] [--seed SEED] [--ref_run REF_RUN]

Argument description:

        --laps_unsup :  number of unsupervised laps where the coherence losses are minimized
        --laps_sup : number of laps on which the supervised templates are provided
        --laps_metrics : number of laps on which the metrics are computed (here the model weight are frozen, no learning is happening)
        --step_size : learning rate, \alpha in the paper
        --lambda_s_in :  \lambda_s in the paper
        --lambda_s_out : \lambda_c in the paper
        --lambda_t : \lambda_t in the paper
        --spread_factor: \beta in the paper
        --num_what : output filters "d" in the HourGlass and FCN_ND models
        --num_pairs: number of edges "e" in the stochastic graph
        --max_supervisions : number of supervisions per object
        --force_gray : if "yes", it corresponds to the "BW" of the paper. "no" requires an RGB stream
        --normalize : "yes" exploits unitary norm features and cosine similarity
        --arch : specify the neural architecture to be used (see the arch_groups dictionary in the following for the mappings)
        --dataset  : specify the input stream
        --seed : specify the seed

In the paper we exploited various architectural variations of the neural models, grouped under the names of HourGlass
and FCN-ND. Here we provide a mapping from this groups to the model instances

Syntax:

    "name" == name of the model group in the paper
    "list" == architectural variation belonging to the group

    arch_groups = {
        'hourglass_group':
            {'name': 'HourGlass',  'list': ['resnetu', 'resnetunobn', 'resnetunolastskip', 'resnetunoskip']},
        'fully_convolutional_group':
            {'name': 'FCN-ND', 'list': ['larger_standard']}
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
            {'name': 'Baseline', 'list': ['identity']},
    }


    'resnetu' == ResNetUnet available here  https://github.com/usuyama/pytorch-unet/blob/master/pytorch_resnet18_unet.ipynb
    'resnetunobn' == 'resnetu' without BatchNormalization
    'resnetunoskip' == 'resnetu' with all the skip connections removed
    'resnetunolastskip' == 'resnetu' with only the last skip connection removed

HOW TO EXTRACT THE METRICS
-------------------

The statistics of each experiment are dumped in the `model_folder/results.json` file.

This file contains a dictionary with multiple metrics. The metrics that are reported in Table 1 of the paper are under
the key `f1_window_foa_global`, while Table 2 reports the metric under the key `f1_window_whole_global`.


HUMAN-LIKE FOCUS OF ATTENTION TRAJECTORIES
------------------------------------------

We provide precomputed focus of attention trajectories obtained from the state-of-the-art GEymol model (Zanca et al,
2019)
in each dataset folder (`.foa` files). By default, the `exp_runner.py` script loads these files.

The FOA trajectory depends on the following parameters:

    'alpha_c': \alpha_b in the paper, weight of the brightness mass
    'alpha_of': \alpha_m in the paper, weight of the motion-based mass (modeled according to the optical-flow measure of motion)
    'dissipation': \rho in the paper, inhibitory signal 
    'fixation_threshold_speed': \nu in the paper

Moreover, we specified the initial conditions for the differential equation (1) of the paper:

    a^0 = random position, close to the center of the frame, sampled from uniform distribution
    a^1 = random velocity, close to 0, sampled from uniform distribution

We carried on a preliminary analysis in order to detect the best performing parameters depending on the visual stream at
hand. We report the chosen parameters for each visual stream:

*EmptySpace* with frame width `w` and height `h`

    'alpha_c': 0.1, 
    'alpha_of': 1.0,
    'dissipation': 0.05,
    'fixation_threshold_speed': 0.1 * 0.5 * (w + h)    

*Solid* with frame width `w` and height `h`

    'alpha_c': 1.0, 
    'alpha_of': 1.0,
    'dissipation': 0.005,
    'fixation_threshold_speed': 0.1 * 0.5 * (w + h)

*LivingRoom* with frame width `w` and height `h`

    'alpha_c': 0.1, 
    'alpha_of': 1.0,
    'dissipation': 0.05,
    'fixation_threshold_speed': 0.1 * 0.5 * (w + h)    

Alternatively, if you do not want to use the precomputed FOA trajectory,  you can customize the FOA parameters  in the `exp_runner.py` script and  set to `None`
the `foa_file` argument in the `InputStream` class (in this way, a new trajectory will be computed frame-wise).




Acknowledgement
---------------

This software was developed in the context of some of the activities of the PRIN 2017 project RexLearn, funded by the Italian Ministry of Education, University and Research (grant no. 2017TWNMH2).This software was developed in the context of some of the activities of the PRIN 2017 project RexLearn, funded by the Italian Ministry of Education, University and Research (grant no. 2017TWNMH2).

