"""Experiment Configuration"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('PANet')
ex.captured_out_filter = apply_backspaces_and_linefeeds

# add the source code in this experiment round to saving.
source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    """Default configurations"""
    input_size = (417, 417)
    seed = 1234
    cuda_visable = '0, 1, 2, 3, 4, 5, 6, 7'
    gpu_id = 0
    mode = 'test' # 'train' or 'test'


    if mode == 'train':
        dataset = 'VOC' # 'VOC' or 'COCO'
        n_steps = 30000 # number of episodes, namely <Support, Query> pair.
        label_sets = 0  # 0, 1, 2, 3 for 4 folds.
        batch_size = 1  # one episodes <S, Q> for c-way_k-shot training S,Q=<Image, Mask>
        lr_milestones = [10000, 20000, 30000]
        align_loss_scaler = 1
        ignore_label = 255
        print_interval = 100
        save_pred_every = 10000

        # yes or not using Prototype Alignment
        model = {
            'align': True,
        }

        task = {
            'n_ways': 1,
            'n_shots': 1,
            'n_queries': 1,
        }

        optim = {
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }

    elif mode == 'test':
        notrain = False     # True for test stage just using Pretrained VGG16 initialization to Segment. "PANet-init" baseline.
        snapshot = './runs/PANet_VOC_sets_0_1way_1shot_[train]/1/snapshots/30000.pth'
        n_runs = 5          # number of round in test stage.
        n_steps = 1000      # number of episodes, namely <S, Q> pair.
        batch_size = 1      # one episodes <S, Q> for c-way_k-shot training S,Q=<Image, Mask>.
        scribble_dilation = 0
        bbox = False        # yes or not making bounding box to mask image.
        scribble = False    # yes or not making scribble to mask image.

        # Set dataset config from the snapshot string
        if 'VOC' in snapshot:
            dataset = 'VOC'
        elif 'COCO' in snapshot:
            dataset = 'COCO'
        else:
            raise ValueError('Wrong snapshot name !')

        # Set model config from the snapshot string
        # use or not Prototype Alignment
        model = {}
        for key in ['align',]:
            model[key] = key in snapshot

        # Set label_sets from the snapshot string
        label_sets = int(snapshot.split('_sets_')[1][0])

        # Set task config from the snapshot string
        task = {
            'n_ways': int(re.search("[0-9]+way", snapshot).group(0)[:-3]),
            'n_shots': int(re.search("[0-9]+shot", snapshot).group(0)[:-4]),
            'n_queries': 1,
        }

    else:
        raise ValueError('Wrong configuration for "mode" !')

    
    # construct the path string.
    exp_str = '_'.join(
        [dataset,]
        + [key for key, value in model.items() if value]
        + [f'sets_{label_sets}', f'{task["n_ways"]}way_{task["n_shots"]}shot_[{mode}]'])


    # model & pretrained & dataset path.
    path = {
        'log_dir': './runs',
        'init_path': './pretrained_model/vgg16-397923af.pth',
        'VOC':{'data_dir': '/data_smr/dataset/VOC/VOCdevkit/VOC2012/',
               'data_split': 'trainaug',},
        'COCO':{'data_dir': '/data_smr/dataset/COCO/COCO2017/',
                'data_split': 'train',},
    }

@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    # the ex.path = “PANet” in Experiment('PANet').
    # eg. ./runs/PANet_VOC_sets_0_1way_1shot_[train]/1/snapshots/30000.pth
    exp_name = f'{ex.path}_{config["exp_str"]}'
    if config['mode'] == 'test':
        if config['notrain']:
            exp_name += '_notrain'
        if config['scribble']:
            exp_name += '_scribble'
        if config['bbox']:
            exp_name += '_bbox'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
