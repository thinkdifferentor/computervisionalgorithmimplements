"""Training Script"""
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from config import ex


# _run, _config, _log are sacred build-in fields.
@ex.automain
def main(_run, _config, _log):
    
    # create realation path and save the source code in this experiment round.
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'), exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    # set all random seed
    set_seed(_config['seed'])
    # set on cudnn & accelerate
    cudnn.enabled = True        
    cudnn.benchmark = True
    # set globel gpu
    torch.cuda.set_device(device=_config['gpu_id'])
    # set pytorch running threads
    torch.set_num_threads(4)


    _log.info('###### Create model ######')
    # pretrained_path is VGG16.
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    # model.cuda() load the model to gpu. the better way is model.to(device=[_config['gpu_id'],])
    # make gpu to accelerate the trainning speed.
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    # turn on train mode.
    model.train()


    _log.info('###### Load data ######')
    # get the dataloaders(function for VOC | COCO).
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'COCO':
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    # COCO 81 classes; VOC 21 classes.
    labels = CLASS_LABELS[data_name][_config['label_sets']] 
    # transform define(Resize, RandomMirror) and compose together.
    transforms = Compose([Resize(size=_config['input_size']), RandomMirror()])
    # get dataset and to DataLoader.
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries']
    )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,    # eable pinned memory or page locked memory. default is False.
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    # learning ratet weight_decay in milestones
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    # do not calculate background loss. ignore_label = 255
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
    _log.info('###### Training ######')
    # for each batch. this batch_size = 1.
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        support_images = [[shot.cuda() for shot in way] for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way] for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way] for way in sample_batched['support_mask']]

        query_images = [query_image.cuda() for query_image in sample_batched['query_images']]
        query_labels = torch.cat([query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # Forward and Backward
        optimizer.zero_grad() # clear history gradient in each batch. this can save gpu memory at training stage.
        query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask, query_images)
        query_loss = criterion(query_pred, query_labels)
        loss = query_loss + align_loss * _config['align_loss_scaler']
        # calculate gradient.
        loss.backward()
        # update gradient.
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss)
        _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss

        # print loss
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}')
        # take snapshots
        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(), os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(), os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
