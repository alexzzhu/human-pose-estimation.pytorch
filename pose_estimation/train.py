# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import time
from tqdm import trange

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--event_gan_path',
                        default='/NAS/home/event_gan')
    parser.add_argument('--gan_model_path',
                        default='/NAS/home/event_gan/logs/cycle-aux-bigskip-radam/checkpoints/2019_10_06-00_20_44.pt')
    
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers

def load_gan_model(event_gan_path, gan_model_path):
    import sys
    sys.path.append(event_gan_path)
    from model import unet

    model = unet.UNet(num_input_channels=2,
                      num_output_channels=18,
                      skip_type='concat',
                      activation='relu',
                      num_encoders=4,
                      base_num_channels=32,
                      num_residual_blocks=2,
                      norm='BN',
                      use_upsample_conv=True,
                      with_activation=True,
                      sn=True,
                      multi=False)

    print('=> loading GAN model from {}'.format(gan_model_path))
    checkpoint = torch.load(gan_model_path)['gen']
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    return model

def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    config.MODEL.PRETRAINED = config.MODEL.PRETRAINED if os.path.exists(config.MODEL.PRETRAINED) \
                              else 'models/pytorch/imagenet/resnet50-19c8e357.pth'
    print("Loading pretrained model from {}".format(config.MODEL.PRETRAINED))
    
    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
        final_output_dir)

    checkpoint = torch.load(config.MODEL.PRETRAINED)
    begin_epoch = config.TRAIN.BEGIN_EPOCH
    it = 0
    step = 0
    if 'epoch' in checkpoint:
        begin_epoch = checkpoint['epoch']
        it = checkpoint['iter']
        step = checkpoint['step']
    
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': step,
        'valid_global_steps': 0,
    }

    print("Starting training at epoch {}, step {}, iter {}".format(begin_epoch, step, it))
    
    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    
    #writer_dict['writer'].add_graph(model, (dump_input, ))

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    optimizer = get_optimizer(config, model)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )
    
    # Data loading code
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    if 'image' in config.DATASET.DATASET:
        gan_model = load_gan_model(args.event_gan_path, args.gan_model_path)
        train_dataset = eval('dataset.'+config.DATASET.DATASET)(
            config,
            config.DATASET.ROOT,
            config.DATASET.TRAIN_SET,
            True,
            transforms.ToTensor()
        )
    else:
        gan_model = None
        train_dataset = eval('dataset.'+config.DATASET.DATASET)(
            config,
            config.DATASET.ROOT,
            config.DATASET.TRAIN_SET,
            config.DATASET.HDF5_PATH,
            True,
            transforms.ToTensor()
        )

        
    #valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
    #    config,
    #    config.DATASET.ROOT,
    #    config.DATASET.TEST_SET,
    #    False,
    #    transforms.Compose([
    #        transforms.ToTensor(),
    #        #normalize,
    #    ])
    #)

    sampler = None
    if 'comb' in config.DATASET.DATASET:
        train_dataset, sampler = train_dataset
        config.TRAIN.SHUFFLE = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        sampler=sampler,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    #valid_loader = torch.utils.data.DataLoader(
    #    valid_dataset,
    #    batch_size=config.TEST.BATCH_SIZE*len(gpus),
    #    shuffle=False,
    #    num_workers=config.WORKERS,
    #    pin_memory=True
    #)

    best_perf = 0.0
    best_model = False

    save_checkpoint({
        'epoch': begin_epoch,
        'step': step,
        'iter': it,
        'model': get_model_name(config),
        'state_dict': model.state_dict(),
        #'perf': perf_indicator,
        'optimizer': optimizer.state_dict(),
    }, True, final_output_dir)

    start_time = time.time()
    
    for epoch in trange(begin_epoch, config.TRAIN.END_EPOCH, desc="Epochs: "):
        lr_scheduler.step()
        if epoch < begin_epoch:
            continue
        print("Training!")
        # train for one epoch
        n_steps = train(config, train_loader, model, criterion, optimizer, epoch,
                        final_output_dir, tb_log_dir, writer_dict, it, start_time, gan_model)
        best_model = True
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        if n_steps > 0:
            step += n_steps
        else:
            step += len(train_loader)

        # Return wasn't -1, and hadn't finished a full batch. Therefore, timed out.
        if n_steps > 0 and n_steps + it < len(train_loader):
            save_checkpoint({
                'epoch': epoch,
                'step': step,
                'iter': n_steps,
                'model': get_model_name(config),
                'state_dict': model.state_dict(),
                #'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)
            return
        else:
            # Start new epoch.
            save_checkpoint({
                'epoch': epoch + 1,
                'step': step,
                'iter': 0,
                'model': get_model_name(config),
                'state_dict': model.state_dict(),
                #'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)
            it = 0
        # evaluate on validation set
        #perf_indicator = validate(config, valid_loader, valid_dataset, model,
        #                          criterion, final_output_dir, tb_log_dir,
        #                          writer_dict)
        #
        #if perf_indicator > best_perf:
        #    best_perf = perf_indicator
        #    best_model = True
        #else:
        #    best_model = False
        
        
    
    final_model_state_file = os.path.join(final_output_dir,
                                          'final_model.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

if __name__ == '__main__':
    main()
