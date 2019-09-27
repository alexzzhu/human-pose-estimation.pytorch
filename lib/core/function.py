# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import cv2
import torch

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds, get_max_preds
from utils.transforms import flip_back
from utils.vis import get_debug_images, save_batch_image_with_joints, \
    draw_batch_image_with_skeleton


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, n_iters, start_time):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta, image) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)

            input_vis = torch.sum(input, dim=1, keepdim=True).clamp(0., 1.)
            zeros = torch.zeros(input_vis.shape)
            input_vis_rgb = torch.cat((zeros, zeros, input_vis), dim=1)

            input_vis_rgb = torch.where(input_vis > 0,
                                         input_vis_rgb,
                                         image)
            
            gt_joints, \
                pred_joints, \
                gt_heatmap, \
                pred_heatmap = get_debug_images(
                    config, input_vis_rgb, meta, target, pred*4, output,
                    prefix)
            
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer.add_image('train_gt_joints', gt_joints, global_steps)
            writer.add_image('train_pred_joints', pred_joints, global_steps)
            writer.add_image('train_gt_heatmaps', gt_heatmap, global_steps)
            writer.add_image('train_pred_heatmaps', pred_heatmap, global_steps)
            
        writer_dict['train_global_steps'] = global_steps + 1

        if i + n_iters >= len(train_loader):
            return i+1
        if time.time() - start_time > 3550:
            return i+1
    return -1

def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        #for i, (input, target, target_weight, meta) in enumerate(val_loader):
        for i, input in enumerate(val_loader):
            # compute output
            output = model(input)
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5
            pred, _ = get_max_preds(output.detach().cpu().numpy())
            num_images = input.size(0)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            input_vis = input.sum(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            pred_joints = draw_batch_image_with_skeleton(
                input_vis, pred*4, np.ones(pred.shape),
                'DHP_pred.jpg'
            )

            pred_joints = pred_joints.transpose((1, 2, 0))

            cv2.imwrite('images/{:04d}_pred.jpg'.format(i), pred_joints)

            idx += num_images


    return 0


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
