# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2

from core.inference import get_max_preds


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    batch_image = batch_image.clamp(0., 1.)
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    parents = [1, 2, 6, 6, 3, 4, 7, 8, 9, -1, 11, 12, 8, 8, 13, 14]
    
    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]
            
            #for joint, joint_vis in zip(joints, joints_vis):
            for i in range(joints.shape[0]):
                curr_joint = ( int(x * width + padding + joints[i, 0]),
                               int(y * height + padding + joints[i, 1]) )

                if joints_vis[0]:
                    cv2.circle(ndarr, curr_joint, 2, (255, 0, 0), 2)

                    if (joints[i, 0] <= 0 and joints[i, 1] <= 0) or \
                       (joints[parents[i], 0] <= 0 and joints[parents[i], 1] <= 0) or \
                       not joints_vis[parents[i]] or parents[i] < 0:
                        continue
                    prev_joint = (int(x * width + padding + joints[parents[i], 0]),
                                  int(y * height + padding + joints[parents[i], 1]))
                    cv2.line(ndarr, curr_joint, prev_joint, (255, 255, 255), 2)

                    
            k = k + 1
    return ndarr.transpose(2, 0, 1)
    #cv2.imwrite(file_name, ndarr)
    
def draw_batch_image_with_multi_skeleton(batch_image, batch_joints, batch_joints_vis,
                                         file_name, nrow=8, padding=2, probs=None):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    #batch_image = batch_image.clamp(0., 1.)
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, False)
    ndarr = grid.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    parents = [1, 2, 6, 6, 3, 4, 7, 8, 9, -1, 11, 12, 8, 8, 13, 14]
    ignore = []#0, 1, 2, 3, 4, 5]

    nmaps = batch_joints.shape[0]
    xmaps = min(nrow, batch_image.shape[0])
    ymaps = int(math.ceil(float(batch_image.shape[0]) / xmaps))
    height = int(batch_image.shape[2] + padding)
    width = int(batch_image.shape[3] + padding)

    for y in range(ymaps):
        for x in range(xmaps):
            for k in range(nmaps):
                joints = batch_joints[k]

                for i in range(joints.shape[0]):
                    curr_joint = (int(x * width + padding + joints[i, 0]),
                                  int(y * height + padding + joints[i, 1]))

                    color = 255
                    if probs is not None:
                        color = int(math.log(probs[k][i] * 255.) * 255. / math.log(255.))
                
                    cv2.circle(ndarr, curr_joint, 2, [color, 0, 0], 2)

                    if (joints[i, 0] <= 0 and joints[i, 1] <= 0) or parents[i] < 0 or \
                       (joints[parents[i], 0] <= 0 and joints[parents[i], 1] <= 0):
                        continue
                    
                    prev_joint = (int(x * width + padding + joints[parents[i], 0]),
                                  int(y * height + padding + joints[parents[i], 1]))
                    
                    cv2.line(ndarr, curr_joint, prev_joint, (0, 0, color), 2)
                
    return ndarr.transpose(2, 0, 1)    
    
def draw_batch_image_with_skeleton(batch_image, batch_joints, batch_joints_vis,
                                   file_name, nrow=8, padding=2, probs=None):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    batch_image = batch_image.clamp(0., 1.)
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    parents = [1, 2, 6, 6, 3, 4, 7, 8, 9, -1, 11, 12, 8, 8, 13, 14]
    ignore = []#0, 1, 2, 3, 4, 5]

    nmaps = batch_joints.shape[0]
    xmaps = min(nrow, 1)
    ymaps = int(math.ceil(float(1) / xmaps))
    height = int(batch_image.shape[2] + padding)
    width = int(batch_image.shape[3] + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]

            for i in range(joints.shape[0]):
                if parents[i] < 0 or i in ignore:
                    continue

                curr_joint = (int(x * width + padding + joints[i, 0]),
                              int(y * height + padding + joints[i, 1]))

                prev_joint = (int(x * width + padding + joints[parents[i], 0]),
                              int(y * height + padding + joints[parents[i], 1]))

                color = 255
                if probs is not None:
                    color = int(math.log(probs[k][i] * 255.) * 255. / math.log(255.))
                
                cv2.circle(ndarr, curr_joint, 2, [color, 0, 0], 2)

                if (joints[i, 0] <= 0 and joints[i, 1] <= 0) or \
                   (joints[parents[i], 0] <= 0 and joints[parents[i], 1] <= 0):
                    continue

                cv2.line(ndarr, curr_joint, prev_joint, (0, 0, color), 2)
                

            k = k + 1

    return ndarr.transpose(2, 0, 1)    

def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        batch_image = batch_image.clamp(0., 1.)
        min = float(batch_image.min())
        max = float(batch_image.max())
        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    #cv2.imwrite(file_name, grid_image)
    return grid_image.transpose(2, 0, 1)

def get_debug_images(input, joints, joints_vis, joints_pred, joints_pred_vis):
    gt_joints = save_batch_image_with_joints(
        input, joints, joints_vis
    )
    pred_joints = save_batch_image_with_joints(
        input, joints_pred, joints_pred_vis
    )
    #gt_heatmap = save_batch_heatmaps(
    #    input, target, '{}_hm_gt.jpg'.format(prefix)
    #)
    #pred_heatmap = save_batch_heatmaps(
    #    input, output, '{}_hm_pred.jpg'.format(prefix)
    #)

    return gt_joints, pred_joints#, gt_heatmap, pred_heatmap

def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )
