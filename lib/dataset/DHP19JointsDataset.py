# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
import h5py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from utils.event_utils import gen_event_volume_np

logger = logging.getLogger(__name__)


class DHP19JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, hdf5_path, is_train, transform=None):
        self.num_joints = 16
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.hdf5_path = hdf5_path

        self.split = 'avg' if 'avg' in image_set else 'raw'
        imgsetpath = os.path.join(root, '{}.txt'.format(image_set))
        with open(imgsetpath) as f:
            self.image_ids = f.read().splitlines()

        if is_train:
            if 'train' in image_set:
                self.image_ids = self.image_ids[::10]
            else:
                self.image_ids = self.image_ids[::100]
        else:
            self.image_ids = self.image_ids[::10]
            
        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.transform = transform
        self.db = []

        self.load()
        self.close()

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.image_ids)

    def load(self):
        self.sequence = h5py.File(self.hdf5_path, 'r')
        self.loaded = True

    def close(self):
        self.sequence.close()
        self.sequence = None
        self.loaded = False

    def _normalize_event_volume(self, event_volume):
        event_volume_flat = event_volume.contiguous().view(-1)
        nonzero = torch.nonzero(event_volume_flat)
        nonzero_values = event_volume_flat[nonzero]
        if nonzero_values.shape[0]:
            lower = torch.kthvalue(nonzero_values,
                                   max(int(0.02 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            upper = torch.kthvalue(nonzero_values,
                                   max(int(0.98 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            max_val = max(abs(lower), upper)
            event_volume = torch.clamp(event_volume, -max_val, max_val)
            event_volume /= max_val
        return event_volume

    def _get_event_volume(self, data, idx):
        all_events = data['events']
        n_events = all_events.shape[0]
        if self.split == 'avg':
            event_ind = data['{}_inds'.format(self.split)][0, idx]
        else:
            event_ind = data['{}_inds'.format(self.split)][idx]
        events = all_events[max(event_ind-3500, 0):min(event_ind+3500, n_events), :]
        all_events = None
        event_volume = gen_event_volume_np(events, (260, 346, 18))[0].astype(np.float32)
        return event_volume
        
    def _get_label(self, data, idx):
        c = data['{}_center'.format(self.split)][idx]
        s = data['{}_scale'.format(self.split)][idx] * 1.5
        joints = data['{}_part'.format(self.split)][idx].T
        joints_vis = data['{}_mask'.format(self.split)][idx].T

        return c, s, joints, joints_vis
        
    def __getitem__(self, idx):
        #if not self.loaded:
        #    self.load()

        sequence = h5py.File(self.hdf5_path, 'r')
            
        image_id = self.image_ids[idx]
        subj, sess, mov, cam, it = image_id.split(' ')
        data = sequence[subj][sess][mov][cam]
        data_numpy = self._get_event_volume(data, int(it))
        c, s, joints, joints_vis = self._get_label(data, int(it))
        data = None
        sequence.close()
        sequence = None

        orig_size = (260, 346)
        joint_scale = (np.array(data_numpy.shape[:2]) / np.array(orig_size))[::-1]
        joints[:, :2] = joints[:, :2] * joint_scale[np.newaxis, :]

        c = c * joint_scale
        s = s * joint_scale
        r = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0
            
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)

        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        input = self._normalize_event_volume(input)
        
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        shift_vis = np.all(np.logical_and(joints[:, :2] >= 0,
                                                  joints[:, :2] < 256),
                           axis=1, keepdims=True)
        joints_vis[:, :2] = np.logical_and(joints_vis[:, :2], shift_vis)
                
        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        output = { 'input' : input,
                   'target' : target,
                   'target_weight' : target_weight,
                   'image' : input.sum(dim=0, keepdim=True),
                   'joints' : joints,
                   'joints_vis' : joints_vis }
        return output
    
    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
