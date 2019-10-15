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
from skimage import exposure

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class ImageJointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

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

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def _get_prev_next_images(self, idx):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        prev_path, next_path = self._get_prev_next_images(idx)
        filename = db_rec['filename'] if 'filename' in db_rec else ''

        prev_image = cv2.imread(
            prev_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        next_image = cv2.imread(
            next_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if next_image is None or prev_image is None:
            logger.error('=> fail to read {} or {}'.format(prev_path, next_path))
            raise ValueError('Fail to read {} or {}'.format(prev_path, next_path))
        
        orig_size = prev_image.shape[:2]
        prev_image = cv2.resize(prev_image, tuple(self.image_size))
        next_image = cv2.resize(next_image, tuple(self.image_size))

        image_out = next_image.astype(np.float32)
        
        prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        
        data_numpy = np.concatenate((prev_image, next_image), axis=-1).astype(np.float32)

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']

        joint_scale = (np.array(data_numpy.shape[:2]) / np.array(orig_size))[::-1]
        joints[:, :2] = joints[:, :2] * joint_scale[np.newaxis, :]

        c = c * joint_scale
        s = s * joint_scale
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0
        shift = np.array((0, 0))

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0
            shift = np.clip(np.random.randn(2) * 0.2, -0.2, 0.2)

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                image_out = image_out[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
            data_numpy /= 255.
            image_out /= 255.
            gamma = np.clip(np.random.randn() * 0.2, -0.2, 0.2) + 1.
            data_numpy = exposure.adjust_gamma(data_numpy, gamma)
            image_out = exposure.adjust_gamma(image_out, gamma)
            data_numpy = np.clip(data_numpy, 0., 1.)
            image_out = np.clip(image_out, 0., 1.)

            data_numpy = 2. * (data_numpy - 0.5)
            image_out = 2. * (image_out - 0.5)
                
        trans = get_affine_transform(c, s, r, self.image_size, shift)
        data_numpy = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        image_out = cv2.warpAffine(
            image_out,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        
        if self.transform:
            data_numpy = self.transform(data_numpy)
            image_out = self.transform(image_out)
        
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, :2] = affine_transform(joints[i, 0:2], trans)

        shift_vis = np.all(np.logical_and(joints[:, :2] >= 0,
                                                  joints[:, :2] < 256),
                           axis=1, keepdims=True)
        joints_vis[:, :2] = np.logical_and(joints_vis[:, :2], shift_vis)
        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'filename': filename,
            'imgnum': idx,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
        }

        return data_numpy, target, target_weight, meta, image_out

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
