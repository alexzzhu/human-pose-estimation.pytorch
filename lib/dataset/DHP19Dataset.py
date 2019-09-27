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

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from utils.event_utils import gen_discretized_event_volume

logger = logging.getLogger(__name__)


class DHP19Dataset(Dataset):
    def __init__(self, cfg, root, image_set, hdf5_path, is_train, transform=None):
        
        self.is_train = is_train
        self.hdf5_path = hdf5_path
        self.n_events_per_window = cfg.DATASET.N_EVENTS_PER_WINDOW
        
        self.image_size = cfg.MODEL.IMAGE_SIZE

        self.transform = transform

        self.load()
        self.close()

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return self.n_el

    def normalize_event_volume(self, event_volume):
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
    
    def load(self):
        self.sequence = h5py.File(self.hdf5_path, 'r')
        
        cam0 = self.sequence['cam3']
        ts = np.array(cam0['timeStamp'], dtype=np.float32)[0, ...] * 1e-6
        ts = ts - ts[0]
        p = (np.array(cam0['polarity'], dtype=np.float32)[0, ...] - 0.5) * 2.
        x = np.array(cam0['x'], dtype=np.float32)[0, ...]
        y = 259. - np.array(cam0['y'], dtype=np.float32)[0, ...]
        
        self.events = np.stack((x, y, ts, p), axis=-1)
        self.n_el = int(self.events.shape[0] / self.n_events_per_window)
        self.loaded = True

    def close(self):
        self.events = None
        self.sequence.close()
        self.sequence = None
        self.loaded = False
    
    def __getitem__(self, idx):
        if not self.loaded:
            self.load()

        ind = idx * self.n_events_per_window
        curr_events = self.events[ind:ind+self.n_events_per_window, ...].astype(np.float32)

        curr_events = torch.tensor(curr_events)
        event_volume = gen_discretized_event_volume(curr_events, [18, 260, 346])
        event_volume = event_volume[:, (260-256)//2:260-(260-256)//2, (346-256)//2:346-(346-256)//2]
        event_volume = self.normalize_event_volume(event_volume)

        return event_volume
