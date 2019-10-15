# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .event_mpii import EventMPIIDataset as event_mpii
from .image_mpii import ImageMPIIDataset as image_mpii
from .DHP19Dataset import DHP19Dataset as dhp19
from .event_comb_mpii_h36m import get_joint_dataset as event_comb_mpii_h36m
from .image_comb_mpii_h36m import get_joint_image_dataset as image_comb_mpii_h36m
#from .coco import COCODataset as coco
