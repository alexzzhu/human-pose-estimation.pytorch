from importlib import reload
import os
import torch

from pytorch_utils import BaseOptions
from event_pose_trainer import EventPoseTrainer

options = BaseOptions()
options.parser.add_argument('--cfg',
                            help='experiment configure file name',
                            required=True,
                            type=str)
options.parser.add_argument('--event_gan_path',
                            default='/NAS/home/event_gan')
options.parser.add_argument(
    '--gan_model_path',
    default='/NAS/home/event_gan/logs/cycle-aux-bigskip-radam/checkpoints/2019_10_06-00_20_44.pt')
args = options.parse_args()

trainer = EventPoseTrainer(args) 

trainer.train()
