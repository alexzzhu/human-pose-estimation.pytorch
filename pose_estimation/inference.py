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

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

#import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from core.inference import get_max_preds

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Duration
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from realtime_event_network.msg import ImageEventClassification

import dataset
import models
import rospy
import numpy as np
#import cv2

from utils.vis import draw_batch_image_with_skeleton

class EventHumanPose:
    def __init__(self, model_name, model_path):
        self._bridge = CvBridge()
        
        self.model = eval('models.'+model_name+'.get_pose_net')(
            config, is_train=False
        )
        
        self.model = torch.nn.DataParallel(self.model).cuda()

        print('=> loading model from {}'.format(model_path))
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        
        rospy.Subscriber("classification_inputs",
                         ImageEventClassification,
                         self._inference_callback,
                         queue_size=1,
                         buff_size=6553600)
        self._network_time_pub = rospy.Publisher('network_time', Duration, queue_size=1)
        self._results_pub = rospy.Publisher('network_output', Image, queue_size=1)

    def _ros_array_to_np(self, ros_msg):
        n_rows = ros_msg.layout.dim[0].size
        n_cols = ros_msg.layout.dim[1].size
        arr_np = np.reshape(np.array(ros_msg.data),
                            (n_rows, n_cols))
        return arr_np

    def _calc_floor_ceil_delta(self, x):
        x_fl = np.floor(x + 1e-8)
        x_ce = np.ceil(x - 1e-8)
        x_ce_fake = x_fl + 1

        dx_ce = x - x_fl
        dx_fl = x_ce_fake - x
        return [x_fl.astype(np.int32), dx_fl], [x_ce.astype(np.int32), dx_ce]

    def _gen_event_volume(self, events):
        x = events[:, 0]
        y = events[:, 1]
        x *= 256. / 346.
        y *= 256. / 260.
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        
        t = events[:, 2]
        t_scaled = (t - t.min()) * 8. / (t.max() - t.min())
        t_scaled[events[:, 3] <= 0] += 9.
        
        ts_fl, ts_ce = self._calc_floor_ceil_delta(t_scaled)

        event_volume = np.zeros((256, 256, 18))
        np.add.at(event_volume, [y, x, ts_fl[0]], ts_fl[1])
        np.add.at(event_volume, [y, x, ts_ce[0]], ts_ce[1])

        return event_volume[np.newaxis, ...]

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
    
    def _inference_callback(self, image_event_msg):
        t_received = rospy.Time.now()
        events_np = self._ros_array_to_np(image_event_msg.events)

        event_volume = self._gen_event_volume(events_np).transpose(0, 3, 1, 2).astype(np.float32)

        network_start = rospy.Time.now()
        event_volume_torch = torch.from_numpy(event_volume).to('cuda')
        event_volume_torch = self._normalize_event_volume(event_volume_torch)
        pose_output = self.model(event_volume_torch)
        network_time = rospy.Time.now() - network_start
        
        pred_joints, _ = get_max_preds(pose_output.detach().cpu().numpy())

        event_image = event_volume_torch.sum(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        prediction_image = draw_batch_image_with_skeleton(
            event_image, pred_joints*4, np.ones(pred_joints.shape),
            'DHP_pred.jpg', nrow=1
        )
        
        prediction_image = prediction_image.transpose(1, 2, 0)
        #prediction_image = cv2.resize(prediction_image, None, fx=image_scale, fy=image_scale)
        prediction_image_ros = self._bridge.cv2_to_imgmsg(prediction_image, encoding="bgr8")
        
        prediction_image_ros.header.stamp = image_event_msg.header.stamp

        self._results_pub.publish(prediction_image_ros)
        if self._network_time_pub.get_num_connections():
            self._network_time_pub.publish(Duration(network_time))

        return        
        
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
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file
        
def main():
    args = parse_args()
    reset_config(config, args)

    rospy.init_node('event_human_pose')
    
    model_name = rospy.get_param('~model_name')
    model_path = rospy.get_param('~model_path')
    event_human_pose = EventHumanPose(model_name, model_path)
    # cudnn related setting
    #cudnn.benchmark = config.CUDNN.BENCHMARK
    #torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    #torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    

if __name__ == '__main__':
    main()
