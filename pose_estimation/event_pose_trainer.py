import numpy as np
import torch
import time
import os
import torchvision.transforms as transforms
from tqdm import tqdm

import pytorch_utils
from pytorch_utils import CheckpointDataLoader

import _init_paths
from core.config import config, update_config
from core.evaluate import accuracy
from core.loss import JointsMSELoss
from utils.utils import get_optimizer
from utils.vis import get_debug_images
import dataset
import models

def none_safe_collate(batch):
    batch = [x for x in batch if x is not None]
    if batch:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)
    else:
        return {}

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

class EventPoseTrainer(pytorch_utils.BaseTrainer):
    def __init__(self, options, train=True):
        self.is_training = train
        super().__init__(options)
        self.lr_schedulers = {
            k : torch.optim.lr_scheduler.MultiStepLR(
                v,
                config.TRAIN.LR_STEP,
                config.TRAIN.LR_FACTOR) \
            for k, v in self.optimizers_dict.items() }

        for opt in self.optimizers_dict:
            self.lr_schedulers[opt].step()

    def init_fn(self):
        update_config(self.options.cfg)
        
        pretrained_model = config.MODEL.BASE_PRETRAINED
        print("Loading pretrained model from {}".format(pretrained_model))
    
        self.model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
            config, config.MODEL.BASE_PRETRAINED, is_train=True
        )

        if self.options.fine_tune:
            for m in self.model.modules():
                m.requires_grad = False
            for m in self.model.final_layer.modules():
                m.requires_grad = True
        
        self.model = torch.nn.DataParallel(self.model).cuda()

        self.models_dict = { 'model' : self.model }
        self.criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()

        optimizer = get_optimizer(config, self.model)
        self.optimizers_dict = { 'optimizer' : optimizer }

        if 'image' in config.DATASET.DATASET:
            self.gan_model = load_gan_model(self.options.event_gan_path,
                                            self.options.gan_model_path)
            train_dataset = eval('dataset.'+config.DATASET.DATASET)(
                config,
                config.DATASET.ROOT,
                config.DATASET.TRAIN_SET,
                True,
                transforms.ToTensor()
            )
        else:
            self.gan_model = None
            train_dataset = eval('dataset.'+config.DATASET.DATASET)(
                config,
                config.DATASET.ROOT,
                config.DATASET.TRAIN_SET,
                config.DATASET.HDF5_PATH,
                True,
                transforms.ToTensor()
            )

        valid_dataset = None
        if 'test' in config.DATASET.TEST_SET:
            valid_dataset = eval('dataset.dhp19')(#+config.DATASET.DATASET)(
                config,
                config.TEST.ROOT,
                config.DATASET.TEST_SET,
                config.TEST.HDF5_PATH,
                False,
                transforms.Compose([
                    transforms.ToTensor(),
                ])
            )

        sampler = None
        if 'comb' in config.DATASET.DATASET:
            train_dataset, sampler = train_dataset

        self.train_ds = train_dataset
        self.validation_ds = valid_dataset
        self.cdl_kwargs["sampler"] = sampler
        self.cdl_kwargs["collate_fn"] = none_safe_collate

    def gen_joint_vis(self, network_input, image, joints, joints_vis, pred, pred_mask, dhp=False):
        input_vis = torch.sum(network_input, dim=1, keepdim=True).clamp(0., 1.)
        zeros = torch.zeros(input_vis.shape).cpu()
        input_vis_rgb = torch.cat((zeros, zeros, input_vis), dim=1)
        input_vis_rgb = torch.where(input_vis > 0,
                                    input_vis_rgb,
                                    image)
        joint_img = get_debug_images(
            input_vis_rgb, joints, joints_vis,
            pred*4, pred_mask[..., 0:1], dhp=dhp)
        
        return { 'joints': joint_img }
    
    def forward_step(self, batch, train=True, gen_output=False, dhp=False):
        network_input = batch['input']
        target = batch['target']
        target_weight = batch['target_weight']
        image = batch['image'].cpu()
        joints = batch['joints'].cpu().numpy()
        joints_vis = batch['joints_vis'].cpu().numpy()
        model = self.models_dict['model']
        
        if self.gan_model is not None and train:
            with torch.no_grad():
                [network_input] = self.gan_model(network_input)       

        output = model(network_input)

        loss = self.criterion(output, target, target_weight)
        _, avg_acc, cnt, pred, pred_mask = accuracy(output.detach().cpu().numpy(),
                                                    target.detach().cpu().numpy())

        network_input = network_input.cpu()
        
        mask = joints_vis * pred_mask[..., 0:1]
        err = np.mean(np.linalg.norm(((pred*4. - joints) * mask), axis=-1).sum(axis=-1) \
                      / (np.sum(mask, axis=(-2, -1)) + 1e-5))
        if (self.step_count % self.options.summary_steps == 0 and train) or gen_output:
            outputs = self.gen_joint_vis(network_input, image, joints, joints_vis, pred, pred_mask,
                                         dhp=dhp)
        else:
            outputs = {}
            
        losses = { 'loss' : loss if train else loss.cpu().numpy(),
                   'acc' : avg_acc,
                   'err' : err }
        
        return losses, outputs
    
    def train_step(self, input_batch):
        if not input_batch:
            return {}, {}

        optimizer = self.optimizers_dict['optimizer']

        losses, outputs = self.forward_step(input_batch, dhp='dhp' in self.options.name)
        # compute gradient and do update step
        optimizer.zero_grad()
        losses['loss'].backward()
        optimizer.step()

        return losses, outputs

    def test(self):
        test_cdl = { k:v for k,v in self.cdl_kwargs.items() }
        test_cdl["batch_size"] = config.TEST.BATCH_SIZE
        test_cdl["sampler"] = None

        test_data_loader = CheckpointDataLoader(self.validation_ds,
                                                **test_cdl)
        test_data_loader.next_epoch(None)
        print("Running with {} elements and batch size of {}, total {}"
              .format(len(test_data_loader), config.TEST.BATCH_SIZE, len(self.validation_ds)))
        final_outputs = {}
        cumulative_losses = {}
        with torch.no_grad():
            for model in self.models_dict:
                self.models_dict[model].eval()
            i = 0
            for step, batch in enumerate(tqdm(test_data_loader, desc='Validation: ')):
                batch = {k: v.to(self.device) for k,v in batch.items() }
                losses, outputs = self.forward_step(batch, train=False, gen_output=step==0,
                                                    dhp=True)

                if step == 0:
                    final_outputs = outputs

                for k, v in losses.items():
                    if k in cumulative_losses:
                        cumulative_losses[k] += v
                    else:
                        cumulative_losses[k] = v
                i += 1
        cumulative_losses = { k:v/float(i) for k, v in cumulative_losses.items() }
                
        for model in self.models_dict:
            self.models_dict[model].train()
        self.summaries(batch, cumulative_losses, final_outputs, mode="test")

    def summaries(self, input_batch, losses, output, mode='train'):
        #nrow = 4
        self.summary_writer.add_scalar("{}/learning_rate".format(mode),
                                       self.get_lr(),
                                       self.step_count)
        for k, v in losses.items():
            self.summary_writer.add_scalar("{}/{}".format(mode, k), v, self.step_count)
        for k, v in output.items():
            if 'hist' in k:
                self.summary_writer.add_histogram("{}/{}".format(mode, k),
                                                  v, self.step_count)
            else:
                #images = make_grid(v, nrow=nrow)
                self.summary_writer.add_image("{}/{}".format(mode, k),
                                              v, self.step_count)
                
    def train_summaries(self, input_batch, losses, output):
        self.summaries(input_batch,
                       losses,
                       output,
                       mode="train")

