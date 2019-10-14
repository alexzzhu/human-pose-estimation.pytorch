import dataset
import torch
import numpy as np

def get_joint_dataset(cfg, root, image_set, hdf5_path, is_train, transform=None):
    datasets = []
    for i in range(len(root)):
        datasets.append(dataset.event_mpii(cfg,
                                           root[i],
                                           image_set[i],
                                           hdf5_path[i],
                                           is_train,
                                           transform))

    joint_dataset = torch.utils.data.ConcatDataset(datasets)

    
    
    weights = np.concatenate((np.ones((len(datasets[0]))) * 0.8 / len(datasets[0]),
                              np.ones((len(datasets[1]))) * 0.2 / len(datasets[1])))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(joint_dataset))
    
    return joint_dataset, sampler
