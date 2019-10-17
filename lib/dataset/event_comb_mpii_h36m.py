import dataset
import torch
import numpy as np
import pytorch_utils

class WeightedRandomSampler(pytorch_utils.data_loader.CheckpointSampler):
    """
    Samples from a data_source with weighted probabilities for each element.
    Weights do not need to sum to 1. 
    Typical use case is when you have multiple datasets, the weights for each dataset are
    set to 1/len(ds). This ensures even sampling amongst datasets with different lengths.
    weights - tensor with numel=len(data_source)
    
    """
    def __init__(self, data_source, weights):
        super(WeightedRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.weights = weights

    def next_dataset_perm(self):
        return torch.multinomial(self.weights, len(self.data_source), replacement=True).tolist()

def get_joint_dataset(cfg, root, image_set, hdf5_path, is_train,
                      transform=None):
    datasets = []
    for i in range(len(root)):
        datasets.append(dataset.event_mpii(cfg,
                                           root[i],
                                           image_set[i],
                                           hdf5_path[i],
                                           is_train,
                                           transform))
        
    weights = np.concatenate((np.ones((len(datasets[0]))) * 0.8 / len(datasets[0]),
                              np.ones((len(datasets[1]))) * 0.2 / len(datasets[1])))
    ds = torch.utils.data.ConcatDataset(datasets)
    sampler = WeightedRandomSampler(ds, torch.Tensor(weights))
    
    #weights = np.concatenate((np.ones((len(datasets[0]))) * 0.8 / len(datasets[0]),
    #                          np.ones((len(datasets[1]))) * 0.2 / len(datasets[1])))
    #
    #weights = torch.DoubleTensor(weights)
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(joint_dataset))
    
    return ds, sampler
