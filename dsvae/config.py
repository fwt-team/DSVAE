# encoding: utf-8
import os
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(DEVICE))
# Local directory of CypherCat API
DUALSVAE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory containing entire repo
REPO_DIR = os.path.split(DUALSVAE_DIR)[0]

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'datasets')

# Local directory for runs
RUNS_DIR = os.path.join(REPO_DIR, 'runs')

# difference datasets config
# pretrain_batch_size, train_batch_size, latent_dim, picture_size, cshape, all_data_size, pre_epoch, pre_lr, train_lr
DATA_PARAMS = {
    'mnist': (700, 256, 7, (1, 28, 28), (128, 7, 7), 70000, 50, 2e-3, 1e-3),
}
