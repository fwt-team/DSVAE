# Deep Clustering Analysis via Dual Variational Autoencoder with Spherical Latent Embeddings

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
conda activate Test
```

## File

    datasets/  # container of data  
    dsvae/     # core code  
    train.py   # training code  
    vmfmix/    # files of von-Mises Fisher mixture mode  
    runs/      # runing result  

## Training

To train the model(s) in the paper, run this command:  

    __params:__  
    -r   # name of runing folder, default is dsvae  
    -n   # training epoch, default is 150  
    -s   # data set name, default is mnist  
    -v   # version of training, default is 1  

```train
python train.py -s mnist -n 150 -v 1 -r dsvae
```

Note: the reparameterization trick code is taken from (https://github.com/nicola-decao/s-vae-pytorch)


