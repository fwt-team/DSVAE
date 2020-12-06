# -*- coding: utf-8 -*-
try:
    import os
    import sys
    import math

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from dsvae.utils import init_weights, d_besseli, besseli
    from dsvae.config import DEVICE
    from vmfmix.von_mises_fisher import VonMisesFisher, HypersphericalUniform

except ImportError as e:
    print(e)
    raise ImportError


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'shape={}'.format(
            self.shape
        )


class Decoder(nn.Module):

    def __init__(self, latent_dim=50, x_shape=(1, 28, 28), cshape=(128, 7, 7), verbose=False):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.ishape = cshape
        self.iels = int(np.prod(self.ishape))
        self.x_shape = x_shape
        self.output_channels = x_shape[0]
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(True),

            nn.Linear(1024, self.iels),
            nn.ReLU(True),
            #
            Reshape(self.ishape),
        )
        self.model = nn.Sequential(
            self.model,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(64, self.output_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):
        gen_img = self.model(x)
        return gen_img.view(x.size(0), *self.x_shape)


class Encoder(nn.Module):

    def __init__(self, input_channels=1, output_channels=64, cshape=(128, 7, 7), r=80, verbose=False):
        super(Encoder, self).__init__()

        self.cshape = cshape
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.r = r
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),

            Reshape(self.lshape),

            nn.Linear(self.iels, 1024),
            nn.ReLU(True),
        )

        self.mu = nn.Linear(1024, self.output_channels)
        self.k = nn.Linear(1024, 1)

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):

        x = self.model(x)
        mu = self.mu(x)

        # We limit kappa to be greater than a certain threshold, because larger kappa will make the cluster more compact.
        k = F.softplus(self.k(x)) + self.r

        mu = mu / mu.norm(dim=1, keepdim=True)
        z = VonMisesFisher(mu, k).rsample()

        return z, mu, k


class VMFMM(nn.Module):

    def __init__(self, n_cluster=10, n_features=10):
        super(VMFMM, self).__init__()

        self.n_cluster = n_cluster
        self.n_features = n_features

        mu = torch.FloatTensor(self.n_cluster, self.n_features).normal_(0, 0.02)
        self.pi_ = nn.Parameter(torch.FloatTensor(self.n_cluster, ).fill_(1) / self.n_cluster, requires_grad=True)
        self.mu_c = nn.Parameter(mu / mu.norm(dim=-1, keepdim=True), requires_grad=True)
        self.k_c = nn.Parameter(torch.FloatTensor(self.n_cluster, ).uniform_(1, 5), requires_grad=True)

    def predict(self, z):

        pi = self.pi_
        mu_c = self.mu_c
        k_c = self.k_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.vmfmm_pdfs_log(z, mu_c, k_c))

        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)

    def sample_by_k(self, k, num=10):

        mu = self.mu_c[k:k+1]
        k = self.k_c[k].view((1, 1))
        z = None
        for i in range(num):
            _z = VonMisesFisher(mu, k).rsample()
            if z is None:
                z = _z
            else:
                z = torch.cat((z, _z))
        return z

    def vmfmm_pdfs_log(self, x, mu_c, k_c):

        VMF = []
        for c in range(self.n_cluster):
            VMF.append(self.vmfmm_pdf_log(x, mu_c[c:c + 1, :], k_c[c]).view(-1, 1))
        return torch.cat(VMF, 1)

    @staticmethod
    def vmfmm_pdf_log(x, mu, k):
        D = x.size(1)
        log_pdf = (D / 2 - 1) * torch.log(k) - D / 2 * math.log(math.pi) - torch.log(besseli(D / 2 - 1, k)) \
                  + x.mm(torch.transpose(mu, 1, 0) * k)
        return log_pdf

    def vmfmm_Loss(self, z, z_mu, z_k):

        det = 1e-10
        pi = self.pi_
        mu_c = self.mu_c
        k_c = self.k_c

        D = self.n_features
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.vmfmm_pdfs_log(z, mu_c, k_c)) + det
        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))  # batch_size*Clusters

        # batch * n_cluster
        e_k_mu_z = (d_besseli(D / 2 - 1, z_k) * z_mu).mm((k_c.unsqueeze(1) * mu_c).transpose(1, 0))

        # batch * 1
        e_k_mu_z_new = torch.sum((d_besseli(D / 2 - 1, z_k) * z_mu) * (z_k * z_mu), 1, keepdim=True)

        # e_log_z_x
        Loss = torch.mean((D * ((D / 2 - 1) * torch.log(z_k) - D / 2 * math.log(math.pi) - torch.log(besseli(D / 2 - 1, z_k)) + e_k_mu_z_new)))

        # e_log_z_c
        Loss -= torch.mean(torch.sum(yita_c * (
                D * ((D / 2 - 1) * torch.log(k_c) - D / 2 * math.log(math.pi) - torch.log(besseli(D / 2 - 1, k_c)) + e_k_mu_z)), 1))

        Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / yita_c), 1))
        return Loss
