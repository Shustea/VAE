"""NICE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, latent_dim,device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )


    def sample(self,sample_size,mu=None,logvar=None):
        '''
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        '''
        if mu==None:
            mu = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        if logvar == None:
            logvar = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        
        z = self.z_sample(mu, logvar)

        generated_samples = self.decoder(self.upsample(z).view(z.shape[0], 64, 7, 7))

        return generated_samples


    def z_sample(self, mu, logvar):
        return mu + (0.5*logvar).exp()*torch.randn_like(mu)

    def loss(self,x,recon,mu,logvar):
        recon_error = F.binary_cross_entropy(recon, x, reduction='sum')

        #from the HW solution assuming q ~ N(0,I)
        kl_divergence = -0.5 * torch.sum((logvar + 1 + - (logvar.exp() + mu**2)))

        return (recon_error + kl_divergence) / x.size(0) #ELBO is calculated per datapoint

    def forward(self, x):
        encoded_x = self.encoder(x)

        mu = self.mu(encoded_x.view(x.shape[0], encoded_x.shape[1] * encoded_x.shape[2] * encoded_x.shape[3]))
        logvar = self.logvar(encoded_x.view(x.shape[0], encoded_x.shape[1] * encoded_x.shape[2] * encoded_x.shape[3]))

        z = self.z_sample(mu, logvar)

        recon = self.decoder(self.upsample(z).view(z.shape[0], 64, 7, 7))

        return self.loss(x, recon, mu, logvar)



