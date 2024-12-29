"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
from VAE import Model
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def train(vae, trainloader, optimizer, epoch, writer):
    vae.train()  # set to training mode
    loss_list = []

    for inputs,_ in trainloader:
        loss = vae(inputs.to(vae.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(-loss.item())

    if writer:
        writer.add_scalar("Epoch train ELBO", sum(loss_list) / len(loss_list), epoch)
        writer.close()


def test(vae, testloader, filename, epoch, writer):
    loss_list = []

    vae.eval()  # set to inference mode
    with torch.no_grad():
        samples = vae.sample(100).cpu()
        # a,b = samples.min(), samples.max()
        # samples = (samples-a)/(b-a+1e-10) 
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './VAE/samples/' + filename.split('.')[0] + 'epoch%d.png' % epoch)
        for inputs,_ in testloader:
        
            loss = vae(inputs.to(vae.device))

            loss_list.append(-loss.item())

    if writer:
        writer.add_scalar("Epoch test ELBO", sum(loss_list) / len(loss_list), epoch)
        writer.close()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)), #dequantization
        transforms.Normalize((0.,), (257./256.,)), #rescales to [0,1]

    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'mid%d_' % args.latent_dim

    vae = Model(latent_dim=args.latent_dim,device=device).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir='logs/' + filename + '_logs')

    for epoch in tqdm(range(args.epochs)):
        train(vae, trainloader, optimizer, epoch, writer)
        test(vae, testloader, filename, epoch, writer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='fashion-mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
