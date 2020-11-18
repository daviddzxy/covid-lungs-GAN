import os
import torch
import itertools
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import CycleGanDataset
from generators import UnetGenerator2D
from discriminators import BaseDiscriminator
from utils import weights_init, denormalize
from matplotlib import pyplot as plt
import config
import argparse


start_time = datetime.today().strftime('%d-%m-%Y-%H:%M:%S')
writer = SummaryWriter(log_dir=config.training_logs + start_time)
parser = argparse.ArgumentParser("Training script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-e", "--epochs", default=20, type=int, help="Set number of epochs.")
parser.add_argument("-b", "--batch-size", default=2, type=int, help="Set batch size.")
parser.add_argument("--gpu", default=True, nargs="?", help="Use graphics card during training.")
parser.add_argument("--learning-rate-generators", default=0.0002, type=float, help="Set learning rate of "
                                                                                   "Generators.")
parser.add_argument("--learning-rate-discriminator-a", default=0.0002, type=float, help="Set learning rate "
                                                                                        "of Discriminator A.")
parser.add_argument("--learning-rate-discriminator-b", default=0.0002, type=float, help="Set learning rate "
                                                                                        "of Discriminator B.")
parser.add_argument("--filters-generators", default=2, type=int, help="Set multiplier of convolutional filters "
                                                                      "in generators.")
parser.add_argument("--filters-discriminators", default=2, type=int, help="Set multiplier of convolutional "
                                                                          "filters in discriminators.")
parser.add_argument("--convolutional-layers-discriminators", default=2, type=int, help="Set number of "
                                                                                       "convolutional layers "
                                                                                       "in discriminator.")

args = parser.parse_args()

os.sys.path.append(config.project_root)

dataset = CycleGanDataset()
dataloader = DataLoader(dataset, shuffle=True, num_workers=2, batch_size=args.batch_size, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

netG_A2B = UnetGenerator2D(args.filters_generators).to(device).apply(weights_init)
netG_B2A = UnetGenerator2D(args.filters_generators).to(device).apply(weights_init)
netD_A = BaseDiscriminator(
    args.filters_discriminators, args.convolutional_layers_discriminators).to(device).apply(weights_init)
netD_B = BaseDiscriminator(
    args.filters_discriminators, args.convolutional_layers_discriminators).to(device).apply(weights_init)

cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)

optimizer_G = torch.optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=args.learning_rate_generators, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(
    netD_A.parameters(), lr=args.learning_rate_discriminator_a, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(
    netD_B.parameters(), lr=args.learning_rate_discriminator_b, betas=(0.5, 0.999))

identity_losses = []
gan_losses = []
cycle_losses = []

for epoch in range(0, args.epochs):
    for i, data in enumerate(dataloader):
        real_A, real_B = data
        real_A, real_B = real_A.float().to(device), real_B.float().to(device)

        optimizer_G.zero_grad()

        # Identity loss
        # G_B2A(A) should equal A if real A is fed
        identity_image_A = netG_B2A(real_A)
        loss_identity_A = identity_loss(identity_image_A, real_A) * 5.0
        # G_A2B(B) should equal B if real B is fed
        identity_image_B = netG_A2B(real_B)
        loss_identity_B = identity_loss(identity_image_B, real_B) * 5.0

        # GAN loss
        # GAN loss D_A(G_A(A))
        fake_image_A = netG_B2A(real_B)
        fake_output_A = netD_A(fake_image_A)

        # initialize labels
        real_label = torch.ones(fake_output_A.shape).to(device)
        fake_label = torch.zeros(fake_output_A.shape).to(device)

        loss_GAN_B2A = adversarial_loss(fake_output_A, real_label)
        # GAN loss D_B(G_B(B))
        fake_image_B = netG_A2B(real_A)
        fake_output_B = netD_B(fake_image_B)
        loss_GAN_A2B = adversarial_loss(fake_output_B, real_label)

        # Cycle loss
        recovered_image_A = netG_B2A(fake_image_B)
        loss_cycle_ABA = cycle_loss(recovered_image_A, real_A) * 10.0

        recovered_image_B = netG_A2B(fake_image_A)
        loss_cycle_BAB = cycle_loss(recovered_image_B, real_B) * 10.0

        # Combined loss and calculate gradients
        errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

        # Calculate gradients for G_A and G_B
        errG.backward()
        # Update G_A and G_B's weights
        optimizer_G.step()

        ##############################################
        # (2) Update D network: Discriminator A
        ##############################################

        # Set D_A gradients to zero
        optimizer_D_A.zero_grad()

        # Real A image loss
        real_output_A = netD_A(real_A)
        errD_real_A = adversarial_loss(real_output_A, real_label)

        # Fake A image loss
        fake_output_A = netD_A(fake_image_A.detach())
        errD_fake_A = adversarial_loss(fake_output_A, fake_label)

        # Combined loss and calculate gradients
        errD_A = (errD_real_A + errD_fake_A) / 2

        # Calculate gradients for D_A
        errD_A.backward()
        # Update D_A weights
        optimizer_D_A.step()

        ##############################################
        # (3) Update D network: Discriminator B
        ##############################################

        # Set D_B gradients to zero
        optimizer_D_B.zero_grad()

        # Real B image loss
        real_output_B = netD_B(real_B)
        errD_real_B = adversarial_loss(real_output_B, real_label)

        # Fake B image loss
        fake_output_B = netD_B(fake_image_B.detach())
        errD_fake_B = adversarial_loss(fake_output_B, fake_label)

        # Combined loss and calculate gradients
        errD_B = (errD_real_B + errD_fake_B) / 2

        # Calculate gradients for D_B
        errD_B.backward()
        # Update D_B weights
        optimizer_D_B.step()

        # Logging
        writer.add_scalar("Loss/Generator Error", errG, i)
        writer.add_scalar("Loss/DiscriminatorA Error", errD_A, i)
        writer.add_scalar("Loss/DiscriminatorB Error", errD_B, i)

        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(denormalize(real_A[0, 0, :, :].detach().cpu()), cmap=plt.cm.gray)
        f.add_subplot(1, 2, 2)
        plt.imshow(denormalize(fake_image_B[0, 0, :, :].detach().cpu()), cmap=plt.cm.gray)
        writer.add_figure("Image outputs/A to B", f, i)
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(denormalize(real_B[0, 0, :, :].detach().cpu()), cmap=plt.cm.gray)
        f.add_subplot(1, 2, 2)
        plt.imshow(denormalize(fake_image_A[0, 0, :, :].detach().cpu()), cmap=plt.cm.gray)
        writer.add_figure("Image outputs/B to A", f, i)

writer.flush()
writer.close()
