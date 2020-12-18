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

start_time = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
writer = SummaryWriter(log_dir=config.training_logs + start_time)
parser = argparse.ArgumentParser("Training script.")
parser.add_argument("-e", "--epochs", default=config.epochs, type=int,
                    help="Set number of epochs.")
parser.add_argument("-b", "--batch-size", default=config.batch_size, type=int,
                    help="Set batch size.")
parser.add_argument("--gpu", default=config.gpu, nargs="?",
                    help="Use graphics card during training.")
parser.add_argument("--learning-rate-generators", default=config.learning_rate_generators, type=float,
                    help="Set learning rate of Generators.")
parser.add_argument("--learning-rate-discriminator-a", default=config.learning_rate_discriminator_a, type=float,
                    help="Set learning rate of Discriminator A.")
parser.add_argument("--learning-rate-discriminator-b", default=config.learning_rate_discriminator_b, type=float,
                    help="Set learning rate of Discriminator B.")
parser.add_argument("--filters-generators", default=config.filters_generators, type=int,
                    help="Set multiplier of convolutional filters in generators.")
parser.add_argument("--depth-generators", default=config.depth_generators, type=int,
                    help="Set depth of Unet generator architecture")
parser.add_argument("--filters-discriminators", default=config.filters_discriminators, type=int,
                    help="Set multiplier of convolutional filters in discriminators.")
parser.add_argument("--depth-discriminators", default=config.depth_discriminators, type=int,
                    help="Set number of convolutional layers in discriminator.")
parser.add_argument("--save-model", default=config.save_model, nargs="?",
                    help="Turn on model saving.")
parser.add_argument("--save-model-epoch", default=config.save_model_epoch,
                    help="Save model every n-th epoch.")
parser.add_argument("--load-model", default="", nargs="?",
                    help="Load saved model from model_path directory. Enter filename as argument.")
args = parser.parse_args()

os.sys.path.append(config.project_root)

dataset = CycleGanDataset()
dataloader = DataLoader(dataset, shuffle=True, num_workers=2, batch_size=args.batch_size, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

netG_A2B = UnetGenerator2D(depth=args.depth_generators, filters=args.filters_generators).to(device).apply(weights_init)
netG_B2A = UnetGenerator2D(depth=args.depth_generators, filters=args.filters_generators).to(device).apply(weights_init)
netD_A = BaseDiscriminator(
    args.filters_discriminators, args.depth_discriminators).to(device).apply(weights_init)
netD_B = BaseDiscriminator(
    args.filters_discriminators, args.depth_discriminators).to(device).apply(weights_init)

epoch_start = 0
if args.load_model:
    checkpoint = torch.load(os.path.join(config.model_path, args.load_model))
    netG_A2B.load_state_dict(checkpoint["gen_a2b"])
    netG_B2A.load_state_dict(checkpoint["gen_b2a"])
    netD_A.load_state_dict(checkpoint["disc_a"])
    netD_B.load_state_dict(checkpoint["disc_b"])

    optimizer_G = torch.optim.Adam(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=checkpoint["lr_g"],
        betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(
        netD_A.parameters(), lr=checkpoint["lr_d_a"], betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(
        netD_B.parameters(), lr=checkpoint["lr_d_b"], betas=(0.5, 0.999))

    optimizer_G.load_state_dict(checkpoint["optim_g"])
    optimizer_D_A.load_state_dict(checkpoint["optim_d_a"])
    optimizer_D_B.load_state_dict(checkpoint["optim_d_b"])
    epoch_start = checkpoint["epoch"]

else:
    optimizer_G = torch.optim.Adam(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=args.learning_rate_generators,
        betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(
        netD_A.parameters(), lr=args.learning_rate_discriminator_a, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(
        netD_B.parameters(), lr=args.learning_rate_discriminator_b, betas=(0.5, 0.999))

cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)

total_batch_counter = 0
for epoch in range(epoch_start, args.epochs):
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
        writer.add_scalar("Loss/Generator Error", errG, total_batch_counter)
        writer.add_scalar("Loss/DiscriminatorA Error", errD_A, total_batch_counter)
        writer.add_scalar("Loss/DiscriminatorB Error", errD_B, total_batch_counter)
        total_batch_counter += 1

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    plt.imshow(denormalize(real_A[0, 0, :, :].detach().cpu()), cmap=plt.cm.gray)
    f.add_subplot(1, 3, 2)
    plt.imshow(denormalize(fake_image_B[0, 0, :, :].detach().cpu()), cmap=plt.cm.gray)
    f.add_subplot(1, 3, 3)
    plt.imshow(denormalize(recovered_image_A[0, 0, :, :].detach().cpu()), cmap=plt.cm.gray)
    f.tight_layout()
    writer.add_figure("Image outputs/A to B to A", f, epoch)

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    plt.imshow(denormalize(real_B[0, 0, :, :].detach().cpu()), cmap=plt.cm.gray)
    f.add_subplot(1, 3, 2)
    plt.imshow(denormalize(fake_image_A[0, 0, :, :].detach().cpu()), cmap=plt.cm.gray)
    f.add_subplot(1, 3, 3)
    plt.imshow(denormalize(recovered_image_B[0, 0, :, :].detach().cpu()), cmap=plt.cm.gray)
    f.tight_layout()
    writer.add_figure("Image outputs/B to A to B", f, epoch)

    if args.save_model and epoch % args.save_model_epoch == 0:
        torch.save({
            "epoch": epoch,
            "gen_a2b": netG_A2B.state_dict(),
            "gen_b2a": netG_B2A.state_dict(),
            "disc_a": netD_A.state_dict(),
            "disc_b": netD_B.state_dict(),
            "optim_g": optimizer_G.state_dict(),
            "optim_d_a": optimizer_D_A.state_dict(),
            "optim_d_b": optimizer_D_B.state_dict(),
            "lr_g": args.learning_rate_generators,
            "lr_d_a": args.learning_rate_discriminator_a,
            "lr_d_b": args.learning_rate_discriminator_b
            }, os.path.join(config.model_path, "{}-epoch-{}.pt".format(start_time, epoch))
        )

writer.flush()
writer.close()
