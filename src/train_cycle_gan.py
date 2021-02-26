import os
import torch
import itertools
import pickle
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import CycleGanDataset
from generators import UnetGenerator2D, ResNetGenerator2D
from discriminators import PatchGanDiscriminator
from utils import weights_init, denormalize, create_figure, Buffer
from transformations import RandomRotation, Crop, ApplyMask, Normalize
import argparse
import config
from config import cyclegan_parameters as parameters

start_time = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
writer = SummaryWriter(log_dir=config.training_logs + start_time)
parser = argparse.ArgumentParser("Training script.")
parser.add_argument("-e", "--epochs", default=parameters["epochs"], type=int,
                    help="Set number of epochs.")
parser.add_argument("-b", "--batch-size", default=parameters["batch_size"], type=int,
                    help="Set batch size.")
parser.add_argument("--gpu", default=parameters["gpu"], nargs="?",
                    help="Use graphics card during training.")
parser.add_argument("--generator", default=parameters["generators"], nargs="?", choices=["Unet", "Resnet"],
                    help="Use graphics card during training.")
parser.add_argument("--learning-rate-generators", default=parameters["learning_rate_generators"], type=float,
                    help="Set learning rate of Generators.")
parser.add_argument("--learning-rate-discriminator-a", default=parameters["learning_rate_discriminator_a"], type=float,
                    help="Set learning rate of Discriminator A.")
parser.add_argument("--learning-rate-discriminator-b", default=parameters["learning_rate_discriminator_b"], type=float,
                    help="Set learning rate of Discriminator B.")
parser.add_argument("--filters-generators", default=parameters["filters_generators"], type=int,
                    help="Set multiplier of convolutional filters in generators.")
parser.add_argument("--depth-generators", default=parameters["depth_generators"], type=int,
                    help="Set depth of generator architecture.")
parser.add_argument("--filters-discriminators", default=parameters["filters_discriminators"], type=int,
                    help="Set multiplier of convolutional filters in discriminators.")
parser.add_argument("--depth-discriminators", default=parameters["depth_discriminators"], type=int,
                    help="Set number of convolutional layers in discriminator.")
parser.add_argument("--identity-weight", default=parameters["identity_weight"], type=float,
                    help="Set weight of identity loss function.")
parser.add_argument("--cycle-weight", default=parameters["cycle_weight"], type=float,
                    help="Set weight of cycle loss function.")
parser.add_argument("--save-model", default=parameters["save_model"], nargs="?",
                    help="Turn on model saving.")
parser.add_argument("--load-model", default="", nargs="?",
                    help="Load saved model from model_path directory. Enter filename as argument.")
parser.add_argument("--learning-rate-decay", type=float, default=parameters["learning_rate_decay"], nargs=2,
                    help="Set learning rate decay of generators. First argument is value of decay factor,"
                         " second value is period of learning rate decay")
parser.add_argument("--random-rotation", type=int, default=parameters["random_rotation"],
                    help="Set max degrees of random rotation.")
parser.add_argument("--crop", type=int, default=parameters["crop"], help="Set length of image crop.")
args = parser.parse_args()

os.sys.path.append(config.project_root)

rotation = None
if args.random_rotation != 0:
    rotation = RandomRotation(args.random_rotation)

crop = None
if args.crop != 0:
    crop = Crop([args.crop, args.crop])

normalize = None
with open(config.cyclegan_dataset_metadata, "rb") as handle:
    metadata = pickle.load(handle)
    normalize = Normalize(metadata["min"], metadata["max"])

mask = ApplyMask(config.mask_values["non_lung_tissue"])

dataset = CycleGanDataset(images_A=config.cyclegan_data_train["A"],
                          images_B=config.cyclegan_data_train["B"],
                          mask=mask,
                          rotation=rotation,
                          crop=crop,
                          normalize=normalize
                          )
dataloader = DataLoader(dataset, shuffle=True, num_workers=2, batch_size=args.batch_size, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

if args.generator == "Unet":
    netG_A2B = UnetGenerator2D(depth=args.depth_generators,
                               filters=args.filters_generators).to(device).apply(weights_init)
    netG_B2A = UnetGenerator2D(depth=args.depth_generators,
                               filters=args.filters_generators).to(device).apply(weights_init)
elif args.generator == "Resnet":
    netG_A2B = ResNetGenerator2D(resnet_depth=args.depth_generators,
                                 filters=args.filters_generators).to(device).apply(weights_init)
    netG_B2A = ResNetGenerator2D(resnet_depth=args.depth_generators,
                                 filters=args.filters_generators).to(device).apply(weights_init)

netD_A = PatchGanDiscriminator(filters=args.filters_discriminators,
                               depth=args.depth_discriminators).to(device).apply(weights_init)

netD_B = PatchGanDiscriminator(filters=args.filters_discriminators,
                               depth=args.depth_discriminators).to(device).apply(weights_init)

optimizer_G = torch.optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=args.learning_rate_generators,
    betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(
    netD_A.parameters(), lr=args.learning_rate_discriminator_a, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(
    netD_B.parameters(), lr=args.learning_rate_discriminator_b, betas=(0.5, 0.999))

scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G,
                                              gamma=args.learning_rate_decay[0],
                                              step_size=args.learning_rate_decay[1])

epoch_start = 0
if args.load_model:
    checkpoint = torch.load(os.path.join(config.model_path, args.load_model))
    netG_A2B.load_state_dict(checkpoint["gen_a2b"])
    netG_B2A.load_state_dict(checkpoint["gen_b2a"])
    netD_A.load_state_dict(checkpoint["disc_a"])
    netD_B.load_state_dict(checkpoint["disc_b"])
    optimizer_G.load_state_dict(checkpoint["optim_g"])
    optimizer_D_A.load_state_dict(checkpoint["optim_d_a"])
    optimizer_D_B.load_state_dict(checkpoint["optim_d_b"])
    scheduler_G.load_state_dict(checkpoint["scheduler_g"])
    epoch_start = checkpoint["epoch"]

cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)
total_batch_counter = 0

buffer_A = Buffer(parameters["buffer_length"])
buffer_B = Buffer(parameters["buffer_length"])

try:
    for epoch in range(epoch_start, args.epochs):
        print("Current epoch {}.".format(epoch))
        for i, data in enumerate(dataloader):
            real_A, real_B = data
            real_A, real_B = real_A.float().to(device), real_B.float().to(device)
            optimizer_G.zero_grad()

            identity_image_A = netG_B2A(real_A)
            identity_image_B = netG_A2B(real_B)

            loss_identity_A = identity_loss(identity_image_A, real_A) * args.identity_weight
            loss_identity_B = identity_loss(identity_image_B, real_B) * args.identity_weight

            fake_image_A = netG_B2A(real_B)
            fake_image_B = netG_A2B(real_A)

            fake_output_A = netD_A(fake_image_A)
            fake_output_B = netD_B(fake_image_B)

            # initialize labels
            real_label = torch.ones(fake_output_A.shape).to(device)
            fake_label = torch.zeros(fake_output_A.shape).to(device)

            loss_GAN_B2A = adversarial_loss(fake_output_A, real_label)
            loss_GAN_A2B = adversarial_loss(fake_output_B, real_label)

            recovered_image_A = netG_B2A(fake_image_B)
            recovered_image_B = netG_A2B(fake_image_A)

            loss_cycle_ABA = cycle_loss(recovered_image_A, real_A) * args.cycle_weight
            loss_cycle_BAB = cycle_loss(recovered_image_B, real_B) * args.cycle_weight

            errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            errG.backward()
            optimizer_G.step()

            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()

            real_output_A = netD_A(real_A)
            real_output_B = netD_B(real_B)

            errD_real_A = adversarial_loss(real_output_A, real_label)
            errD_real_B = adversarial_loss(real_output_B, real_label)

            fake_image_A = buffer_A.push_and_pop(fake_image_A)
            fake_image_B = buffer_A.push_and_pop(fake_image_B)

            fake_output_A = netD_A(fake_image_A.detach())
            fake_output_B = netD_B(fake_image_B.detach())

            errD_fake_A = adversarial_loss(fake_output_A, fake_label)
            errD_fake_B = adversarial_loss(fake_output_B, fake_label)

            errD_A = (errD_real_A + errD_fake_A) / 2
            errD_B = (errD_real_B + errD_fake_B) / 2

            errD_A.backward()
            optimizer_D_A.step()

            errD_B.backward()
            optimizer_D_B.step()

            # Logging
            writer.add_scalar("Loss/Generator Error", errG, total_batch_counter)
            writer.add_scalar("Loss/DiscriminatorA Error", errD_A, total_batch_counter)
            writer.add_scalar("Loss/DiscriminatorB Error", errD_B, total_batch_counter)
            total_batch_counter += 1

        scheduler_G.step()

        with torch.no_grad():
            netG_A2B.eval()
            netG_B2A.eval()
            fake_B = netG_A2B(real_A)
            fake_A = netG_B2A(real_B)
            recovered_A = netG_B2A(fake_B)
            recovered_B = netG_A2B(fake_A)
            netG_A2B.train()
            netG_B2A.train()

        f = create_figure([denormalize(real_A[0, 0, :, :].detach().cpu()),
                           denormalize(fake_B[0, 0, :, :].detach().cpu()),
                           denormalize(recovered_A[0, 0, :, :].detach().cpu())],
                          figsize=(12, 4)
                          )
        writer.add_figure("Image outputs/A to B to A", f, epoch)

        f = create_figure([denormalize(real_B[0, 0, :, :].detach().cpu()),
                           denormalize(fake_A[0, 0, :, :].detach().cpu()),
                           denormalize(recovered_B[0, 0, :, :].detach().cpu())],
                          figsize=(12, 4)
                          )
        writer.add_figure("Image outputs/B to A to B", f, epoch)

        state = {
            "epoch": epoch,
            "gen_a2b": netG_A2B.state_dict(),
            "gen_b2a": netG_B2A.state_dict(),
            "disc_a": netD_A.state_dict(),
            "disc_b": netD_B.state_dict(),
            "optim_g": optimizer_G.state_dict(),
            "optim_d_a": optimizer_D_A.state_dict(),
            "optim_d_b": optimizer_D_B.state_dict(),
            "scheduler_g": scheduler_G.state_dict()
        }

except KeyboardInterrupt:
    if args.save_model:
        torch.save(state, os.path.join(config.model_path, "{}.pt".format(start_time)))


if args.save_model:
    torch.save(state, os.path.join(config.model_path, "{}.pt".format(start_time)))

writer.flush()
writer.close()
