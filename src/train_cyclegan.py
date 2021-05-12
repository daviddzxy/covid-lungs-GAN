import os
import torch
import itertools
import gc
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import CovidLungHealthyLungDataset
from generators import UnetGenerator2D, ResNetGenerator2D
from discriminators import PatchGanDiscriminator
from utils import weights_init, create_figure, Buffer, log_images
from transformations import RandomRotation, Crop, ApplyMask, Normalize
import argparse
import config
import matplotlib.pyplot as plt
from config import cyclegan_parameters as parameters

start_time = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_logs, start_time))
parser = argparse.ArgumentParser("Training script.")
parser.add_argument("-e", "--epochs", default=parameters["epochs"], type=int,
                    help="Set number of epochs.")
parser.add_argument("-b", "--batch-size", default=parameters["batch_size"], type=int,
                    help="Set batch size.")
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
parser.add_argument("--random-rotation", type=int, default=parameters["random_rotation"],
                    help="Set max degrees of random rotation.")
parser.add_argument("--crop", type=int, default=parameters["crop"], help="Set length of image crop.")
parser.add_argument("--generators-normalization", type=str, default=parameters["g_norm_layer"],
                    nargs="?", choices=["batch_norm", "instance_norm", "none"],
                    help="Set type of normalization layer in generators.")
parser.add_argument("--discriminators-normalization", type=str, default=parameters["d_norm_layer"],
                    nargs="?", choices=["batch_norm", "instance_norm", "none"],
                    help="Set type of normalization layer in discriminators.")
parser.add_argument("--resnet-resnet-depth", type=int, default=parameters["resnet_resnet_depth"],
                    help="Set length of resnet path.")
parser.add_argument("--resnet-scale-depth", type=int, default=parameters["resnet_scale_depth"],
                    help="Set length of image crop.")
parser.add_argument("--generators-learning-rate-decay", type=float, default=parameters["generator_learning_decay"],
                    nargs=2, help="Set learning rate decay of generator. First argument is value of learning rate,"
                                  " second argument determines period of learning rate deacy.")
parser.add_argument("--discriminators-learning-rate-decay", type=float,
                    default=parameters["discriminator_learning_decay"],
                    nargs=2, help="Set learning rate decay of discriminator. First argument is value of learning rate,"
                                  " second argument determines period of learning rate deacy.")
args = parser.parse_args()

writer.add_text("Parameters", text_string=str(args))

os.sys.path.append(config.project_root)

rotation = None
if args.random_rotation != 0:
    rotation = RandomRotation(args.random_rotation)

crop = None
if args.crop != 0:
    crop = Crop([args.crop, args.crop])

normalize = Normalize(config.cyclegan_parameters["min"], config.cgan_parameters["max"])
mask = ApplyMask(config.mask_values["non_lung_tissue"])

dataset = CovidLungHealthyLungDataset(images_A=config.cyclegan_data_train["A"],
                                      images_B=config.cyclegan_data_train["B"],
                                      mask=mask,
                                      rotation=rotation,
                                      crop=crop,
                                      normalize=normalize
                                      )
valid_dataset = CovidLungHealthyLungDataset(images_A=config.cyclegan_data_test["A"],
                                            images_B=config.cyclegan_data_test["B"],
                                            mask=mask,
                                            normalize=normalize)

dataloader = DataLoader(dataset, shuffle=True, num_workers=1, batch_size=args.batch_size, drop_last=True)

valid_dataloader = DataLoader(valid_dataset, shuffle=True, num_workers=1, batch_size=args.batch_size, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.generator == "Unet":
    netG_A2B = UnetGenerator2D(depth=args.depth_generators,
                               filters=args.filters_generators,
                               norm=args.generators_normalization).to(device).apply(weights_init)
    netG_B2A = UnetGenerator2D(depth=args.depth_generators,
                               filters=args.filters_generators,
                               norm=args.generators_normalization).to(device).apply(weights_init)
elif args.generator == "Resnet":
    netG_A2B = ResNetGenerator2D(resnet_depth=args.depth_generators,
                                 filters=args.filters_generators,
                                 norm=args.generators_normalization,
                                 scale_depth=args.scale_depth).to(device).apply(weights_init)
    netG_B2A = ResNetGenerator2D(resnet_depth=args.depth_generators,
                                 norm=args.generators_normalization,
                                 filters=args.filters_generators,
                                 scale_depth=args.scale_depth).to(device).apply(weights_init)

netD_A = PatchGanDiscriminator(filters=args.filters_discriminators,
                               depth=args.depth_discriminators,
                               norm=args.discriminators_normalization).to(device).apply(weights_init)

netD_B = PatchGanDiscriminator(filters=args.filters_discriminators,
                               depth=args.depth_discriminators,
                               norm=args.discriminators_normalization).to(device).apply(weights_init)

optimizer_G = torch.optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=args.learning_rate_generators,
    betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(
    netD_A.parameters(), lr=args.learning_rate_discriminator_a, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(
    netD_B.parameters(), lr=args.learning_rate_discriminator_b, betas=(0.5, 0.999))

scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G,
                                              gamma=args.generators_learning_rate_decay[0],
                                              step_size=args.generators_learning_rate_decay[1])
scheduler_D_A = torch.optim.lr_scheduler.StepLR(optimizer_D_A,
                                              gamma=args.discriminators_learning_rate_decay[0],
                                              step_size=args.discriminators_learning_rate_decay[1])
scheduler_D_B = torch.optim.lr_scheduler.StepLR(optimizer_D_B,
                                              gamma=args.discriminators_learning_rate_decay[0],
                                              step_size=args.discriminators_learning_rate_decay[1])

epoch_start = 0

cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)

total_batch_counter = 0
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
        writer.add_scalar("Train loss/Generator Error", errG, total_batch_counter)
        writer.add_scalar("Train loss/DiscriminatorA Error", errD_A, total_batch_counter)
        writer.add_scalar("Train loss/DiscriminatorB Error", errD_B, total_batch_counter)
        total_batch_counter += 1

    scheduler_G.step()
    scheduler_D_A.step()
    scheduler_D_B.step()
    with torch.no_grad():
        real_A = real_A.cpu().numpy()
        real_B = real_B.cpu().numpy()
        fake_image_A = fake_image_A.cpu().numpy()
        fake_image_B = fake_image_B.cpu().numpy()
        recovered_image_A = recovered_image_A.cpu().numpy()
        recovered_image_B = recovered_image_B.cpu().numpy()
        f = create_figure([real_A[0, 0, :, :],
                           fake_image_B[0, 0, :, :],
                           recovered_image_A[0, 0, :, :]],
                          figsize=(12, 4)
                          )
        writer.add_figure("Image outputs/A to B to A", f, epoch)

        f = create_figure([real_B[0, 0, :, :],
                           fake_image_A[0, 0, :, :],
                           recovered_image_B[0, 0, :, :]],
                          figsize=(12, 4)
                          )
        writer.add_figure("Image outputs/B to A to B", f, epoch)

        log_images([real_A, fake_image_B, recovered_image_A],
                   path=config.image_logs,
                   run_id=start_time,
                   step=epoch,
                   context="train_ABA",
                   figsize=(12, 4))

        log_images([real_B, fake_image_A, recovered_image_B],
                   path=config.image_logs,
                   run_id=start_time,
                   step=epoch,
                   context="train_BAB",
                   figsize=(12, 4))

        data = next(iter(valid_dataloader))
        real_A, real_B = data
        real_A, real_B = real_A.float().to(device), real_B.float().to(device)
        netG_B2A.eval()
        netG_A2B.eval()

        fake_image_A = netG_B2A(real_B)
        fake_image_B = netG_A2B(real_A)

        recovered_image_A = netG_B2A(fake_image_B)
        recovered_image_B = netG_A2B(fake_image_A)

        netG_B2A.train()
        netG_A2B.train()
        real_A = real_A.cpu().numpy()
        real_B = real_B.cpu().numpy()
        fake_image_A = fake_image_A.cpu().numpy()
        fake_image_B = fake_image_B.cpu().numpy()
        recovered_image_A = recovered_image_A.cpu().numpy()
        recovered_image_B = recovered_image_B.cpu().numpy()
        log_images([real_A, fake_image_B, recovered_image_A],
                   path=config.image_logs,
                   run_id=start_time,
                   step=epoch,
                   context="valid_ABA",
                   figsize=(12, 4))

        log_images([real_B, fake_image_A, recovered_image_B],
                   path=config.image_logs,
                   run_id=start_time,
                   step=epoch,
                   context="valid_BAB",
                   figsize=(12, 4))
        plt.close("all")
        plt.clf()
        plt.cla()
        gc.collect()

writer.flush()
writer.close()
