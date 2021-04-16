import torch
import os
import gc
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import CoivdLungMaskLungDataset
from generators import UnetGenerator2D, ResNetGenerator2D
from discriminators import PatchGanDiscriminator
from utils import weights_init, create_figure, log_images, log_heatmap, log_data, scale, mae
from transformations import Rotation, Crop, ApplyMask, Normalize, Boundary
import config
import argparse
import matplotlib.pyplot as plt
from config import cgan_parameters as parameters

start_time = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_logs, start_time))
parser = argparse.ArgumentParser("Training script.")
parser.add_argument("-e", "--epochs", default=parameters["epochs"], type=int,
                    help="Set number of epochs.")
parser.add_argument("-b", "--batch-size", default=parameters["batch_size"], type=int,
                    help="Set batch size.")
parser.add_argument("--generator", default=parameters["generator"], nargs="?", choices=["Unet", "Resnet"],
                    help="Use graphics card during training.")
parser.add_argument("--learning-rate-generator", default=parameters["learning_rate_generator"], type=float,
                    help="Set learning rate of Generator.")
parser.add_argument("--learning-rate-discriminator", default=parameters["learning_rate_discriminator"], type=float,
                    help="Set learning rate of Discriminator.")
parser.add_argument("--filters-generator", default=parameters["filters_generator"], type=int,
                    help="Set multiplier of convolutional filters in generator.")
parser.add_argument("--depth-generator", default=parameters["depth_generator"], type=int,
                    help="Set depth of generator architecture.")
parser.add_argument("--filters-discriminator", default=parameters["filters_discriminator"], type=int,
                    help="Set multiplier of convolutional filters in discriminator.")
parser.add_argument("--depth-discriminator", default=parameters["depth_discriminator"], type=int,
                    help="Set number of convolutional layers in discriminator.")
parser.add_argument("--load-model", default="", nargs="?",
                    help="Load saved model from model_path directory. Enter filename as argument.")
parser.add_argument("--rotation", type=int, default=parameters["rotation"],
                    help="Set max degrees of random rotation.")
parser.add_argument("--mask-covid", type=int, default=parameters["mask_covid"],
                    help="Value of mask covering damaged tissue.")
parser.add_argument("--crop", type=int, default=parameters["crop"], help="Set length of image crop.")
parser.add_argument("--resnet-resnet-depth", type=int, default=parameters["resnet_resnet_depth"],
                    help="Set length of resnet path.")
parser.add_argument("--resnet-scale-depth", type=int, default=parameters["resnet_scale_depth"],
                    help="Set length of image crop.")
parser.add_argument("--generator-learning-rate-decay", type=float, default=parameters["generator_learning_decay"], nargs=2,
                    help="Set learning rate decay of generator. First argument is value of learning rate,"
                         " second argument determines period of learning rate deacy.")
parser.add_argument("--discriminator-learning-rate-decay", type=float, default=parameters["discriminator_learning_decay"],
                    nargs=2, help="Set learning rate decay of discriminator. First argument is value of learning rate,"
                                  " second argument determines period of learning rate deacy.")
parser.add_argument("--generator-normalization", type=str, default=parameters["g_norm_layer"],
                    nargs="?", choices=["batch_norm", "instance_norm", "none"],
                    help="Set type of normalization layer in generator.")
parser.add_argument("--discriminator-normalization", type=str, default=parameters["d_norm_layer"],
                    nargs="?", choices=["batch_norm", "instance_norm", "none"],
                    help="Set type of normalization layer in discriminator.")
parser.add_argument("--boundary-transform", default=parameters["boundary_transform"], action='store_true',
                    help="Turn on boundary transforms, that transforms covid mask.")
args = parser.parse_args()

writer.add_text("Parameters", text_string=str(args))

rotation = None
max_rotation = None
if args.rotation != 0:
    rotation = Rotation()
    max_rotation = args.rotation

crop = None
if args.crop != 0:
    crop = Crop([args.crop, args.crop])

boundary = None
if args.boundary_transform:
    boundary = Boundary(value_to_erode=config.mask_values["covid_tissue"], iterations=parameters["iterations"])



normalize = Normalize(config.cgan_parameters["min"], config.cgan_parameters["max"])
mask_lungs = ApplyMask(config.mask_values["non_lung_tissue"])
mask_covid = ApplyMask(config.mask_values["covid_tissue"], args.mask_covid)

dataset = CoivdLungMaskLungDataset(images=config.cgan_data_train,
                                   mask_covid=mask_covid,
                                   mask_lungs=mask_lungs,
                                   max_rotation=max_rotation,
                                   rotation=rotation,
                                   crop=crop,
                                   normalize=normalize,
                                   boundary=boundary)

valid_dataset = CoivdLungMaskLungDataset(images=config.cgan_data_test, mask_covid=mask_covid, mask_lungs=mask_lungs,
                                         crop=crop, normalize=normalize)

dataloader = DataLoader(dataset, shuffle=True, num_workers=1, batch_size=args.batch_size, drop_last=True)

valid_dataloader = DataLoader(valid_dataset, shuffle=True, num_workers=1, batch_size=args.batch_size, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = None
if args.generator == "Unet":
    generator = UnetGenerator2D(depth=args.depth_generator,
                                filters=args.filters_generator,
                                norm=args.generator_normalization).to(device).apply(weights_init)
elif args.generator == "Resnet":
    generator = ResNetGenerator2D(resnet_depth=args.resnet_resnet_depth,
                                  scale_depth=args.resnet_scale_depth,
                                  filters=args.filters_generator,
                                  norm=args.generator_normalization).to(device).apply(weights_init)

discriminator = PatchGanDiscriminator(filters=args.filters_discriminator,
                                      depth=args.depth_discriminator,
                                      norm=args.discriminator_normalization,
                                      in_channels=2).to(device).apply(weights_init)

optimizer_G = torch.optim.Adam(generator.parameters(),
                               lr=args.learning_rate_generator,
                               betas=(0.5, 0.999))

optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=args.learning_rate_discriminator,
                               betas=(0.5, 0.999))

scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G,
                                              gamma=args.generator_learning_rate_decay[0],
                                              step_size=int(args.generator_learning_rate_decay[1]))

scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_G,
                                              gamma=args.discriminator_learning_rate_decay[0],
                                              step_size=int(args.discriminator_learning_rate_decay[1]))

l1 = torch.nn.L1Loss().to(device)
l2 = torch.nn.MSELoss().to(device)
total_batch_counter = 0
for epoch in range(0, args.epochs):
    print("Current epoch {}.".format(epoch))
    for i, data in enumerate(dataloader):
        image, masked_image, _ = data
        image, masked_image = image.float().to(device), masked_image.float().to(device)
        fake_image = generator(masked_image)
        optimizer_D.zero_grad()

        fake_input_D = torch.cat([fake_image, masked_image], 1)
        pred_fake = discriminator(fake_input_D.detach())

        real_label = torch.ones(pred_fake.shape).to(device)
        fake_label = torch.zeros(pred_fake.shape).to(device)

        fake_loss_D = l2(pred_fake, fake_label)

        real_input_D = torch.cat([image, masked_image], 1)
        pred_real = discriminator(real_input_D)
        real_loss_D = l2(pred_real, real_label)

        loss_D = (fake_loss_D + real_loss_D) * 0.5
        loss_D.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        fake_input_D = torch.cat([fake_image, masked_image], 1)
        pred_fake = discriminator(fake_input_D)
        adversarial_loss = l2(pred_fake, real_label)
        identity_loss = l1(fake_image, image)

        loss_G = adversarial_loss + identity_loss
        loss_G.backward()
        optimizer_G.step()

        writer.add_scalar("Train loss cgan/Generator Error", loss_G, total_batch_counter)
        writer.add_scalar("Train loss cgan/Discriminator Error", loss_D, total_batch_counter)
        writer.add_scalar("Train loss cgan/G Identity_loss", identity_loss, total_batch_counter)
        writer.add_scalar("Train loss cgan/G Adversarial_loss", adversarial_loss, total_batch_counter)
        writer.add_scalar("Train loss cgan/D fake_loss_D", fake_loss_D, total_batch_counter)
        writer.add_scalar("Train loss cgan/D real_loss_D", real_loss_D, total_batch_counter)
        total_batch_counter += 1

    scheduler_G.step()
    scheduler_D.step()
    with torch.no_grad():
        masked_image = masked_image.cpu().numpy()
        fake_image = fake_image.cpu().numpy()
        image = image.cpu().numpy()
        l1_diff = mae(scale(image,
                            config.cgan_parameters["min"],
                            config.cgan_parameters["max"],
                            mask=masked_image,
                            mask_val=config.mask_values["non_lung_tissue"]),
                      scale(fake_image,
                            config.cgan_parameters["min"],
                            config.cgan_parameters["max"],
                            mask=masked_image,
                            mask_val=config.mask_values["non_lung_tissue"]),
                      mask=masked_image,
                      mask_val=config.mask_values["non_lung_tissue"])
        writer.add_scalar("L1 diff/Train", l1_diff, epoch)

        f = create_figure([masked_image[0, 0, :, :],
                           fake_image[0, 0, :, :],
                           image[0, 0, :, :]], figsize=(12, 4))

        writer.add_figure("Image outputs/Real image, fake image, mask", f, epoch)

        log_images([masked_image, fake_image, image],
                   path=config.image_logs,
                   run_id=start_time,
                   step=epoch,
                   context="train",
                   figsize=(12, 4))

        data = next(iter(valid_dataloader))
        valid_image, valid_masked_image, valid_mask = data
        valid_image, valid_masked_image = valid_image.float().to(device), valid_masked_image.float().to(device)
        generator.eval()
        valid_fake_image = generator(valid_masked_image)
        valid_image = valid_image.float().detach().cpu().numpy()
        valid_masked_image = valid_masked_image.float().detach().cpu().numpy()
        valid_fake_image = valid_fake_image.detach().cpu().numpy()

        valid_fake_image = mask_lungs(valid_fake_image, valid_mask)

        log_data(valid_fake_image, config.image_logs, run_id=start_time, step=epoch, context="raw")

        l1_diff = mae(
            scale(valid_image,
                  config.cgan_parameters["min"],
                  config.cgan_parameters["max"],
                  mask=valid_masked_image,
                  mask_val=config.mask_values["non_lung_tissue"]),
            scale(valid_fake_image,
                  config.cgan_parameters["min"],
                  config.cgan_parameters["max"],
                  mask=valid_masked_image,
                  mask_val=config.mask_values["non_lung_tissue"]
                  ),
            mask=valid_masked_image,
            mask_val=config.mask_values["non_lung_tissue"])

        writer.add_scalar("L1 diff/Valid", l1_diff, epoch)
        generator.train()
        log_images([valid_masked_image, valid_fake_image, valid_image],
                   path=config.image_logs,
                   run_id=start_time,
                   step=epoch,
                   context="valid",
                   figsize=(12, 4))

        log_heatmap(scale(valid_image, config.cgan_parameters["min"], config.cgan_parameters["max"],
                          mask=valid_masked_image, mask_val=config.mask_values["non_lung_tissue"]),
                    scale(valid_fake_image, config.cgan_parameters["min"], config.cgan_parameters["max"],
                          mask=valid_masked_image, mask_val=config.mask_values["non_lung_tissue"]),
                    path=config.image_logs,
                    run_id=start_time,
                    step=epoch,
                    context="heat",
                    figsize=(14, 5))

    plt.close("all")
    plt.clf()
    plt.cla()
    gc.collect()

writer.flush()
writer.close()
