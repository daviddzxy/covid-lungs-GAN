import torch
import pickle
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import CganDataset
from generators import UnetGenerator2D, ResNetGenerator2D
from discriminators import PatchGanDiscriminator
from utils import weights_init, create_figure, log_images, log_heatmap, scale
from transformations import Rotation, Crop, ApplyMask, Normalize
import config
import argparse
import matplotlib.pyplot as plt
from config import cgan_parameters as parameters

start_time = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
writer = SummaryWriter(log_dir=config.tensorboard_logs + start_time)
parser = argparse.ArgumentParser("Training script.")
parser.add_argument("-e", "--epochs", default=parameters["epochs"], type=int,
                    help="Set number of epochs.")
parser.add_argument("-b", "--batch-size", default=parameters["batch_size"], type=int,
                    help="Set batch size.")
parser.add_argument("--gpu", default=parameters["gpu"], nargs="?",
                    help="Use graphics card during training.")
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
parser.add_argument("--save-model", default=parameters["save_model"], nargs="?",
                    help="Turn on model saving.")
parser.add_argument("--load-model", default="", nargs="?",
                    help="Load saved model from model_path directory. Enter filename as argument.")
parser.add_argument("--rotation", type=int, default=parameters["rotation"],
                    help="Set max degrees of random rotation.")
parser.add_argument("--crop", type=int, default=parameters["crop"], help="Set length of image crop.")
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


normalize = Normalize(config.cgan_parameters["min"], config.cgan_parameters["max"])
mask_lungs = ApplyMask(config.mask_values["non_lung_tissue"])
mask_covid = ApplyMask(config.mask_values["covid_tissue"], 1)

dataset = CganDataset(images=config.cgan_data_train,
                      mask_covid=mask_covid,
                      mask_lungs=mask_lungs,
                      max_rotation=max_rotation,
                      rotation=rotation,
                      crop=crop,
                      normalize=normalize)

valid_dataset = CganDataset(images=config.cgan_data_test, mask_covid=mask_covid, normalize=normalize)

dataloader = DataLoader(dataset, shuffle=True, num_workers=1, batch_size=args.batch_size, drop_last=True)

valid_dataloader = DataLoader(dataset, shuffle=False, num_workers=1, batch_size=args.batch_size, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

generator = None
if args.generator == "Unet":
    generator = UnetGenerator2D(depth=args.depth_generator,
                                filters=args.filters_generator).to(device).apply(weights_init)
elif args.generator == "Resnet":
    generator = ResNetGenerator2D(resnet_depth=args.depth_generator,
                                  filters=args.filters_generator).to(device).apply(weights_init)

discriminator = PatchGanDiscriminator(filters=args.filters_discriminator,
                                      depth=args.depth_discriminator,
                                      in_channels=2).to(device).apply(weights_init)

optimizer_G = torch.optim.Adam(generator.parameters(),
                               lr=args.learning_rate_generator,
                               betas=(0.5, 0.999))

optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=args.learning_rate_discriminator,
                               betas=(0.5, 0.999))

l1 = torch.nn.L1Loss().to(device)
l2 = torch.nn.MSELoss().to(device)

total_batch_counter = 0
for epoch in range(0, args.epochs):
    print("Current epoch {}.".format(epoch))
    for i, data in enumerate(dataloader):
        image, masked_image = data
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
        ake_input_D = torch.cat([fake_image, masked_image], 1)
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

    print("G Identity_loss {}".format(identity_loss))
    print("G Adversarial_loss {}".format(adversarial_loss))
    print("D fake_loss_D {}".format(fake_loss_D))
    print("D real_loss_D {}".format(real_loss_D))

    writer.add_scalar("L1 diff/Train", identity_loss, epoch)
    f = create_figure([masked_image[0, 0, :, :].detach().cpu(),
                       fake_image[0, 0, :, :].detach().cpu(),
                       image[0, 0, :, :].detach().cpu()], figsize=(12, 4))

    writer.add_figure("Image outputs/Real image, fake image, mask", f, epoch)

    log_images([masked_image, fake_image, image],
               path=config.image_logs,
               run_id=start_time,
               step=epoch,
               context="train",
               figsize=(12, 4))

    with torch.no_grad():
        data = next(iter(valid_dataloader))
        valid_image, valid_masked_image = data
        valid_image, valid_masked_image = valid_image.float().to(device), valid_masked_image.float().to(device)
        generator.eval()
        valid_fake_image = generator(valid_masked_image)
        l1_diff = l1(valid_image, valid_fake_image)
        writer.add_scalar("L1 diff/Valid", l1_diff, epoch)
        generator.train()
        log_images([masked_image, fake_image, image],
                   path=config.image_logs,
                   run_id=start_time,
                   step=epoch,
                   context="valid",
                   figsize=(12, 4))

        log_heatmap(scale(image.detach().cpu().numpy(), config.cgan_parameters["min"], config.cgan_parameters["max"],
                          mask=valid_masked_image.detach().cpu().numpy()),
                    scale(fake_image.detach().cpu().numpy(), config.cgan_parameters ["min"], config.cgan_parameters["max"],
                          mask=valid_masked_image.detach().cpu().numpy()),
                    path=config.image_logs,
                    run_id=start_time,
                    step=epoch,
                    context="heat",
                    figsize=(14, 5))
    plt.clf()


writer.flush()
writer.close()