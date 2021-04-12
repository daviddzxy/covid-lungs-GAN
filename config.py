import os

project_root = os.path.dirname(os.path.realpath(__file__))
preprocessing_logs = os.path.join(project_root, "logs/preprocessing_logs/")
tensorboard_logs = os.path.join(project_root, "logs/tensorboard_logs/")
image_logs = os.path.join(project_root, "logs/image_logs")

data = {
    "CT0": "/media/david/DATA/Covid-Data/COVID19_1110/studies/CT-0",
    "CT1": "/media/david/DATA/Covid-Data/COVID19_1110/studies/CT-1",
    "CT2": "/media/david/DATA/Covid-Data/COVID19_1110/studies/CT-2",
    "CT3": "/media/david/DATA/Covid-Data/COVID19_1110/studies/CT-3",
    "CT4": "/media/david/DATA/Covid-Data/COVID19_1110/studies/CT-4"
}

masks = "/media/david/DATA/Covid-Data/COVID19_1110/masks"

cyclegan_data = {
    "CT0": "/media/david/DATA/Covid-Data/cyclegan_data/data/CT0",
    "CT1": "/media/david/DATA/Covid-Data/cyclegan_data/data/CT1",
    "CT2": "/media/david/DATA/Covid-Data/cyclegan_data/data/CT2",
    "CT3": "/media/david/DATA/Covid-Data/cyclegan_data/data/CT3",
    "CT4": "/media/david/DATA/Covid-Data/cyclegan_data/data/CT4"
}

cyclegan_data_train = {
    "A": "/media/david/DATA/Covid-Data/cyclegan_data/train/A",
    "B": "/media/david/DATA/Covid-Data/cyclegan_data/train/B"
}
cyclegan_data_test = {
    "A": "/media/david/DATA/Covid-Data/cyclegan_data/test/A",
    "B": "/media/david/DATA/Covid-Data/cyclegan_data/test/B"
}

cgan_data = "/media/david/DATA/Covid-Data/cgan_data/data"
cgan_data_train = "/media/david/DATA/Covid-Data/cgan_data/train"
cgan_data_test = "/media/david/DATA/Covid-Data/cgan_data/test"

model_path = "/media/david/DATA/Covid-lungs-models"

padding_shape = [512, 512, 64]
min_covid_pixels = 100

mask_values = {
    "covid_tissue": 3,
    "non_lung_tissue": 0
}

#  training parameters
cyclegan_parameters = {
    "min": -1000,
    "max": 100,
    "epochs": 500,
    "batch_size": 1,
    "generators": "Unet",
    "learning_rate_generators": 0.0002,
    "learning_rate_discriminator_a": 0.00002,
    "learning_rate_discriminator_b": 0.00002,
    "filters_generators": 16,
    "depth_generators": 5,
    "filters_discriminators": 20,
    "depth_discriminators": 2,
    "generator_learning_decay": [1.0, 1],
    "discriminator_learning_decay": [1.0, 1],
    "random_rotation": 8,
    "crop": 256,
    "identity_weight": 5.0,
    "cycle_weight": 10.0,
    "buffer_length": 20,
    "mask_covid": 1,
    "resnet_scale_depth": 1,
    "resnet_resnet_depth": 5,
    "g_norm_layer": "batch_norm",
    "d_norm_layer": "none"
}

cyclegan_parameters_covidlungmask = {
    "min": -1000,
    "max": 100,
    "epochs": 500,
    "batch_size": 1,
    "generators": "Unet",
    "learning_rate_generators": 0.0002,
    "learning_rate_discriminator_a": 0.00002,
    "learning_rate_discriminator_b": 0.00002,
    "filters_generators": 16,
    "depth_generators": 5,
    "filters_discriminators": 20,
    "depth_discriminators": 2,
    "generator_learning_decay": [1.0, 1],
    "discriminator_learning_decay": [1.0, 1],
    "rotation": 8,
    "crop": 256,
    "identity_weight": 5.0,
    "cycle_weight": 10.0,
    "buffer_length": 20,
    "mask_covid": 1,
    "resnet_scale_depth": 1,
    "resnet_resnet_depth": 5,
    "g_norm_layer": "batch_norm",
    "d_norm_layer": "none"
}

cgan_parameters = {
    "min": -1000,
    "max": 100,
    "epochs": 500,
    "batch_size": 4,
    "generator": "Unet",
    "learning_rate_generator": 0.0002,
    "learning_rate_discriminator": 0.00002,
    "filters_generator": 16,
    "depth_generator": 5,
    "filters_discriminator": 20,
    "depth_discriminator": 2,
    "rotation": 8,
    "crop": 256,
    "mask_covid": 2,
    "resnet_scale_depth": 1,
    "resnet_resnet_depth": 5,
    "generator_learning_decay": [1.0, 1],
    "discriminator_learning_decay": [1.0, 1],
    "g_norm_layer": "batch_norm",
    "d_norm_layer": "none",
    "boundary_transform": False,
    "iterations": 6
}
