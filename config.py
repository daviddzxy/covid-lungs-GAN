import os

project_root = os.path.dirname(os.path.realpath(__file__))
preprocessing_logs = os.path.join(project_root, "logs/preprocessing_logs/")
training_logs = os.path.join(project_root, "logs/training_logs/")

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

cgan_covid_data = "/media/david/DATA/Covid-Data/cgan_data/data"
cgan_covid_data_train = "/media/david/DATA/Covid-Data/cgan_data/train"
cgan_covid_data_test = "/media/david/DATA/Covid-Data/cgan_data/test"

dataset_metadata = "/media/david/DATA/Covid-Data/Cycle_gan_data/Training_data/dataset_metadata.pkl"
model_path = "/media/david/DATA/Covid-lungs-models"

padding_shape = [512, 512, 64]
min_covid_pixels = 100
mask_values = {
        "covid_tissue": 3,
        "non_lung_tissue": 0
}


#  training parameters
epochs = 500
batch_size = 1
generators = "Unet"
learning_rate_generators = 0.0002
learning_rate_discriminator_a = 0.00002
learning_rate_discriminator_b = 0.00002
filters_generators = 16
depth_generators = 5
filters_discriminators = 1
depth_discriminators = 4
gpu = True
save_model = [True, 50]
learning_rate_decay = [0.95, 10]
random_rotation = 8
crop = 256
identity_weight = 5.0
cycle_weight = 10.0
buffer_length = 20
