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

preprocessed_data = {
        "CT0": "/media/david/DATA/Covid-Data/Preprocessed_data/CT0",
        "CT1": "/media/david/DATA/Covid-Data/Preprocessed_data/CT1",
        "CT2": "/media/david/DATA/Covid-Data/Preprocessed_data/CT2",
        "CT3": "/media/david/DATA/Covid-Data/Preprocessed_data/CT3",
        "CT4": "/media/david/DATA/Covid-Data/Preprocessed_data/CT4"
        }

training_data = {
        "A": "/media/david/DATA/Covid-Data/Training_data/A",
        "B": "/media/david/DATA/Covid-Data/Training_data/B"
}

model_path = "/media/david/DATA/Covid-lungs-models"

padding_shape = [512, 512, 64]

#  training parameters
epochs = 500
batch_size = 1
learning_rate_generators = 0.0001
learning_rate_discriminator_a = 0.00005
learning_rate_discriminator_b = 0.00005
filters_generators = 6
depth_generators = 5
filters_discriminators = 1
depth_discriminators = 2
gpu = True
save_model = True
save_model_epoch = 10
learning_rate_decay = [0.95, 10]
random_rotation = 3
crop = 256
