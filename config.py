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
        "A": "/media/david/DATA/Covid-Data/A",
        "B": "/media/david/DATA/Covid-Data/B"
}

padding_shape = [512, 512, 64]
