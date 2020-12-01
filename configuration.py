"""
create by: Huynh Long, NGUYEN TY
create date: 28-05-2020
"""
import logging
import six
from environs import Env

env = Env()
env.read_env()  # read .env file, if it exists

default_settings = {

    # """Data name list:
    #     "color": Full dataset
    #     "tomato": 10 classes of tomato
    #     "11": 11 classes of different leaf
    #     "MNIST": MNIST dataset
    #     "Plant_Pathtology": Plant_Pathtology dataset with 21 classes
    #     "4": 4 classes
    # """

    'DATA_DIR': env('DATA_DIR', 'color'),
    'EPOCHS': env.int('EPOCHS', 100),
    'EMBEDDING_SIZE': env.int('EMBEDDING_SIZE', 50),
    'BATCH_SIZE': env.int('BATCH_SIZE', 32),
    'INPUT_SHAPE': env.int('INPUT_SHAPE', 50),
    'STEP': env.int('STEP', 20),
    'MODEL_VERSION': env.int('MODEL_VERSION', 1),
    'MODEL_EXPORT_DIR': env('MODEL_EXPORT_DIR', "data/face"),
    'JSON_PREDICT': env('JSON_PREDICT', 'data/data.json'),
    'GPU_MEMORY_LIMIT': env.float('GPU_MEMORY_LIMIT', 0.7),
    'MODEL_SAVE': env('MODEL_SAVE', 'data/models/'),
    'PAIR' : env.int('PAIR', 10)
}


class Settings():

    def __init__(self, default_settings):
        self.__load_default_settings(default_settings)

    def __load_default_settings(self, default_settings):
        for setting_name, setting_value in six.iteritems(default_settings):
            setattr(self, setting_name, setting_value)

    def __getattribute__(self, attr):
        return super(Settings, self).__getattribute__(attr)


settings = Settings(default_settings)
