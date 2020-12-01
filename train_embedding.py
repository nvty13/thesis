# Necessary libraries, packages
import cv2
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time
from datetime import datetime
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Flatten, Lambda, Dense, concatenate, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization, ReLU, Activation
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import backend as K

from keras.applications.vgg16 import preprocess_input
# Import from local
from configuration import settings
from triplet_loss_new import *
# alkdjflasdjfld
from keras.datasets import mnist
import random

# GPU Configuration
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

time = datetime.now()


def keras_batch_hard_triplet_loss(labels, y_pred):
    # As omoindrot's loss functions expects the labels to have shape (batch_size,), labels are flattaned.
    # Before flattening, they have shape (batch_size,1).

    labels = K.flatten(labels)
    return batch_hard_triplet_loss(labels, y_pred, margin=0.5)


class ModelRecognize:

    def __init__(self, input_shape=512):
        self.epochs = settings.EPOCHS
        self.batch_size = settings.BATCH_SIZE
        self.embedding_size = settings.EMBEDDING_SIZE
        self.input_shape = settings.INPUT_SHAPE
        self.input_image_shape = (224, 224, 3)
        self.step = settings.STEP
        self.model_export_dir = settings.MODEL_EXPORT_DIR
        self.model_version = settings.MODEL_VERSION
        self.data_name = settings.DATA_DIR
        self.data_dir = "../../data/" + settings.DATA_DIR
        self.number_class = len(os.listdir(self.data_dir))
        self.config = ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = settings.GPU_MEMORY_LIMIT
        self.config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config=self.config)
        #self.model_save_dir = os.path.join(settings.MODEL_SAVE,settings.MODEL_VERSION)


    def base_model(self):
        #base_model = MobileNet(
        #    include_top=False, weights='imagenet', 
        #    input_shape=(224, 224, 3))   
        base_model = VGG16(
            include_top=False, weights='imagenet', 
            input_shape=(224, 224, 3))           
        for layer in base_model.layers[:5]:
            layer.trainable = False
        
        base_model.summary()
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(50))
        model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
        return model

    def preprocess(self):
        leaf_images = glob.glob(self.data_dir + "/*/*.*")
        labels = []
        images = []
        dist = {}
        index = 0
        for img in leaf_images:
            images.append(img)
            label = img.split("/")[-2]
            # label = label.split("_")[0]
            if label not in dist:
                dist[label] = index
                index += 1
            labels.append(dist[label])
        # print(labels)
        print("[INFO] Processing the image...")

        xTrain, xTest, yTrain, yTest = train_test_split(
            images, labels, test_size=0.2, random_state=0
        )
        train_img = []
        test_img = []
        for url_img in xTrain:
            img = cv2.imread(url_img, 1)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            # img = img / 255.
            # img = np.expand_dims(img, axis=0)
            train_img.append(img)
        train_img = np.array(train_img)
        for url_img in xTest:
            img = cv2.imread(url_img, 1)

            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            # img = img / 255.
            # img = np.expand_dims(img, axis=0)
            test_img.append(img)
        test_img = np.array(test_img)
        x_train = np.array(train_img)
        y_train = np.array(yTrain)
        x_test = np.array(test_img)
        y_test = np.array(yTest)

        # The data, split between train and test sets
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train /= 255.0
        x_test /= 255.0
        input_image_shape = self.input_image_shape
        x_val = x_test[:2000, :, :]
        y_val = y_test[:2000]
        return x_train, x_test, y_train, y_test
        print("[INFO] Finished preprocessing image...")

    def train(self):
        print("[INFO] Training stage...")
        K.set_image_data_format("channels_last")

        x_train, x_test, y_train, y_test = self.preprocess()
        model = self.base_model()
        model.compile(loss=keras_batch_hard_triplet_loss,
                      optimizer=tf.keras.optimizers.Adam(1e-4))
        # model.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer=tf.keras.optimizers.Adam(1e-4))
        #model.compile(loss=triplet_loss, optimizer=Adam(lr=0.0001))

        # Uses 'dummy' embeddings + dummy gt labels. Will be removed as soon as
        # loaded, to free memory

        model.summary()
        filepath = "../../checkpoints/checkpoint_%s_ep50_BS%d.hdf5" % (
            self.data_dir, self.batch_size)
        # checkpoint = ModelCheckpoint(
        #     filepath, monitor='val_loss', verbose=1, save_best_only=False, period=50)
        # callbacks_list = [checkpoint]
        x_train = np.reshape(
            x_train, (len(x_train), x_train.shape[1], x_train.shape[1], 3))

        H = model.fit(
            x_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_test, y_test),

            # callbacks=callbacks_list
        )
        print("[INFO] Plotting the loss...")
        plt.figure(figsize=(8, 8))
        plt.plot(H.history['loss'], label='training loss')
        plt.plot(H.history['val_loss'], label='validation loss')
        plt.legend()
        plt.title('Train/validation loss')
        plt.savefig("../../plotting_results/%s (%d epochs) loss.jpg" %
                    (time, self.epochs))
        # plt.show()
        # model = load_model('checkpoints/checkpoint_%s_ep50_BS%d.hdf5' % (self.data_dir, self.batch_size),
        #                   custom_objects={'keras_batch_hard_triplet_loss': keras_batch_hard_triplet_loss})
        testing_embeddings = self.base_model()
        model.save("%s_model.h5" % self.data_name)
        #self.test(x_train, y_train, testing_embeddings, model)
        # self.img2emb('embedding')

    def test(self, x_train, y_train):
        testing_embeddings = self.base_model()
        model = self.base_model()
        model.load_weights("%s_model.h5" % self.data_name)
        x_embeddings_before_train = testing_embeddings.predict(
            np.reshape(x_train, (len(x_train), 224, 224, 3)))

        # for layer_target, layer_source in zip(testing_embeddings.layers, model.layers[2].layers):
        #     weights = layer_source.get_weights()
        #     layer_target.set_weights(weights)
        #     del weights
        x_embeddings = model.predict(
            np.reshape(x_train, (len(x_train), 224, 224, 3)))

        dict_embeddings = {}
        dict_gray = {}
        test_class_labels = np.unique(np.array(y_train))

        pca = PCA(n_components=3)
        step = self.step
        decomposed_embeddings = pca.fit_transform(x_embeddings)
        decomposed_gray = pca.fit_transform(x_embeddings_before_train)
        print("[INFO] Plotting cluster...")
        fig = plt.figure(figsize=(16, 8))
        for label in test_class_labels:
            print(label)
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            decomposed_embeddings_class = decomposed_embeddings[
                y_train == label]
            decomposed_gray_class = decomposed_gray[y_train == label]

            plt.subplot(1, 2, 1)
            plt.scatter(
                decomposed_gray_class[::step, 1], decomposed_gray_class[::step, 0], color=color, label=str(label))
            plt.title('before training (embeddings)')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.scatter(decomposed_embeddings_class[::step, 1],
                        decomposed_embeddings_class[::step, 0], color=color, label=str(label))
            plt.title('after @%d epochs' % self.epochs)
            plt.legend()
        fig.savefig("../../plotting_results/%s_(%d epochs)_CLUSTER.png" %
                    (time, self.epochs))
        print("[INFO] Congratulation! Finished saving figure...")

    def get_path_img(self, input_path):
        path_images = []
        for root, subdirs, files in os.walk(input_path):
            for name in files:
                file_ext = name.split(".")[-1]
                if file_ext.lower() in ["jpg", "jpeg", "png"]:
                    filename = os.path.join(root, name)
                    path_images.append(filename)
        return path_images

    def img2emb(self, output_path):
        print("Start creating embedding...")
        f = open("feature_vectors.csv", "w")
        i = 0
        testing_embeddings = self.base_model()
        testing_embeddings.load_weights("%s_model.h5" % self.data_name)
        path_images = self.get_path_img(self.data_dir)

        for path_img in path_images:
            i += 1
            dirs_img = path_img.split("/")
            output_emb = os.path.join(output_path, dirs_img[-2])
            img = cv2.imread(path_img)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            img = np.array([img])
            img = np.expand_dims(img, -1)
            img = img.astype('float32')
            img /= 255.
            emb = testing_embeddings.predict(np.reshape(img, (1, 224, 224, 3)))
            emb = emb[0]

            print("\n", dirs_img[4], emb, file=f)

            if not os.path.isdir(output_emb):
                os.makedirs(output_emb)
            np.save(os.path.join(output_emb, dirs_img[-1] + ".npy"), emb)
            print("[{:04.2f}%] Saved to".format(i/54305*100),
                  os.path.join(output_emb, dirs_img[-1] + ".npy"))
        f.close()

        # Write the embeding to csv file
        f = open("feature_vectors.csv", "r")
        text = f.read()
        text = " ".join(text.split())
        text = text.replace(" ", ",")
        text = text.replace("[", "")
        text = text.replace("],", "\n")
        print("Finished replacing...")
        outputfile = "feature_vectors_output.csv"
        with open(outputfile, 'w') as of:
            of.write(text)
        of.close()
        f.close()
        print("Saved embedding features to", outputfile)


if __name__ == '__main__':
    start = datetime.now()

    model_recognize = ModelRecognize()
    model_recognize.train()
    x_train, x_test, y_train, y_test = model_recognize.preprocess()
    model_recognize.test(x_train, y_train)
    model_recognize.img2emb('../../embedding/'+model_recognize.data_name)

    stop = datetime.now()
    training_time = stop - start
    print(f"Total time for training triplet: {training_time}")
