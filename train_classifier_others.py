from configuration import settings
import requests
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
)
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold,  GridSearchCV
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from keras.utils import np_utils
import glob
import json
import random
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from datetime import datetime, date
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix, ConfusionMatrixDisplay
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm

time = datetime.now()


# Import from local


class ModelClassification:
    def __init__(self):
        # self.epochs = settings.EPOCHS
        self.epochs = 1000
        self.batch_size = 512  # 512
        # self.batch_size = settings.BATCH_SIZE
        self.input_shape = settings.EMBEDDING_SIZE
        self.data_name = settings.DATA_DIR
        self.step = settings.STEP
        self.data_dir = "../../embedding/" + settings.DATA_DIR
        # self.number_class = len(os.listdir(self.data_dir))
        self.number_class = 38
        self.time = datetime.now()
        self.pair = settings.PAIR
        self.embedding_size = settings.EMBEDDING_SIZE
        self.epochs_emb = settings.EPOCHS

    def Model(self):
        model = Sequential()
        model.add(
            Dense(2048, input_dim=self.input_shape,
                  kernel_initializer="glorot_uniform")
        )
        model.add(Activation("relu"))
        model.add(Dropout(0.25))  # 0.25
        model.add(Dense(self.number_class, kernel_initializer="glorot_uniform"))
        model.add(Activation("softmax"))
        return model

    def save_model_serving(self, model):
        input_names = ["emb"]
        print(model.inputs)
        name_to_input = {
            name: t_input for name, t_input in zip(input_names, model.inputs)
        }
        MODEL_EXPORT_DIR = "face"
        MODEL_VERSION = 1
        MODEL_EXPORT_PATH = os.path.join(MODEL_EXPORT_DIR, str(MODEL_VERSION))
        print("Model dir: ", MODEL_EXPORT_PATH)
        # Save the model to the MODEL_EXPORT_PATH
        # Note using 'name_to_input' mapping, the names defined here will also be used for querying the service later
        tf.saved_model.simple_save(
            tf.keras.backend.get_session(),
            MODEL_EXPORT_PATH,
            inputs=name_to_input,
            outputs={t.name: t for t in model.outputs},
        )

    def performance_matrix(self, true, pred):
        precision = metrics.precision_score(true, pred, average="macro")
        recall = metrics.recall_score(true, pred, average="macro")
        accuracy = metrics.accuracy_score(true, pred)
        f1_score = metrics.f1_score(true, pred, average="macro")
        f1_score_micro = metrics.f1_score(true, pred, average="micro")
        f1_score_weighted = metrics.f1_score(true, pred, average="weighted")
        f = open("Evaluation.txt", "a")
        print("-"*40, file=f)
        print("[{}]EP: {}, \tEMB: {}, \tPAIR: {}, \tDATA: {}".format(datetime.now(
        ),  self.epochs_emb, self.embedding_size, self.pair, self.data_name), file=f)
        print(
            "Precision: {:04.2f} Recall: {:04.2f}, Accuracy: {:04.2f}, f1_score: {:04.2f}".format(
                precision * 100, recall * 100, accuracy * 100, f1_score * 100
            ),
            file=f
        )
        f.close()
        print(
            "Precision: {:04.4f} Recall: {:04.4f}, Accuracy: {:04.4f}, f1_score: {:04.4f}".format(
                precision * 100, recall * 100, accuracy * 100, f1_score * 100
            )
        )
        # print(f"f1_macro: {f1_score}; f1_micro{f1_score_micro}; f1_weighted{f1_score_weighted};")

    def train(self):
        model = self.Model()
        model.summary()
        labels = []
        img_embs = []
        dist = {}
        leaf_images = glob.glob(self.data_dir + "/*/*.*")
        print("[INFO] Loading images...")
        index = 0
        for emb in leaf_images:
            img_embs.append(emb)
            label = emb.split("/")[-2]
            # label = label.split("_")[0]
            if label not in dist:
                dist[label] = index
                index += 1
            labels.append(dist[label])
        save_file_dist = open("data.json", "w")
        json.dump(dist, save_file_dist)
        save_file_dist.close()
        print("[INFO] Processing the image...")

        train_img = []
        for url_img in leaf_images:
            img = np.load(url_img)
            train_img.append(img)
        train_img = np.array(train_img)
        X = np.array(train_img)
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)
        # X_train, X_val, y_train, y_val = train_test_split(
        # X_train, y_train, test_size=0.25, random_state=1)
        X_pred = X_test
        y_test_original = y_test

        # y_train = np_utils.to_categorical(y_train, self.number_class)
        # y_test = np_utils.to_categorical(y_test, self.number_class)
        # # y_val = np_utils.to_categorical(y_val, self.number_class)

        X_train = np.reshape(X_train, (len(X_train), self.input_shape))
        # X_val = np.reshape(X_val, (len(X_val), 128))
        X_test = np.reshape(X_test, (len(X_test), self.input_shape))

        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")
        # X_val = X_val.astype("float32")
        # print(X_train, y_train, X_train, y_test, X_val, y_val)

        # sc_X = StandardScaler()
        # X_train = sc_X.fit_transform(X_train)
        # X_test = sc_X.fit_transform(X_test)
        # X_pred = X_test
        # y_train = sc_X.fit_transform(y_train)
        # X_test = sc_X.transform(X_test)
        # y_test = sc_X.transform(y_test)
        print("X_Train: ", X_train)
        print("Y_train", y_train)

        classifier_list = []
        classifier_list.append(("KNN", KNeighborsClassifier()))
        classifier_list.append(
            ('Bagging Classifier', BaggingClassifier(n_estimators=50)))
        classifier_list.append(
            ("RF", RandomForestClassifier(n_estimators=100, warm_start=True)))
        classifier_list.append(
            ('MLP Classifier', MLPClassifier(max_iter=500, early_stopping=True)))
        classifier_list.append(("SVM_rbf", svm.SVC(
            kernel='rbf', random_state=None, gamma='scale', probability=True)))
        classifier_list.append(("SVM_linear", svm.SVC(
            kernel='linear', random_state=None, gamma='scale', probability=True)))
        classifier_list.append(
            ("Logistic Regression", LogisticRegression(solver='liblinear', max_iter=500)))

        # Train classifiers:
        for clf_name, classifer in classifier_list:
            print("="*40)
            print("[INFO] Training classify with", clf_name)
            start = datetime.now()
            classifer.fit(X_train, y_train)
            stop = datetime.now()
            training_time = stop - start
            y_pred = classifer.predict(X_pred)
            self.performance_matrix(y_test_original, y_pred)
            print(f"Training time: {training_time}", "for", clf_name)

        param_grid = [
            {'C': [1, 10, 100], 'kernel': ['linear']},
            {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]

        # clf = GridSearchCV(clf, param_grid)
        # y_pred = np_utils.to_categorical(y_pred, self.number_class)
        print("y_test", y_test_original)
        print("y_pred", y_pred)

        # pd.DataFrame({"true_label": y_test_original, "predicted_label": y_pred}).to_csv(
        #     "%s_true_pred.csv" % (self.data_name), index=False
        # )

        # Make the confusion matrix
        cm = confusion_matrix(y_test_original, y_pred)
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_display = ConfusionMatrixDisplay(cm).plot()
        plt.show()
        # model.save("model_%s.h5" % (self.data_name))


if __name__ == "__main__":
    model_classification = ModelClassification()
    model_classification.train()
