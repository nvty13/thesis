# FEATURE EXTRACTION WITH TRIPLET LOSS TO CLASSIFY DISEASE ON LEAF DATA

This repo contains the source code for the thesis

Explaination: 
- configuration.py: config the some parameters of model
- train_embedding.py: train the dataset to create the embedding vectors (feature vectors), this file imports the triplet loss function in the triplet_loss_new.py file.
- train_classifier_mlp.py: classify the feature vectors using MLP
- train_classifier_others.py: classify the feature vectors using RandomForest, Support Vector Machine
- triplet_loss_new.py: triplet loss function
