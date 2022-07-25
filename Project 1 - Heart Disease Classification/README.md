# Project 1 - Heart Disease Classifier
Simple heart disease classifier built on Keras

## Table of contents
* [General info](#general-info)
* [Framework](#framework)
* [Methodology](#methodology)
* [Results](#results)

## General info {#general-info}
This project is done to fulfil the requirement for class AI07. The aim of the project is to create a classifier to predict heart disease using data from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) with accuracy at least 90% and training loss and validation difference not more than 15%.

## Framework {#framework}
This project is created using Spyder as the main IDE. The main frameworks used in this project are Pandas, Scikit-learn and TensorFlow Keras.

## Methodology {#methodology}
### Data
Data is imported from Kaggle in form of csv. Preprocessing is not done because the data did not have any discrepancy or missing data. Data is then split into training-test dataset with ratio of 7:3.

### Model
Model is constructed with feedforward neural network. The structure of the model is as follows.

![model](/model.png)

Model is then trained with training dataset with the whole dataset as batch size in 300 epochs with early stopping applied. Training stops at epoch 184 and obtain accuracy of 91.4% with validation accuracy of 92.9%

![accuracy](/accuracy.png)


![loss](/loss.png)

## Results {#results}
Model is evaluated using test dataset and the following are the results.

![result](/results.png)
