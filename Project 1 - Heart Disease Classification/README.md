# Project 1 - Heart Disease Classifier
Simple heart disease classifier built on Keras

## Table of contents
* [General info](https://github.com/illusionikx/AI07_training_projects/tree/main/Project%201%20-%20Heart%20Disease%20Classification#general-info-general-info)
* [Framework](https://github.com/illusionikx/AI07_training_projects/tree/main/Project%201%20-%20Heart%20Disease%20Classification#framework-framework)
* [Methodology](https://github.com/illusionikx/AI07_training_projects/tree/main/Project%201%20-%20Heart%20Disease%20Classification#methodology-methodology)
* [Results](https://github.com/illusionikx/AI07_training_projects/tree/main/Project%201%20-%20Heart%20Disease%20Classification#results-results)

## General info
This project is done to fulfil the requirement for class AI07. The aim of the project is to create a classifier to predict heart disease using data from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) with accuracy at least 90% and training loss and validation difference not more than 15%.

## Framework
This project is created using Spyder as the main IDE. The main frameworks used in this project are Pandas, Scikit-learn and TensorFlow Keras.

## Methodology
### Data
Data is imported from Kaggle in form of csv. Preprocessing is not done because the data did not have any discrepancy or missing data. Data is then split into training-test dataset with ratio of 7:3.

### Model
Model is constructed with feedforward neural network. The structure of the model is as follows.

![model](https://github.com/illusionikx/AI07_training_projects/blob/main/Project%201%20-%20Heart%20Disease%20Classification/model.png)

Model is then trained with training dataset with the whole dataset as batch size in 300 epochs with early stopping applied. Training stops at epoch 184 and obtain accuracy of 91.4% with validation accuracy of 92.9%

![accuracy](https://github.com/illusionikx/AI07_training_projects/blob/main/Project%201%20-%20Heart%20Disease%20Classification/accuracy.png)


![loss](https://github.com/illusionikx/AI07_training_projects/blob/main/Project%201%20-%20Heart%20Disease%20Classification/loss.png)

## Results
Model is evaluated using test dataset and the following are the results.

![result](https://github.com/illusionikx/AI07_training_projects/blob/main/Project%201%20-%20Heart%20Disease%20Classification/results.png)
