
"""
Created on Tue Jan 19 09:09:08 2021

"""
# -*- coding: utf-8 -*-
"""
Saurabh Kaushik (The Ohio State University, Columbus Ohio USA)
Base codes are ddapted from https://towardsdatascience.com/neural-network-for-satellite-data-classification-using-tensorflow-in-python-a13bcf38f3e1 
Original research Article can be found at https://doi.org/10.3390/rs14061352 
Debris cover Delineation using Deep Neural Network
"""

import tensorflow as tf
import os
import numpy as np
from pyrsgis import raster, convert
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Change the directory
os.chdir(r"D:\Carried_work\Sikkim_data")

# Assign file names
prediction = 'test.tif'

# Read the raster as an array
ds3, prediction_data = raster.read(prediction, bands='all')
prediction_data = convert.changeDimension(prediction_data)

# Load the pre-trained model
model = tf.keras.models.load_model('save_model.h5')

# Standardize the prediction data
scaler = StandardScaler()
scaler.fit(prediction_data)
prediction_data = scaler.transform(prediction_data)

# Reshape the data
prediction_data = prediction_data.reshape((prediction_data.shape[0], 1, prediction_data.shape[1]))

# Print the shape of the reshaped data
print(prediction_data.shape)

# Predict new data and export the probability raster
predicted = model.predict(prediction_data)
predicted_prob = predicted[:, 1]
predicted_prob = np.reshape(predicted_prob, (ds3.RasterYSize, ds3.RasterXSize))

outFile = 'D:/Carried_work/Sikkim_data/prediction.tif'
raster.export(predicted_prob, ds3, filename=outFile, dtype='float')
