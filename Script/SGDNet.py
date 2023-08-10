
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

import os
import numpy as np
from tensorflow import keras
from pyrsgis import raster
from pyrsgis.convert import changeDimension, array_to_table
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from tensorflow.keras.layers import Dense, Activation,Dropout 
import matplotlib.pyplot as plt
import pandas as pd

# Change the directory
os.chdir(r"D:\Carried_work\Sikkim_data")

# Assign file names
debris_image = 'image1.tif'
debris_label = 'label1.tif'
prediction = 'test.tif'

# Enter number of images to train from 1,2,3......n
n = 194

# Loop through all images and labels to form training data
for i in range(1,n):
	if(i==1):
		# Read the rasters as array
		ds1, featuresdebris = raster.read(debris_image, bands='all')
		ds2, debris_label = raster.read(debris_label)
		ds3, prediction = raster.read(prediction, bands='all')
		# Clean the labelled data to replace NoData values by zero
		debris_label = (debris_label == 1).astype(int)
		# Reshape the array 
		featuresdebris = array_to_table(featuresdebris)
		debris_label = array_to_table(debris_label)
		prediction = array_to_table(prediction)
		nBands = featuresdebris.shape[1]
	else:
		debris_image2 = 'image'+str(i)+'.tif'
		debris_label2 = 'label'+str(i)+'.tif'
		ds12, featuresdebris2 = raster.read(debris_image2, bands='all')
		ds22, debris_label2 = raster.read(debris_label2)
		debris_label2 = (debris_label2 == 1).astype(int)
		featuresdebris2 = array_to_table(featuresdebris2)
		debris_label2 = array_to_table(debris_label2)
        #stack arrays
		featuresdebris = np.vstack((featuresdebris,featuresdebris2))
		debris_label = np.hstack((debris_label,debris_label2))


print("Debris input data: ", featuresdebris.shape)
print("Debris label data: ", debris_label.shape)
print("prediction data: ", prediction.shape)

# Split testing and training datasets
xTrain, xTest, yTrain, yTest = train_test_split(featuresdebris, debris_label, test_size=0.3, random_state=100)

print(xTrain.shape)
print(yTrain.shape)

print(xTest.shape)
print(yTest.shape)

#standard scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(xTest)
xTrain, xTest = scaler.transform(xTrain), scaler.transform(xTest)
prediction = scaler.transform(prediction)

# Normalise the data
# xTrain = xTrain / 255.0
# xTest = xTest / 255.0
# prediction = prediction / 255.0

# Reshape the data
xTrain = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1]))
xTest = xTest.reshape((xTest.shape[0], 1, xTest.shape[1]))
prediction = prediction.reshape((prediction.shape[0], 1, prediction.shape[1]))

# Print the shape of reshaped data
print(xTrain.shape, xTest.shape, prediction.shape)

# Define the parameters of the model


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, nBands)),
    #keras.layers.Dense(2048, activation='tanh'),
    keras.layers.Dense(1024, activation='tanh'),
    keras.layers.Dense(512, activation='tanh'),
    #keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation='tanh'),
    #keras.layers.Dropout(0.25),
    keras.layers.Dense(128, activation='tanh'),
    #keras.layers.Dropout(0.25),
    keras.layers.Dense(64, activation='tanh'),
    #keras.layers.Dropout(0.25),
    keras.layers.Dense(32, activation='tanh'),
    #keras.layers.Dropout(0.25),
    keras.layers.Dense(2, activation='softmax')])

model.summary()


opt= keras.optimizers.Adam(learning_rate=0.0001,decay=1e-6)
# Define the accuracy metrics and parameters
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)
# Checck point to save model after every epoch
#checkpoint = keras.callbacks.ModelCheckpoint('model{epoch:08d}.h5', period=1) 
# Run the model
history = model.fit(xTrain, yTrain, epochs=400, validation_data= (xTest, yTest), verbose=1, callbacks=(early_stop), batch_size = 2000, use_multiprocessing=True, workers=8)

#pd.DataFrame(model.history.history).plot(figsize=(10,10), dpi=300)
#plt.show()
# model_loss.plot()
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(10,10), dpi=300)
plt.text(100, 0.224, 'A', fontsize = 14, bbox = dict(facecolor = 'none', alpha = 0.9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.title('Model loss')
plt.ylabel('loss (%)')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Test loss'], loc='upper right')

plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(10,10), dpi=300)
plt.text(20, 0.914, 'B', fontsize = 14, bbox = dict(facecolor = 'none', alpha = 0.9))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#plt.title('Model Accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('Epoch')
plt.legend(['Train accuracy', 'Test accuracy'], loc='lower right')

# assuming you stored your model.fit results in a 'history' variable:
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# or save to csv: 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# print(history.history.keys())

# import pandas as pd

# model_loss = pd.DataFrame(model.history.history)
# model_loss.plot()

# Predict for test data 
yTestPredicted = model.predict(xTest)
yTestPredicted = yTestPredicted[:,1]
 
# Calculate and display the error metrics
yTestPredicted = (yTestPredicted>0.5).astype(int)
cMatrix = confusion_matrix(yTest, yTestPredicted)
pScore = precision_score(yTest, yTestPredicted)
rScore = recall_score(yTest, yTestPredicted)
#
print("Confusion matrix: for 14 nodes\n", cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))

predicted = model.predict(prediction)
predicted = predicted[:,1]

# # Predict new data and export the probability raster
prediction = np.reshape(predicted, (ds3.RasterYSize, ds3.RasterXSize))
outFile = 'lake_estimation1.tif'
raster.export(prediction, ds3, filename=outFile, dtype='float')

model.save('save_model.h5')