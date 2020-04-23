#!/usr/bin/env python
# coding: utf-8

# ## Initial Setup/Importing Libraries

# In[4]:


get_ipython().system('pip install google-colab')


# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


tf.__version__


# In[5]:


## Code to read csv file into Colaboratory
## https://pypi.org/project/PyDrive/
## https://towardsdatascience.com/how-to-manage-files-in-google-drive-with-python-d26471d91ecd
get_ipython().system('pip install -U -q PyDrive')
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


# In[ ]:


link = "https://drive.google.com/open?id=1uKOcbFBGRZezTrN_zWlp2VzEMly5LJUs" ## training.csv file


# In[ ]:


fluff, id = link.split('=')
print (id) # Verify that you have everything after '='


# In[ ]:


dataset = drive.CreateFile({'id':id}) 
dataset.GetContentFile('training.csv')  
dataset = pd.read_csv('training.csv')


# ## EDA

# In[ ]:


dataset.head()


# Check the dimensions of the images

# In[ ]:


img_sample = dataset['Image'][0]
img_arr = img_sample.split()
np.sqrt(len(img_arr))


# Image pixels are in the form of string of values seperated by space. Below code puts them in structured array

# In[ ]:


image_data = dataset['Image']
image_pixels = []
img_arr=[]
for img in image_data:
  img_arr = img.split()
  image_pixels.append(img_arr)


# In[ ]:


for img in image_pixels[:5]:
  print(len(img))


# In[ ]:


dataset['Image'] = pd.Series(image_pixels)


# In[ ]:


dataset['Image']


# In[ ]:


image_pixels_df = pd.DataFrame(image_pixels)


# In[ ]:


image_pixels_df.head()


# In[ ]:


image_pixels_df = image_pixels_df.dropna()


# In[ ]:


image_pixels_df.shape


# In[ ]:


dataset.head()


# In[ ]:


dataset.isna()


# In[ ]:


Y = dataset.iloc[:, :-1]


# In[ ]:


Y.fillna(method = "ffill", inplace = True) ## Fill in Null values


# In[ ]:


Y.max()


# In[ ]:


#Scaling values between 0 and 1
Y = (Y-48)/48


# In[ ]:


Y.dtypes


# In[ ]:


X = np.array(image_pixels)
X = X.astype('float64')


# In[ ]:


X[0]


# In[ ]:


X.dtype


# In[ ]:


X = X.reshape(-1, 96,96,1)


# In[ ]:


Y.shape


# In[ ]:


from tensorflow.keras.layers import MaxPooling2D, Conv2D , Flatten, Dropout, BatchNormalization, Dense
from tensorflow.keras.models import Sequential


# In[ ]:


def model():
  model = Sequential()

  model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(96,96,1)))
  model.add(Dropout(0.2)) ## Dropout bottom 20%
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(BatchNormalization())

  model.add(Conv2D(32, (5, 5) ,activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())

  model.add(Conv2D(64, (5, 5) ,activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())

  model.add(Conv2D(128, (3, 3) ,activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())

  model.add(Flatten())

  model.add(Dense(512, activation="relu"))
  model.add(Dropout(0.2))

  model.add(Dense(256, activation="relu"))
  model.add(Dropout(0.1))

  model.add(Dense(128, activation="relu"))
  model.add(Dropout(0.1))

  model.add(Dense(64, activation="relu"))
  model.add(Dropout(0.1))

  model.add(Dense(30))


  model.summary()
  
  return model


# In[ ]:


model_cnn = model()
model_cnn.compile(optimizer='adam',loss='mse',metrics=['mae','accuracy']) ## MSE = Mean Squared Error, MAE = Mean Absolute Error


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) ## specifying random_state ensures that the results will remain consistent


# In[ ]:


history = model_cnn.fit(X_train, y_train, verbose = 1, batch_size=32, epochs = 75, validation_data=(X_test, y_test))


# In[ ]:


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


y_test = np.array(y_test)


# In[ ]:


def plot_sample(x, y, axis):
  img = x.reshape(96, 96)
  axis.imshow(img, cmap='gray') ## The Colormap instance or registered colormap name used to map scalar data to colors
  axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


fig = plt.figure(figsize=(10, 7))
fig.subplots_adjust(
  left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
  axis = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
  plot_sample(X_test[i], y_test[i], axis)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




