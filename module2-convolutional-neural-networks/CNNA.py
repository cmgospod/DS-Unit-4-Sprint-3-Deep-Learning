# Test file to ensure this works outside a notebook environment


import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model # This is the functional API
from skimage.io import imread_collection
from skimage.transform import resize
from sklearn.model_selection import train_test_split


resnet = ResNet50(weights='imagenet', include_top=False)
for layer in resnet.layers:
    layer.trainable = False

x = resnet.output
x = GlobalAveragePooling2D()(x) # This layer is a really fancy flatten
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(resnet.input, predictions)


forests = "/data/forest/*.jpg"
mountains = "/data/mountain/*.jpg"

forests = imread_collection(forests).concatenate()
mountains = imread_collection(mountains).concatenate()

y_0 = np.zeros(forests.shape[0])
y_1 = np.ones(mountains.shape[0])

X = np.concatenate([forests, mountains])
X = resize(X, (702, 224, 224, 3))
y = np.concatenate([y_0, y_1])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.5)
