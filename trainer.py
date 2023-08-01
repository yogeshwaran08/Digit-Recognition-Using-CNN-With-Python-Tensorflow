import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
train_path = "./input/train.csv"
test_path = "./input/test.csv"

# Load Data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


y = train["label"]
x = train.drop(labels=["label"], axis=1)

x = x/255.0
test = test/255.0

x = x.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# label encoding
y = to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=45)
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3),
          activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(10, activation="softmax"))


optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])


datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    zoom_range=0.01,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)

train_gen = datagen.flow(X_train, y_train, batch_size=128)
test_gen = datagen.flow(X_test, y_test, batch_size=128)


epochs = 100
batch_size = 128
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc",
    patience=5,
    verbose=1,
    mode="max",
    restore_best_weights=True,
)

history = model.fit(train_gen,
                    epochs=epochs,
                    steps_per_epoch=X_train.shape[0],
                    validation_data=test_gen,
                    validation_steps=X_test.shape[0],
                    callbacks=es)

model.save("./output/model.h5")
