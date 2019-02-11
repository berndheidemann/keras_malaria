import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

K.set_image_dim_ordering('th')  # color channel first

batch_size = 50
num_classes = 2
image_size = 64

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,  # maximal 20% Winkelveränderungen
    zoom_range=0.2,         # maximal 20% Zoom
    horizontal_flip=True,
    validation_split=0.3)   # Randomly flip inputs horizontally.


#test_datagen = ImageDataGenerator(rescale=1./255)  # only rescale for testing

train_generator = datagen.flow_from_directory(
    'data',  # this is the target directory
    # all images will be resized to 150x150
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = datagen.flow_from_directory(
    'data',
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

print(train_generator.samples)

print(validation_generator.samples)
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(
    3, image_size, image_size), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=300,
    validation_data=validation_generator,
    validation_steps=800 // batch_size)

# always save your weights after training or during training
model.save('first_try.h5')
