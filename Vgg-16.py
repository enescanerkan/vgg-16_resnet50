import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pathlib

# Veri kümesinin URL'si
demo_dataset = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# Veri kümesini indirme ve çıkartma
directory = tf.keras.utils.get_file('flower_photos', origin=demo_dataset, untar=True)

# Veri dizini
data_directory = pathlib.Path(directory)

# Görüntü boyutları
img_height, img_width = 180, 180

# Batch boyutu
batch_size = 32

# Epoch sayısı
epochs = 3

# Eğitim ve doğrulama veri kümelerini oluşturma
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="validation",
    seed=123,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Model oluşturma
vgg16_model = Sequential()

# Evrişim Katmanları
vgg16_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(180, 180, 3)))
vgg16_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
vgg16_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

vgg16_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
vgg16_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
vgg16_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

vgg16_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
vgg16_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
vgg16_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
vgg16_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

vgg16_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
vgg16_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
vgg16_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
vgg16_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

vgg16_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
vgg16_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
vgg16_model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
vgg16_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Tam Bağlantılı Katmanlar
vgg16_model.add(Flatten())
vgg16_model.add(Dense(4096, activation='relu'))
vgg16_model.add(Dense(4096, activation='relu'))

# Çıkış katmanı (sınıf sayısı 5 olduğu için 5 nöronlu softmax katmanı)
vgg16_model.add(Dense(5, activation='softmax'))

# Model özetini görüntüleme
vgg16_model.summary()

# Model derleme
vgg16_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model eğitimi
history = vgg16_model.fit(train_ds, validation_data=validation_ds, epochs=epochs)

# Eğitim sonuçlarını görselleştirme
plt.figure(figsize=(8, 8))
epochs_range = range(epochs)
plt.plot(epochs_range, history.history['accuracy'], label="Training Accuracy")
plt.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
plt.axis(ymin=0.4, ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'])
plt.savefig('output-plot.png')
plt.show()
