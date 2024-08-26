import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
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

# Eğitim ve doğrulama veri kümelerini oluşturma !parametrelere bak data augmentation
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

# Eğitim veri kümesinden birkaç örneği görselleştirme
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for var in range(6):
        ax = plt.subplot(3, 3, var + 1)
        plt.imshow(images[var].numpy().astype("uint8"))
        plt.axis("off")

# Önceden eğitilmiş ResNet50 modelinin yüklenmesi
pretrained_model_for_demo = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(180, 180, 3),
    pooling='avg',
    weights='imagenet'
)

# Önceden eğitilmiş modelin katmanlarını dondurma
for each_layer in pretrained_model_for_demo.layers:
    each_layer.trainable = False

# Model oluşturma
demo_resnet_model = Sequential()
demo_resnet_model.add(pretrained_model_for_demo)
demo_resnet_model.add(Flatten())
demo_resnet_model.add(Dense(512, activation='relu'))
demo_resnet_model.add(Dense(5, activation='softmax'))

# Modeli derleme optimizer ,loss
demo_resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# ModelCheckpoint geri arama işlevini tanımlama
checkpoint_callback = ModelCheckpoint(filepath='model_{epoch:02d}.h5.keras', save_best_only=True)

# Modeli eğitme ve ModelCheckpoint'i kullanarak kaydetme
history = demo_resnet_model.fit(train_ds, validation_data=validation_ds, epochs=epochs, callbacks=[checkpoint_callback])

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

# Eğitim ve doğrulama veri kümesindeki sınıfları ve görüntü dizinlerini inceleme
class_names = train_ds.class_names
num_classes = len(class_names)

print("Eğitim ve Doğrulama Veri Kümesindeki Sınıflar ve Dizinler:")
for class_name in class_names:
    class_dir = data_directory / class_name
    print(f"Sınıf: {class_name}, Dizin: {class_dir}")

# Eğitim veri kümesinden birkaç örnek görüntüyü görselleştirme
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[tf.argmax(labels[i]).numpy()])
        plt.axis("off")
plt.show()
