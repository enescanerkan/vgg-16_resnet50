# FLOWER CLASSIFICATION USING VGG16 AND RESNET50

This project implements image classification using two different models: VGG16 and ResNet50. (Bu proje, VGG16 ve ResNet50 adlı iki farklı model kullanarak görüntü sınıflandırması yapmaktadır.) Both models are trained on the Flower Photos dataset. (Her iki model, Flower Photos veri seti üzerinde eğitilmiştir.)


## TRAINING

Both models are trained on the Flower Photos dataset. (Her iki model, Flower Photos veri seti üzerinde eğitilmiştir.) The dataset consists of 5 classes of flower images: daisy, dandelion, rose, sunflower, and tulip. (Veri seti, papatya, karahindiba, gül, ayçiçeği ve lale olmak üzere 5 çiçek görüntü sınıfından oluşmaktadır.) 

### VGG16 Training Details

![new41](https://github.com/user-attachments/assets/6d0d0865-e149-4de9-afae-0588337aa4d7)

- **Architecture**: The VGG16 model consists of:
  - 13 convolutional layers
  - 5 max-pooling layers
  - 3 fully connected layers
- **Input Shape**: The input shape for VGG16 is set to `(224, 224, 3)`, where 224x224 is the image size and 3 represents the RGB color channels. (VGG16 için giriş şekli `(224, 224, 3)` olarak ayarlanmıştır; burada 224x224 görüntü boyutunu ve 3, RGB renk kanallarını temsil etmektedir.)
- **Activation Functions**: The ReLU activation function is used in convolutional layers, while softmax is used in the output layer. (Konvolüsyonel katmanlarda ReLU aktivasyon fonksiyonu, çıkış katmanında ise softmax kullanılmıştır.)
- **Loss Function**: Categorical Crossentropy is used as the loss function. (Kayıp fonksiyonu olarak Kategorik Çapraz Entropi kullanılmıştır.)
- **Optimizer**: Adam optimizer with a learning rate of 0.001 is used. (0.001 öğrenme oranına sahip Adam optimizasyonu kullanılmıştır.)
- **Epochs**: The model is trained for 50 epochs. (Model, 50 dönem boyunca eğitilmektedir.)
- **Batch Size**: A batch size of 32 is used during training. (Eğitim sırasında 32'lik bir grup boyutu kullanılmıştır.)

### ResNet50 Training Details

![1_rPktw9-nz-dy9CFcddMBdQ](https://github.com/user-attachments/assets/593590a0-380e-4945-9d38-9e7846b7db7b)

- **Architecture**: The ResNet50 model consists of:
  - 49 convolutional layers
  - 1 fully connected layer
- **Input Shape**: The input shape for ResNet50 is also `(224, 224, 3)`. (ResNet50 için giriş şekli de `(224, 224, 3)` olarak belirlenmiştir.)
- **Activation Functions**: ReLU activation is used for convolutional layers, and softmax is used for the output layer. (Konvolüsyonel katmanlarda ReLU aktivasyon fonksiyonu, çıkış katmanında softmax kullanılmıştır.)
- **Loss Function**: Categorical Crossentropy is used as the loss function. (Kayıp fonksiyonu olarak Kategorik Çapraz Entropi kullanılmıştır.)
- **Optimizer**: Adam optimizer with a learning rate of 0.001 is used. (0.001 öğrenme oranına sahip Adam optimizasyonu kullanılmıştır.)
- **Epochs**: The model is trained for 50 epochs. (Model, 50 dönem boyunca eğitilmektedir.)
- **Batch Size**: A batch size of 32 is used during training. (Eğitim sırasında 32'lik bir grup boyutu kullanılmıştır.)

### Results
After training, the accuracy of both models is plotted. (Eğitim tamamlandıktan sonra, her iki modelin doğruluğu çizilir.) The best-performing model is saved in the `.h5.keras` format. (En iyi performans gösteren model, `.h5.keras` formatında kaydedilmektedir.)

## LICENSE
This project is licensed under the MIT License. (Bu proje MIT Lisansı ile lisanslanmıştır.)
