# 🎧 Audio Classification with Transfer Learning (VGG16 & EfficientNet)

This project applies **transfer learning** to perform **audio classification** by converting audio signals into spectrogram images and leveraging **pretrained CNNs** like **VGG16** and **EfficientNetB1**.

By treating spectrograms as 2D images, we can utilize the power of **deep computer vision models** for robust **audio pattern recognition**.

---

## 🧠 What This Project Does

* 📦 Loads a custom audio dataset (`net_audio.zip`)
* 🔄 Converts `.wav` files into **Mel-spectrogram images**
* ⚙️ Fine-tunes pretrained models (VGG16, EfficientNetB1) for classification
* 📈 Evaluates accuracy, precision, recall, and visualizes confusion matrix
* 📊 Compares model performance using classification reports

---

## 🧱 Architecture Summary

```text
Audio Files (.wav)
   ↓
Librosa + Matplotlib
   ↓
Mel-Spectrogram Images (Saved as PNG/JPG)
   ↓
Image Preprocessing (Resize, Normalize)
   ↓
Pretrained CNN (VGG16 or EfficientNetB1)
   ↓
Dense Layer + Softmax
   ↓
Predicted Class
```

---

## 🗂️ Dataset

The dataset is a zipped folder `net_audio.zip` containing subfolders for each class of audio files.

Example structure:

```
net_audio/
├── airplane/
│   ├── a1.wav
│   └── ...
├── car_horn/
├── dog_bark/
├── siren/
└── ...
```

> ✅ The dataset is mounted from **Google Drive** and extracted into `/content/net_audio`.

---

## 🧪 Spectrogram Generation

Mel-spectrograms are generated using **Librosa**, with each `.wav` converted into a **log-magnitude mel scale** image.

Example code:

```python
y, sr = librosa.load(file_path)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
```

These are saved using `matplotlib.pyplot.imsave()` for use with CNNs.

---

## 🧠 Model Architecture

### VGG16

* Pretrained on ImageNet
* Top layers removed
* New dense layers added:

  * Flatten → Dense(256) → Dropout(0.5) → Dense(#classes)

### EfficientNetB1

* Lightweight, optimized CNN
* Pretrained weights from ImageNet
* Modified top layers:

  * GlobalAveragePooling → Dense → Dropout → Output Layer

---

## 🏁 Training Pipeline

1. Convert all audio files into spectrograms.
2. Split dataset into **train** and **test** using `train_test_split()`.
3. Encode labels using `LabelEncoder`.
4. Load images into numpy arrays for training.
5. Fine-tune VGG16 and EfficientNetB1 on spectrogram dataset.
6. Evaluate performance using `classification_report()` and `confusion_matrix()`.

---

## 📈 Results

* Both models reach **high classification accuracy**.
* Confusion matrices and classification reports show strong separation between audio classes.
* EfficientNetB1 performs better in terms of parameter efficiency.

---

## 📊 Visualizations

* Mel-spectrogram plots for sample audio files.
* Training loss and accuracy graphs (if training history is saved).
* Confusion matrix heatmaps using `seaborn.heatmap()`.

---

## ⚙️ Dependencies

```bash
pip install librosa matplotlib numpy seaborn scikit-learn tensorflow keras
```

Or in Colab:

```python
import librosa, keras, sklearn, matplotlib, seaborn
```

---

## 🚀 How to Run

1. Upload your audio dataset to Google Drive.
2. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Unzip the dataset:

```python
!unzip -q "/content/drive/My Drive/net_audio.zip" -d /content/net_audio
```

4. Run the notebook cells to generate spectrograms, train models, and evaluate.

---

## 📌 Notes

* Models are trained on a **small custom dataset**, so performance may vary.
* You can experiment with **other pretrained models** like ResNet, MobileNet, etc.
* Optionally, use **data augmentation** (e.g. time-shifting, pitch shifting) before spectrogram generation.

## EfficientNetB1 
<img width="633" height="618" alt="image" src="https://github.com/user-attachments/assets/ff59caa6-34bd-4636-b984-edfd5d5c3cbe" />
## VGG16 
<img width="655" height="596" alt="image" src="https://github.com/user-attachments/assets/2032bbbf-7db4-4305-8483-7e05aa579f7a" />
## CNN-LSTM Model
<img width="1124" height="539" alt="image" src="https://github.com/user-attachments/assets/91c90991-77e1-4e1c-83b6-c76e2441b8bc" />



