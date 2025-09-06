# UAV / Drone Detection with Deep Learning  

This project implements **drone detection and localization** using **deep learning models (InceptionV3, VGG16, ResNet variants)** on multiple UAV-related datasets. It performs image preprocessing, bounding box prediction, data augmentation, training, evaluation, and visualization of predictions.  

---

## 🚀 Features  

- 📂 Automatic dataset download from **KaggleHub**  
- 🖼 Image preprocessing (resize, normalization, annotation parsing)  
- 🔄 Data augmentation (flipping, rotation)  
- 🧠 Transfer learning using **InceptionV3** with fine-tuning  
- 🎯 Bounding box regression with **4-coordinate outputs**  
- 📊 Training visualization (accuracy & loss plots)  
- ✅ Early stopping for better generalization  
- 🔍 Testing on multiple datasets (UAV, Drone vs Bird, Amateur UAV detection)  
- 📦 Predictions with bounding box overlays for visualization  

---

## 📂 Datasets Used  

1. [Drone Dataset UAV](https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav)  
2. [Drone vs Bird Classification](https://www.kaggle.com/datasets/imbikramsaha/drone-bird-classification)  
3. [Amateur UAV Detection Dataset](https://www.kaggle.com/datasets/mcagriaksoy/amateur-unmanned-air-vehicle-detection-dataset)  

---

## ⚙️ Installation  

### 1. Clone Repository  
```bash
git clone https://github.com/yourusername/drone-detection.git
cd drone-detection
````

### 2. Install Dependencies

Make sure you have **Python 3.8+** installed. Then run:

```bash
pip install -r requirements.txt
```

### 3. Authenticate Kaggle (if needed)

```bash
pip install kagglehub
```

---

## 📖 Usage

### 1. Download Datasets

The script automatically downloads datasets using `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("dasmehdixtr/drone-dataset-uav")
```

### 2. Preprocess Images & Annotations

* Images are resized to **256x256**
* Normalized to `[0,1]`
* Bounding boxes extracted from annotation `.txt` files

### 3. Train the Model

```python
model_history = model.fit(
    augmented_train_images,
    augmented_train_targets,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=2
)
```

### 4. Evaluate & Predict

Run predictions on test sets or new datasets:

```python
predictions = model.predict(testImages)
```

Bounding boxes are drawn on test images using **OpenCV** and **Matplotlib**.

---

## 📊 Results

* **Training vs Validation Accuracy** plotted
* **Training vs Validation Loss** plotted
* Bounding box predictions visualized on multiple test images from each dataset

Example output:

* ✅ Green bounding boxes drawn around drones in test images
* ✅ Robust across datasets (Drone UAV, Drone vs Bird, Amateur UAV)

---

## 📦 Requirements

Main dependencies:

* Python 3.8+
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas, Matplotlib, Seaborn
* Pillow (PIL)
* Scikit-learn
* kagglehub

Install with:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn pillow scikit-learn kagglehub
```

---

## 📌 Notes

* Default input size: **256x256x3**
* Output layer: **4 neurons (bounding box coordinates)**
* Loss function: **Mean Squared Error (MSE)**
* Optimizer: **Adam**

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

Developed by **\[Your Name]** ✨

---

## ✅ To Do

* [ ] Improve bounding box regression with IoU-based loss (e.g., Smooth L1, GIoU)
* [ ] Add class prediction (drone vs bird)
* [ ] Export model for real-time inference (TensorRT / TFLite)
* [ ] Deploy as a web app (Flask / FastAPI / Streamlit)
👉 Do you want me to also generate a **`requirements.txt` file** automatically from your imports so the setup is easier?
```
