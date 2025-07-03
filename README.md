# 🔷 Shape Classifier: Neural Network for Shape Recognition

**Shape Classifier** is a modular Python project that uses a simple fully-connected neural network to classify geometric shapes—**circle**, **square**, **triangle**, or **unknown**—from grayscale PNG images. Built with PyTorch and visualized using Matplotlib, the project supports live visualization of predictions and layer activations.

---

## 🚀 Features

- **Custom Dataset Loader**: Loads labeled images from a folder structure (`data/circle`, `data/square`, etc.).
- **Neural Network Architecture**: 3-layer feedforward model with ReLU activations and softmax output.
- **Training Pipeline**: Modular training loop with live metrics for loss and accuracy.
- **Evaluation**: Generates a classification report with precision, recall, and F1-score.
- **Live Visualization**: Interactive GUI using Matplotlib to show predictions and neural activations in real time.
- **Extensible Codebase**: Follows clean architecture with separate scripts for model, data, training, evaluation, and visualization.

---

## 🛠️ Tech Stack & Libraries Used

- **Python 3.x**
- **Libraries**:
  - **PyTorch**: Neural network and training framework
  - **Torchvision**: Image transforms and datasets
  - **Matplotlib**: For interactive visualizations
  - **Scikit-learn**: For evaluation metrics
  - **Pillow**: Image processing
  - **argparse**: CLI interface

---

## 💻 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/nikpirat/shape-classifier.git
cd shape-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Organize Your Dataset

The data folder should look like this:

```
data/
├── circle/
│   ├── img_1.png
│   └── ...
├── square/
│   └── ...
├── triangle/
│   └── ...
└── unknown/
    └── ...
```

All images should be grayscale `.png` or `.jpg` files.

---

### 4. Train the Model

```bash
python main.py --train --data data --model model/shape_net.pt
```

### 5. Evaluate the Model

```bash
python main.py --eval --data data --model model/shape_net.pt
```

### 6. Run Live Visualization

```bash
python main.py --visualize --data data --model model/shape_net.pt
```

---

## 📊 Evaluation Metrics

After training and evaluation, the model achieved the following classification results:

```
              precision    recall  f1-score   support

      circle       0.96      0.81      0.88       100
      square       0.82      0.96      0.88       101
    triangle       0.98      0.96      0.97       100
     unknown       1.00      1.00      1.00       100

    accuracy                           0.93       401
   macro avg       0.94      0.93      0.93       401
weighted avg       0.94      0.93      0.93       401
```

---

## 📂 Project Structure

```
shape-classifier/
├── data/                  # Folder for training/eval images
│   ├── circle/
│   ├── square/
│   ├── triangle/
│   └── unknown/
├── model/                 # Saved PyTorch model (.pt)
├── src/
│   ├── model.py           # Neural network architecture
│   ├── dataset.py         # Custom dataset class
│   ├── train.py           # Training logic
│   ├── evaluate.py        # Evaluation logic
│   └── visualizer.py      # Live neural net visualization
├── main.py                # CLI entry point
└── README.md              # This file
```

---

## 🧠 Notes

- Make sure the `model/` directory exists before training, or the script will raise a `FileNotFoundError`.
- Visualization requires a display backend that supports `matplotlib.pyplot.show()`.
- Accuracy may vary depending on how balanced and clean your shape dataset is.

---

