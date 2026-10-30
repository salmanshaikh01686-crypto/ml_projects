# DigitIQ — Handwritten Digit Recognizer

A professional Streamlit computer-vision application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on MNIST.

## Features
- Professional multi-tab Streamlit GUI
- Draw digits directly using mouse or touchscreen
- Upload handwritten digit images (PNG/JPG/JPEG)
- CNN predictions for digits 0–9
- Confidence score and probability chart
- Image preprocessing: grayscale, polarity handling, crop, resize and center to 28×28
- Training history and MNIST test metrics
- Saved Keras model for reuse (no retraining on every app launch)

## Project structure

```text
handwritten_digit_recognizer/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/       # trained model and metrics are created here
└── outputs/
```

## Requirements
- Windows 10/11
- Python 3.10 or 3.11 recommended
- Internet connection for first-time MNIST download and package installation
- 4 GB RAM recommended; CPU training is supported

## Setup in VS Code (Windows PowerShell)

Open the terminal in this project folder.

```powershell
py --version
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run this in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

If `pip` is not recognized, use `py -m pip` as shown above. You can also skip virtual-environment activation and use `py -m pip` directly.

## Step 1 — Train the CNN

```powershell
py train_model.py
```

The first run downloads MNIST. The script trains a CNN, evaluates it on the official test set and saves:
- `models/mnist_cnn.keras` — final model used by the GUI
- `models/best_mnist_cnn.keras` — best validation checkpoint
- `models/training_metrics.json` — test metrics
- `models/training_history.csv` — epoch history

Training may take several minutes on CPU. The GUI does not start until `models/mnist_cnn.keras` exists.

## Step 2 — Launch the GUI

```powershell
py -m streamlit run app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`.

## How to use
1. Open **Draw a digit** and draw one large digit in the canvas, then click **Recognize digit**.
2. Or open **Upload an image**, upload one handwritten digit per image, then click **Predict uploaded digit**.
3. Review the predicted class, confidence and probability distribution.
4. Open **Model & learning** to inspect architecture and training metrics.

## CNN architecture

- Input: 28×28 grayscale image
- Conv2D(32) + MaxPooling
- Conv2D(64) + MaxPooling
- Conv2D(96)
- Flatten
- Dense(128) + Dropout
- Dense(10) + Softmax

The model is trained with Adam and sparse categorical cross-entropy.

## Important image-input guidance
The MNIST dataset contains centered, grayscale 28×28 digit images. The app tries to normalize polarity, crop the ink, resize the digit and center it in a 28×28 image. Real-world phone images can still differ significantly from MNIST. Use one dark, clearly written digit per uploaded image and avoid backgrounds, shadows and multiple characters.

## Troubleshooting

### `py` is not recognized
Install Python 3.10 or 3.11 from python.org and enable **Add Python to PATH** during installation. Restart VS Code after installation.

### TensorFlow installation fails
Check that Python is 64-bit and use Python 3.10 or 3.11. Then run:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### `streamlit_drawable_canvas` import error
Make sure the requirements installation completed in the same Python environment used to run Streamlit.

### App says model is not trained
Run `py train_model.py` first, wait for the saved model message, and then restart Streamlit.

## Portfolio description
**DigitIQ — Handwritten Digit Recognition:** Built an interactive computer-vision application using TensorFlow/Keras and a CNN trained on MNIST. Implemented image preprocessing, drawing-canvas and image-upload inference, confidence visualization, and model evaluation metrics through a Streamlit dashboard.

## Limitations
This is an educational prototype. MNIST test accuracy is not a guarantee of performance on arbitrary handwriting, photographs, scanned documents or non-Latin scripts.
