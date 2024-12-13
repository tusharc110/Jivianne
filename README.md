# Jivianne

Our project, Jivianne, is designed to help identify accidents from traffic camera footage using machine learning (ML). By classifying images into two categories—"Accident" and "Non-Accident"—we aim to assist in situations where it’s impractical for humans to monitor every camera feed. The solution is powered by Convolutional Neural Networks (CNNs) and Transfer Learning, with VGG16  at its core. This approach ensures high accuracy and speed, acting as an extra set of eyes in areas where timely detection can make a critical difference.


## Requirements

Ensure you have the following installed on your system:

- Python 3.8 or higher
- Jupyter Notebook or Jupyter Lab
- Required Python libraries:
  - TensorFlow
  - Keras
  - NumPy
  - Matplotlib
  - scikit-learn
  - Pandas
  - Gradio

You can install the required libraries using:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn pandas gradio
```


## File Descriptions

### 1. CNN.ipynb

- Provides code to build and train a CNN model from scratch.
- Includes steps for:
  - Data loading and preprocessing.
  - Model architecture definition.
  - Training and evaluation.

### 2. TRANSFER.ipynb

- Provides code to build and train a VGG16 transfer model from scratch.
- Includes steps for:
  - Data loading and preprocessing.
  - Model architecture definition.
  - Training and evaluation.

### 3. demo.py

- A demo script for testing the `modelTRANSFER.keras` model using Gradio.
- Able to upload an image and receive predictions indicating whether an accident is present or not.

## Running the Code

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Launch Jupyter Notebook for model training:

   ```bash
   jupyter notebook
   ```

3. Open the desired notebook (`CNN.ipynb` or `TRANSFER.ipynb`) and follow the instructions to train the models.

4. To test the `modelTRANSFER.keras` using the demo script:

   ```bash
   python demo.py
   ```

5. Use the Gradio interface to upload an image and view the prediction.


## Dataset Requirements

- Ensure that you have a labeled dataset for training and testing.
- Place your dataset in an accessible directory and update the notebook code to reflect its location.


## Notes

- Customize hyperparameters such as learning rate, batch size, and number of epochs as needed.
- Verify that your hardware supports GPU acceleration to speed up training.


