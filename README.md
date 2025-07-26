# Dog vs Cat Image Classifier

This project is a deep learning-based image classifier that predicts whether a given image is of a **dog** or a **cat** using a Convolutional Neural Network (CNN) model built with TensorFlow and Keras.

## Live Demo

[Click here to try it out!](https://huggingface.co/spaces/SamvedaB/DogCatClassifier)

## About the Project

The goal of this project is to create an image classification model that can distinguish between dog and cat images. It uses a CNN model trained on a labeled dataset of dog and cat images and deployed using **Streamlit** on **Hugging Face Spaces** for interactive use.

### Features

- Upload an image and get a prediction instantly
- Built using a simple and efficient CNN model
- Easy to use and accessible via Hugging Face

## Model Overview

- **Input Shape**: (128, 128, 3)
- **Model Type**: Convolutional Neural Network
- **Framework**: TensorFlow/Keras
- **Activation**: ReLU and Sigmoid
- **Output**: Binary classification (`Dog` or `Cat`)

## Files in the Repository

| File Name                  | Description                                      |
|---------------------------|--------------------------------------------------|
| `app.py`                  | Streamlit app script                             |
| `cat_dog_classifier.keras`| Trained CNN model saved in Keras format          |
| `CNNProject.ipynb`        | Jupyter notebook containing model training code  |
| `requirements.txt`        | List of dependencies                            |
| `.gitignore`              | Specifies files to be excluded from Git tracking |

## Tech Stack

- Python
- TensorFlow/Keras
- Streamlit
- Hugging Face Spaces

## How to Use

1. Open the [Live App](https://huggingface.co/spaces/SamvedaB/DogCatClassifier)
2. Upload a JPEG or PNG image of a dog or a cat
3. Wait for the prediction to be displayed

## Installation (For Local Use)

```bash
git clone https://github.com/SamvedaBoja/DogCatCNN.git
cd DogCatCNN
pip install -r requirements.txt
streamlit run app.py
