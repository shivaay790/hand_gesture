# Hand Gesture Controlled Drone with Voice Commands

This project integrates hand gesture recognition and basic voice commands to control a drone. It uses MediaPipe for landmark detection and a trained classifier to interpret gestures. Voice commands act as an assistive feature.

## Features

- Real-time hand gesture detection using webcam
- 3D hand landmark extraction via MediaPipe
- Voice command input and processing (planned Whisper support)
- Gesture classification using a trained ML model
- Data augmentation and preprocessing
- Handles both left and right hands
- Error handling for no-hand or invalid input

## Workflow

### 1. Dataset Preparation
- Hand gesture images collected and augmented
- Labels include `up`, `down`, `left`, `right`, `front`, `back`, `flip`, etc.

### 2. Feature Extraction
- Hand landmarks (21 per hand) are extracted using MediaPipe
- Features include (x, y, z, visibility) for each landmark, + hand type (left/right)

### 3. Model Training
- A classification model is trained using Keras/TensorFlow
- The model classifies hand gestures into drone commands

### 4. Real-Time Drone Control
- Webcam used to identify gestures in real time
- Commands are mapped to simulate drone actions
- Includes left/right-hand-specific handling and visual feedback

### 5. Voice Command Integration
- Simple voice recognition implemented
- Planned: switch to Whisper for more robust voice-to-text conversion

## Dependencies

- `opencv-python`
- `mediapipe`
- `numpy`
- `tensorflow`
- `speechrecognition`
- `whisper` 

## Usage

```bash
pip install -r requirements.txt
# Launch notebook or convert it
jupyter notebook main3.ipynb
```

---

## README for `strach_from_CNN_final.ipynb`


# Symbol-Based Gesture Recognition using CNN (From Scratch)

This notebook trains a convolutional neural network (CNN) from scratch to classify hand gesture images. It includes data preprocessing, augmentation, training, and evaluation with a clean pipeline.

## Highlights

- Converts raw hand gesture images into clean datasets
- Augments data using Keras’ `ImageDataGenerator`
- Builds a multi-layer CNN architecture
- Evaluates performance using confusion matrix
- Prepares model for future integration into simulation tools like PyBullet

## Workflow

### 1. Image Conversion
- Raw images resized to (1280x720) and saved per class
- Output images are stored in `dataset/raw`

### 2. Data Augmentation
- Techniques include:
  - Rotation, shear, zoom
  - Brightness variation
  - Fill mode = 'nearest'
- Augmented images saved under `dataset/augmented`

### 3. CNN Model Architecture

- **Input:** 128x128 RGB image
- **Conv Layers:** 4 Conv2D layers with increasing filters
- **Pooling:** MaxPooling2D + GlobalAveragePooling2D
- **Dense Layers:** Fully connected + softmax output
- **Output:** 8 gesture classes

### 4. Training

- Uses `ImageDataGenerator` with `flow_from_directory`
- 10 epochs of training with validation split
- Model saved as `gesture_model.h5`

### 5. Evaluation

- Confusion matrix plotted using `seaborn`
- Class balance check before and after training

## Dependencies

- `opencv-python`
- `numpy`
- `tensorflow`
- `seaborn`
- `pybullet` 
- `sklearn`

## Usage

```bash
# Step 1: Convert raw images
# convert_to_raw('dataset/org', 'dataset/raw')

# Step 2: Augment the dataset
# augment_images('dataset/raw', 'dataset/augmented')

# Step 3: Train CNN
jupyter notebook strach_from_CNN_final.ipynb
