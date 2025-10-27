# ASL Alphabet Recognition

Real-time American Sign Language alphabet recognition using deep learning with PyTorch.

## Features
- **95%+ accuracy** on ASL alphabet classification
- Real-time webcam detection
- Word formation by spelling letters
- CNN architecture with 41M parameters
- Optimized for Apple Silicon (M2) GPU

## Model Architecture
- 3 convolutional layers (32, 64, 128 filters)
- MaxPooling for downsampling
- 2 fully connected layers
- Trained on 87,000 images (29 classes: A-Z + space, delete, nothing)

## Requirements
```
torch
torchvision
opencv-python
numpy
pillow
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/asl-alphabet-recognition.git
cd asl-alphabet-recognition
```

2. Install dependencies:
```bash
pip install torch torchvision opencv-python numpy pillow matplotlib
```

3. Download the ASL Alphabet dataset from [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet)

## Usage

### Train the model:
```bash
python asl_classifier.py
```

### Test on images:
```bash
python test_model.py
```

### Real-time letter recognition:
```bash
python webcam_demo.py
```

### Spell words:
```bash
python webcam_words.py
```
Hold each letter steady for ~1 second to add it to the word.

## Results
- Training Accuracy: 98.89%
- Validation Accuracy: 96.75%
- Real-time inference: ~30 FPS on M2 MacBook Pro

## Project Structure
```
asl-alphabet-recognition/
├── asl_classifier.py      # Model training script
├── test_model.py          # Test on validation images
├── webcam_demo.py         # Real-time letter detection
├── webcam_words.py        # Word formation by spelling
└── README.md
```

## Future Improvements
- Add support for full ASL word signs (not just alphabet)
- Improve hand detection with MediaPipe
- Deploy as web application
- Add support for continuous gesture recognition

## Author
Ilyas Mohammed - M.Sc. Computer Science Student at TU Darmstadt

## Dataset
[ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet) by Akash on Kaggle
