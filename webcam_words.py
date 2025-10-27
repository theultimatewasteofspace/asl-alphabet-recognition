import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import time

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define model architecture
class ASL_CNN(nn.Module):
    def __init__(self, num_classes=29):
        super(ASL_CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 25 * 25, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 25 * 25)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Load model
model = ASL_CNN(num_classes=29)
model.load_state_dict(torch.load('asl_model.pth'))
model = model.to(device)
model.eval()

# Class names
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'del', 'nothing', 'space']

# Transform
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Word building variables
current_word = ""
last_prediction = ""
last_change_time = time.time()
stable_frames = 0
STABLE_THRESHOLD = 15  # Number of frames before accepting a letter

# Start webcam
cap = cv2.VideoCapture(0)

print("ASL Word Recognition")
print("Hold each letter steady for 1 second")
print("Show 'space' gesture to add space")
print("Show 'del' gesture to delete last letter")
print("Press 'c' to clear word, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
    
    # Preprocess and predict
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    roi_tensor = transform(roi_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(roi_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        conf_score = confidence.item() * 100
    
    # Word building logic
    if predicted_class == last_prediction and predicted_class != 'nothing':
        stable_frames += 1
        
        # If held steady long enough, add to word
        if stable_frames >= STABLE_THRESHOLD:
            if predicted_class == 'space':
                current_word += ' '
            elif predicted_class == 'del':
                current_word = current_word[:-1]  # Remove last character
            elif predicted_class not in current_word[-1:]:  # Avoid duplicates
                current_word += predicted_class
            
            stable_frames = 0  # Reset after adding
            last_prediction = 'nothing'  # Require gesture change
    else:
        last_prediction = predicted_class
        stable_frames = 0
    
    # Display current letter
    cv2.putText(frame, f"Letter: {predicted_class} ({conf_score:.0f}%)", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display progress bar
    progress = min(stable_frames / STABLE_THRESHOLD, 1.0)
    bar_width = int(300 * progress)
    cv2.rectangle(frame, (100, 420), (400, 440), (50, 50, 50), -1)
    cv2.rectangle(frame, (100, 420), (100 + bar_width, 440), (0, 255, 0), -1)
    
    # Display current word
    cv2.putText(frame, f"Word: {current_word}", 
                (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    cv2.putText(frame, "Hold steady to add letter | 'c' clear | 'q' quit", 
                (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.imshow('ASL Word Recognition', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        current_word = ""

cap.release()
cv2.destroyAllWindows()

print(f"\nFinal word: {current_word}")