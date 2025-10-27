import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define model architecture (same as training)
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

# Load the trained model
model = ASL_CNN(num_classes=29)
model.load_state_dict(torch.load('asl_model.pth'))
model = model.to(device)
model.eval()

# Class names (A-Z + special)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'del', 'nothing', 'space']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting ASL Recognition...")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Define ROI (Region of Interest) for hand
    roi = frame[100:400, 100:400]
    
    # Draw rectangle for ROI
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
    
    # Convert ROI to PIL Image and preprocess
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    roi_tensor = transform(roi_pil).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(roi_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        conf_score = confidence.item() * 100
    
    # Display prediction on frame
    text = f"{predicted_class} ({conf_score:.1f}%)"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0, 255, 0), 3)
    
    cv2.putText(frame, "Place hand in green box", (100, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show frame
    cv2.imshow('ASL Recognition', frame)
    
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()