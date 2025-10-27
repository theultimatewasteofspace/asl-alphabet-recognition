import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import random

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define the same model architecture
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

# Load the saved model
model = ASL_CNN(num_classes=29)
model.load_state_dict(torch.load('asl_model.pth'))
model = model.to(device)
model.eval()

print("Model loaded successfully!")

# Load validation data
train_dir = 'asl_alphabet_train/asl_alphabet_train'

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Test on random images
indices = random.sample(range(len(val_dataset)), 5)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i, idx in enumerate(indices):
    img_tensor, true_label = val_dataset.dataset[val_dataset.indices[idx]]
    
    img_tensor_batch = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor_batch)
        _, predicted = torch.max(output, 1)
    
    class_names = full_dataset.classes
    true_class = class_names[true_label]
    pred_class = class_names[predicted.item()]
    
    # Denormalize
    img = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    
    axes[i].imshow(img)
    axes[i].set_title(f'True: {true_class}\nPred: {pred_class}', 
                     color='green' if true_class == pred_class else 'red')
    axes[i].axis('off')

plt.tight_layout()
plt.show()