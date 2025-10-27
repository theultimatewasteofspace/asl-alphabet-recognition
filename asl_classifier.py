import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os

# Set data paths
train_dir = 'asl_alphabet_train/asl_alphabet_train'
test_dir = 'asl_alphabet_test'

# Let's see what we have
print("Classes in training set:")
print(os.listdir(train_dir))
print(f"\nNumber of classes: {len(os.listdir(train_dir))}")

# Check how many images per class
for letter in os.listdir(train_dir):
    letter_path = os.path.join(train_dir, letter)
    if os.path.isdir(letter_path):
        num_images = len(os.listdir(letter_path))
        print(f"{letter}: {num_images} images")

# Let's visualize some sample images
#from PIL import Image

# Pick a letter to look at
#sample_letter = 'A'
#sample_path = os.path.join(train_dir, sample_letter)

# Get first 9 images from that letter
#image_files = os.listdir(sample_path)[:9]

# Create a grid to display them
#fig, axes = plt.subplots(3, 3, figsize=(10, 10))
#fig.suptitle(f'Sample images for letter: {sample_letter}')

#for i, ax in enumerate(axes.flat):
    #img_path = os.path.join(sample_path, image_files[i])
    #img = Image.open(img_path)
    #ax.imshow(img)
    #ax.axis('off')

#plt.tight_layout()
#plt.show()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load only the training dataset
full_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

# Split it into train (80%) and validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

print(f"\nTotal images: {len(full_dataset)}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Number of classes: {len(full_dataset.classes)}")

# Create DataLoaders to feed data in batches
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"\nNumber of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")

# Define the CNN model
class ASL_CNN(nn.Module):
    def __init__(self, num_classes=29):
        super(ASL_CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 25 * 25, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(torch.relu(self.conv1(x)))
        # Conv block 2
        x = self.pool(torch.relu(self.conv2(x)))
        # Conv block 3
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 25 * 25)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Create the model
model = ASL_CNN(num_classes=29)
print(model)

# Set up training
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device} (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using device: {device}")


model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nModel is ready to train!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 500 batches
            if (i + 1) % 500 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

print("Training function ready. Run train_model() to start training.")

# Start training
print("Starting training... This will take a while.")
train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10
)
train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10
)

# Save model immediately after training
torch.save(model.state_dict(), 'asl_model.pth')
print("\n✓ Model saved as asl_model.pth")