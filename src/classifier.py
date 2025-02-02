import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import nn
import torch
from torchvision import models
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from utils import create_balanced_dataloaders, SpectrogramDataset, train_model, test_model



###########   Prepare Dataset for training ####################
image_folder = "../data/processed/1000mel_dataset_5/specs"
association_csv = "../data/processed/L1000dataset_5seg_valence.csv"

train_loader, val_loader, test_loader = create_balanced_dataloaders(image_folder, association_csv)

# Verify the splits
print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of validation samples: {len(val_loader.dataset)}")
print(f"Number of testing samples: {len(test_loader.dataset)}")



###########   MODEL ####################
num_classes = 3  # 
# Load a pre-trained model
model = models.resnet18(pretrained=True)
#model = models.resnet50(pretrained=True)
#model = models.resnet34(pretrained=True)

#resnet18.fc = nn.Linear(model.fc.in_features, num_classes)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),  # Intermediate layer
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes)  # Output layer for class
)


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Initialize a new W&B run
wandb.init(project='Audio_Class',  
    
    config={
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32,
    # Add other hyperparameters as needed
})
config = wandb.config

# Set training parameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
save_folder = "../results/res_classifier"

num_epochs=10
save_interval=1

# Training phase
for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct = 0, 0
    # Initialize tqdm progress bar for training
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=True)
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Calculate metrics
        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        # Update tqdm bar
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    train_loss /= len(train_loader.dataset)
    train_accuracy = train_correct / len(train_loader.dataset)
    progress_bar.close()
    
    # Log training metrics to W&B
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
    })
    
    # Validation phase
    model.eval()
    val_loss, val_correct = 0, 0
    # Initialize tqdm progress bar for validation
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=True)
    
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Calculate metrics
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            # Update tqdm bar
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    val_loss /= len(val_loader.dataset)
    val_accuracy = val_correct / len(val_loader.dataset)
    progress_bar.close()
    
            # Log validation metrics to W&B
    wandb.log({
        "epoch": epoch + 1,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    })
    # Print epoch summary
    print(f"Epoch [{epoch+1}/{num_epochs}]: "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    # Save the model at specific intervals
    if (epoch + 1) % save_interval == 0:
        save_path = os.path.join("saved_models", f"model_epoch_{epoch+1}.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, save_path)
        print(f"Model saved to {save_path}")
        
      
#### TESTING MODEL #####
class_names=None


model.eval()  # Set the model to evaluation mode
all_preds = []
all_labels = []
with torch.no_grad():  # Disable gradient calculation for testing
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        # Labels are already numeric (encoded 0-4), no need to transform
        # Forward pass
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)  # Get predicted class indices
        # Collect predictions and true labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

# Print results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")



# Optional: Confusion matrix
cm = confusion_matrix(all_labels, all_preds)


if class_names:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    # save the confusion matrix
    plt.savefig(os.path.join(save_folder, "confusion_matrix.png"))






