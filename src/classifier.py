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

# %%
df=pd.read_csv('../data/raw/1000dataset.csv')

# keep only track_id and track_genre
dfg = df[['track_id', 'track_genre']]



# Path to the folder containing images
image_folder = "../data/raw/1000_mel_spec_seg"

# List all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# Create a list to store rows for the new CSV file
image_data = []

# Loop through image files
for image_file in image_files:
    # Extract the track_id from the filename (everything before "_segment")
    track_id = "_".join(image_file.split("_")[:-2])  # Adjust splitting based on naming
    
    # Look up the corresponding genre or class in the dataframe
    if track_id in dfg['track_id'].values:  # Assuming 'track_id' column exists in dfg
        row = dfg[dfg['track_id'] == track_id].iloc[0]
        
        # Extract the class or genre information
        class_info = row['track_genre']  # Replace with one-hot columns if needed
        
        # Add a row to the new data: [image path, class/genre]
        image_data.append([os.path.join(image_folder, image_file), class_info])
    else:
        print(f"Track ID {track_id} not found in the dataframe!")  # Debugging info

# Create a new dataframe for the association file
association_df = pd.DataFrame(image_data, columns=["image_path", "class"])
#association_df.head()


# label encode association_df
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
association_df['class'] = le.fit_transform(association_df['class'])

association_df.to_csv('../data/raw/Limage_class_association.csv', index=False)




###########   Prepare Dataset for training ####################

# Step 1: Group images by track ID and class
def group_tracks_by_class_and_id(association_csv):
    df = pd.read_csv(association_csv)
    df['track_id'] = df['image_path'].apply(lambda x: "_".join(os.path.basename(x).split("_")[:-2]))
    class_groups = defaultdict(list)
    
    # Group track IDs by their class
    for track_id, group in df.groupby('track_id'):
        track_class = group.iloc[0]['class']
        class_groups[track_class].append(track_id)
    
    return class_groups, df

# Step 2: Split track IDs for each class
def split_tracks_by_class(class_groups, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    train_ids, val_ids, test_ids = [], [], []
    
    for track_class, track_ids in class_groups.items():
        random.shuffle(track_ids)  # Shuffle track IDs within the class
        
        # Perform splits
        train, temp = train_test_split(track_ids, test_size=(1 - train_ratio))
        val, test = train_test_split(temp, test_size=(test_ratio / (test_ratio + val_ratio)))
        
        # Append to respective splits
        train_ids.extend(train)
        val_ids.extend(val)
        test_ids.extend(test)
    
    return train_ids, val_ids, test_ids


# %%
# Step 3: Create a custom PyTorch Dataset
class SpectrogramDataset(Dataset):
    def __init__(self, df, track_ids, transform=None):
        self.data = df[df['track_id'].isin(track_ids)]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        label = row['class']
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

# %%
# Step 4: Create DataLoaders
def create_balanced_dataloaders(image_folder, association_csv, batch_size=32):
    # Group by class and track ID
    class_groups, df = group_tracks_by_class_and_id(association_csv)
    
    # Perform class-balanced splits
    train_ids, val_ids, test_ids = split_tracks_by_class(class_groups)
    
    # Define image transformations (e.g., resizing, normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to a consistent size
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    
    # Create datasets
    train_dataset = SpectrogramDataset(df, train_ids, transform=transform)
    val_dataset = SpectrogramDataset(df, val_ids, transform=transform)
    test_dataset = SpectrogramDataset(df, test_ids, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


image_folder = "../data/raw/1000mel_spec_seg"
association_csv = "../data/raw/Limage_class_association.csv"

train_loader, val_loader, test_loader = create_balanced_dataloaders(image_folder, association_csv)

# Verify the splits
print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of validation samples: {len(val_loader.dataset)}")
print(f"Number of testing samples: {len(test_loader.dataset)}")


# %%
dataset = train_loader.dataset
#print(dataset[0])

###########   MODEL ####################

# Load a pre-trained model
model = models.resnet18(pretrained=True)

# Modify the final layer for the number of genres 
num_genres = 5  # Replace with the actual number of genres
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),  # Intermediate layer
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 5)  # Output layer for genres
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

# Access the configuration
config = wandb.config



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_interval=1):
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



##### TESTING MODEL #####

def test_model(model, test_loader, device, class_names=None):
    """
    Test the trained model on the test dataset.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set.
        device: Torch device (CPU or CUDA).
        class_names: List of class names corresponding to the labels (optional).
    
    Returns:
        None. Prints accuracy, precision, recall, and F1-score.
    """
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


# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
save_folder = "../models/classifier"

# %%
# Assuming model, train_loader, val_loader, criterion, optimizer are defined
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")


# %%
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# %%
# test model
test_model(model, test_loader, device)

# %%



