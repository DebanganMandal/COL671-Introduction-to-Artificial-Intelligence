import sys

import torch 
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def test_model(model, test_loader, criterion, device, output_csv='bird.csv'):
    model.eval()  
    total = 0
    results = []
    with torch.no_grad():  
        for images, _ in test_loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            # Get predictions
            # Save predicted labels
            results.extend(predicted.cpu().numpy())

    # Write only predicted labels to a CSV file
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted_Label'])
        for label in results:
            writer.writerow([label])


class BirdClassifierCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BirdClassifierCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)        
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # Output layer with num_classes outputs
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers
        x = self.fc_layers(x)
        return x



from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

target_layer = model.conv_layers[-1]

cam = GradCAM(model=model, target_layers=[target_layer])

model.eval()

# Set the number of images per row in the grid
grid_size = 4  # Adjust grid size based on batch size or desired layout

for images, labels in val_loader:
    images, labels = images.to(device), labels.to(device)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle("Grad-CAM Visualizations", fontsize=16)
    
    for i in range(min(images.shape[0], grid_size * grid_size)):
        row = i // grid_size
        col = i % grid_size

        rgb_img = images[i].cpu().numpy().transpose(1, 2, 0)
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())  # Normalize for display

        # Preprocess for Grad-CAM
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        target_category = labels[i].item()

        # Generate Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_category)])[0]

        # Overlay CAM on the image
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Display the Grad-CAM in the grid
        axes[row, col].imshow(cam_image)
        axes[row, col].set_title(f'Class: {target_category}')
        axes[row, col].axis('off')

    # Hide any unused subplots if batch size < grid size squared
    for j in range(i + 1, grid_size * grid_size):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()

    # Only visualize the first batch for demonstration
    break
        
if __name__ == "__main__": 
    dataPath = sys.argv[1]
    trainStatus = sys.argv[2]
    modelPath = sys.argv[3] if len(sys.argv) > 3 else "default/model/path"


    if trainStatus == "train":
        print("training")

        full_dataset = datasets.ImageFolder(root=dataPath)
        train_indices, val_indices = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        batch_size = 32

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BirdClassifierCNN(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        num_epochs = 15
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                # Move data to device
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

            # Validation loop
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    # Move validation data to device
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Validation Accuracy: {accuracy:.2f}%')

        torch.save(model.state_dict(), modelPath)
    else:
        print("infer")
        test_dataset = datasets.ImageFolder(root=dataPath)
        batch_size = 1
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = BirdClassifierCNN()
        model.load_state_dict(torch.load(modelPath))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        test_model(model, test_loader, criterion, device)
    print(f"Training: {trainStatus}")
    print(f"path to dataset: {dataPath}")
    print(f"path to model: {modelPath}")
