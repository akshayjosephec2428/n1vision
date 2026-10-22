import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def main():
    # 1. Path to your dataset folders
    data_dir = r'C:\Users\aksha\Downloads\food project'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validate')

    # 2. Data Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validate': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 3. Load Datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'validate': datasets.ImageFolder(val_dir, data_transforms['validate'])
    }

    # IMPORTANT: num_workers MUST be 0 on Windows to avoid multiprocessing crashes
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=0),
        'validate': DataLoader(image_datasets['validate'], batch_size=32, shuffle=False, num_workers=0)
    }

    class_names = image_datasets['train'].classes
    print(f"Detected {len(class_names)} food classes: {class_names}")

    # 4. Model Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 5. Training Loop
    num_epochs = 5
    print("\nStarting training...")

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 15)

        for phase in ['train', 'validate']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Save model
    save_path = os.path.join(data_dir, 'food_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete! Model saved successfully to: {save_path}")

if __name__ == '__main__':
    main()