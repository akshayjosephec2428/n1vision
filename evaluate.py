import os
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

def evaluate_model():
    # 1. Class names (must match your dataset classes)
    class_names = [
        'Alfahm', 'Burger', 'Butter Chicken', 'Chicken Biriyani', 
        'Dosa', 'Idli', 'Kerala Sadhya', 'Pizza', 'Porotta', 'Sambar'
    ]

    # 2. Paths
    data_dir = r'C:\Users\aksha\Downloads\food project'
    test_dir = os.path.join(data_dir, 'test')  # Uses your test folder

    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    # 3. Data Transforms (Same as validation transform)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Load Test Dataset
    test_dataset = datasets.ImageFolder(test_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Loaded {len(test_dataset)} test images across {len(class_names)} classes.")

    # 5. Load Trained Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

    model_path = os.path.join(data_dir, 'food_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 6. Run Evaluation Loop
    all_preds = []
    all_labels = []

    print("\nRunning inference on the test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 7. Print Classification Report (Precision, Recall, F1-Score)
    print("\n" + "="*70)
    print("📊 MODEL PERFORMANCE EVALUATION REPORT")
    print("="*70)
    
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    print("="*70)

if __name__ == '__main__':
    evaluate_model()