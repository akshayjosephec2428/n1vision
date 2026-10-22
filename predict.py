import os
import torch
from torchvision import models, transforms
from PIL import Image

# 1. Define your class names
class_names = [
    'Alfahm', 'Burger', 'Butter Chicken', 'Chicken Biriyani', 
    'Dosa', 'Idli', 'Kerala Sadhya', 'Pizza', 'Porotta', 'Sambar'
]

# 2. Nutrient Database normalized PER 100 GRAMS
nutrient_database_per_100g = {
    'Alfahm': {'calories': 220, 'protein': 22.0, 'carbs': 3.0, 'fat': 13.0},
    'Burger': {'calories': 295, 'protein': 14.0, 'carbs': 25.0, 'fat': 15.0},
    'Butter Chicken': {'calories': 180, 'protein': 11.0, 'carbs': 4.0, 'fat': 13.0},
    'Chicken Biriyani': {'calories': 200, 'protein': 9.0, 'carbs': 22.0, 'fat': 8.0},
    'Dosa': {'calories': 170, 'protein': 4.0, 'carbs': 29.0, 'fat': 4.0},
    'Idli': {'calories': 125, 'protein': 4.0, 'carbs': 26.0, 'fat': 0.5},
    'Kerala Sadhya': {'calories': 150, 'protein': 3.0, 'carbs': 22.0, 'fat': 5.0},
    'Pizza': {'calories': 266, 'protein': 11.0, 'carbs': 33.0, 'fat': 10.0},
    'Porotta': {'calories': 320, 'protein': 7.0, 'carbs': 42.0, 'fat': 13.0},
    'Sambar': {'calories': 60, 'protein': 2.0, 'carbs': 9.0, 'fat': 1.5}
}

# 3. Load the Trained Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

model_path = r'C:\Users\aksha\Downloads\food project\food_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# 4. Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_food():
    image_path = input("Enter the full path to a food image: ").strip('"')
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Ask for weight in grams
    try:
        weight_grams = float(input("Enter the estimated weight of the food in grams (e.g., 150): "))
    except ValueError:
        print("Invalid weight entered. Defaulting to 150g.")
        weight_grams = 150.0

    # Open and transform image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    food_item = class_names[predicted_idx.item()]
    conf_score = confidence.item() * 100

    print("\n" + "="*45)
    print(f"🍽️  Detected Food: {food_item} ({conf_score:.2f}% confidence)")
    print(f"⚖️  Portion Size: {weight_grams} grams")
    print("="*45)
    
    if food_item in nutrient_database_per_100g:
        base = nutrient_database_per_100g[food_item]
        # Scaling factor: (User Grams / 100)
        scale = weight_grams / 100.0
        
        calories = base['calories'] * scale
        protein = base['protein'] * scale
        carbs = base['carbs'] * scale
        fat = base['fat'] * scale

        print(f"📊 Scaled Nutritional Breakdown:")
        print(f"   • Calories: {calories:.1f} kcal")
        print(f"   • Protein:  {protein:.1f}g")
        print(f"   • Carbs:    {carbs:.1f}g")
        print(f"   • Fat:      {fat:.1f}g")
    else:
        print("⚠️ Nutritional data not found in database.")
    print("="*45 + "\n")

if __name__ == '__main__':
    predict_food()