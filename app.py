
from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)

# Load the model (the same code you used in Colab for loading the model)
model_path = 'C:/Users/Lavanya/Downloads/resnet.pth'

# model = models.resnet18(pretrained=True)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model.fc = torch.nn.Linear(model.fc.in_features, 10)  

checkpoint = torch.load(model_path, weights_only=True)
# checkpoint = torch.load(model_path)

adjusted_checkpoint = {}
for key in checkpoint:
    new_key = key.replace('fc1', 'fc')
    adjusted_checkpoint[new_key] = checkpoint[key]

model.fc.load_state_dict(adjusted_checkpoint, strict=False)
model.eval() 

# Define the classes (same as in your Colab code)
classes = [
    'Eczema', 
    'Warts Molluscum and other Viral Infections', 
    'Basal Cell Carcinoma (BCC)', 
    'Psoriasis', 
    'Melanocytic Nevi (NV)', 
    'Tinea Ringworm Candidiasis', 
    'Atopic Dermatitis', 
    'Benign Keratosis-like Lesions (BKL)', 
    'Seborrheic Keratoses and other Benign Tumors', 
    'Melanoma'
]

# Define the image transformation for preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict the disease from the image
def load_model_and_predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)  
    # Make predictions
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    predicted_class_name = classes[predicted_class.item()]
    return predicted_class_name

# Route for handling form submission
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            error = "No file part"
            return render_template('index.html', error=error)
        
        file = request.files['image']
        if file.filename == '':
            error = "No selected file"
            return render_template('index.html', error=error)
        
        if file:
            if not os.path.exists('static/uploads'):
                os.makedirs('static/uploads')

            image_path = os.path.join('static/uploads', file.filename)
            file.save(image_path)

            # Predict disease from the uploaded image
            disease_name = load_model_and_predict(image_path, model)

            # dictionary with plant information
            disease_to_plant = {
                'Eczema': {
                    'plant_name': ["Aloe Vera", "Muli", "Papaya", "Achillea biebersteinii"],
                    'image_url': ['aloevera.jpeg', 'Muli.jpg', 'papaya.jpg', 'Achillea.jpg']
                },
                "Warts Molluscum and other Viral Infections":{
                    'plant_name': ["Guarumbo","Greater celandine","Melaleuca alternifolia"],
                    'image_url': ['Guarumbo.webp', 'Greater.webp', 'Melaleuca.png']
                },
                'Basal Cell Carcinoma (BCC)':{
                    'plant_name': ["Aloe Vera","Curcumin"],
                    'image_url': ['aloevera.jpeg','Curcumin.htm']
                },
                'Psoriasis':{
                    'plant_name': ["Resveratrol","Aloe vera","Green tea","Curcumin"],
                    'image_url': ['aloevera.jpeg', 'Muli.jpg', 'papaya.jpg', 'Achillea.jpg']
                },
                'Melanocytic Nevi (NV)':{
                    'plant_name': ["Aloe vera","Green tea"],
                    'image_url': ['aloevera.jpeg', 'Green tea.jpg']
                },
                'Tinea Ringworm Candidiasis':{
                    'plant_name': ["Neem","Garlic","Calendula"],
                    'image_url': ['Neem.webp','Garlic.png','Calendula.jpg']
                },
                'Atopic Dermatitis':{
                    'plant_name':["hamamelis leaves","Chamomile","St John's wort"],
                    'image_url' :['hamamelis.jpeg','Chamomile.webp','Johnwort.jpg']
                }, 
                'Benign Keratosis-like Lesions (BKL)':{
                    'plant_name':["Chamomile","Alovera","Calendula"],
                    'image_url' :['Chamomile.webp','aloevera.jpg','Calendula.jpg']
                }, 
                'Seborrheic Keratoses and other Benign Tumors':{
                    'plant_name':["Alovera","Chamomile","Calendula"],
                    'image_url' :['aloevera.jpg','Chamomile.webp','Calendula.jpg']
                }, 
                'Melanoma':{
                    'plant_name':["Euphorbia peplus "],
                    'image_url' :['Euphorbia.jpg']
                }
            }

            # Retrieve plant information for the predicted disease
            if disease_name in disease_to_plant:
                plant_info = disease_to_plant[disease_name]
                plants = list(zip(plant_info['plant_name'], plant_info['image_url']))
                return render_template('result.html', disease=disease_name, plants=plants)
            else:
                error = "Disease not found. Please try another."
                return render_template('index.html', error=error)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)



