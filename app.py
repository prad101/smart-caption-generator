from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import base64
import os
import pickle
import pdb

# from models.ImageCaptionModel import ImageCaptionModel
from models.ImageCaptionModel import ImageCaptionModel, PositionalEncoding

# pdb.set_trace()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained model
# Allow the custom class to be loaded safely
torch.serialization.add_safe_globals([ImageCaptionModel])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('data/BestModel.pth',
                   map_location=device,
                   weights_only=False)
model.eval()

# Load vocabularies
with open('data/word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)
    
with open('data/index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

# Load pre-encoded validation images (for existing images feature)
with open('data/EncodedImageValidResNet.pkl', 'rb') as f:
    valid_img_embed = pickle.load(f)

# Image preprocessing for ResNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load ResNet encoder for new images
resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-2])  # Remove final layers
resnet.to(device)
resnet.eval()

def encode_image(image):
    """Encode image using ResNet"""
    with torch.no_grad():
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        features = resnet(img_tensor)
        # Reorder and flatten: [1, 2048, 7, 7] -> [1, 49, 2048] (if resnet 50)
        # Reorder and flatten: [1, 512, 7, 7] -> [1, 49, 512] (if resnet 18)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(3))
    return features

def generate_caption_from_embed(img_embed):
    """Generate caption from image embedding"""
    pad_token = word_to_index["<pad>"]
    start_token = word_to_index["<start>"]
    end_token = word_to_index["<end>"]
    max_seq_len = 33

    input_seq = torch.full((1, max_seq_len), pad_token, dtype=torch.long).to(device)
    input_seq[0, 0] = start_token
    
    predicted_sentence = []
    
    with torch.no_grad():
        for eval_iter in range(max_seq_len - 1):
            output, _ = model.forward(img_embed, input_seq)
            logits = output[eval_iter, 0, :]
            
            # Greedy decoding
            next_word_index = torch.argmax(logits).item()
            next_word = index_to_word[next_word_index]
            
            if next_word == '<end>':
                break
            
            if next_word not in ['<start>', '<pad>']:
                predicted_sentence.append(next_word)
            
            if eval_iter + 1 < max_seq_len:
                input_seq[0, eval_iter + 1] = next_word_index
    
    return " ".join(predicted_sentence)

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    try:
        data = request.json
        source = data.get('source')
        
        if source == 'upload':
            # Handle uploaded image
            image_data = data['image']
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Encode image
            img_embed = encode_image(image)
            
        elif source == 'existing':
            # Handle existing image
            image_name = data['image_name']
            if image_name not in valid_img_embed:
                return jsonify({'error': 'Image not found'}), 404
            
            img_embed = valid_img_embed[image_name].to(device)
            img_embed = img_embed.permute(0, 2, 3, 1)
            img_embed = img_embed.view(img_embed.size(0), -1, img_embed.size(3))
        
        else:
            return jsonify({'error': 'Invalid source'}), 400
        
        # Generate caption
        caption = generate_caption_from_embed(img_embed)
        
        return jsonify({
            'caption': caption,
            'success': True
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/list-images', methods=['GET'])
def list_images():
    """Return list of available existing images"""
    try:
        images = list(valid_img_embed.keys())
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from data/Images directory"""
    return send_from_directory('data/Images', filename)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    print("Starting Smart Caption Generator API...")
    print(f"Model loaded on: {device}")
    print(f"Vocabulary size: {len(word_to_index)}")
    app.run(debug=True, host='0.0.0.0', port=8000)