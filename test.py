import os
import torch
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Import the model class definition
from train import OCRModel, remove_duplicates




import torch
import torch.serialization
from sklearn.preprocessing import LabelEncoder



def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    # Open and resize the image
    image = Image.open(image_path)
    image = image.resize((128, 64), resample=Image.BILINEAR)
    
    # Convert to numpy array
    image = np.array(image)
    
    # Convert to grayscale if image is in color
    if len(image.shape) == 3 and image.shape[2] > 1:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # Ensure it's a single channel image
    image = np.expand_dims(image, axis=2)
    
    # Normalize image to [0,1] range
    image = image / 255.0
    
    # Reshape to tensor format (C, H, W)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    
    # Convert to PyTorch tensor
    image_tensor = torch.tensor(image, dtype=torch.float).unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def decode_prediction(pred, label_encoder):
    """Decode the model's prediction to text"""
    pred = torch.softmax(pred, 2)
    pred = torch.argmax(pred, 2)
    pred = pred.detach().cpu().numpy()
    
    result = []
    for k in pred[0]:  # Only one sequence in the batch
        if k == 0:  # Blank token
            result.append('°')
        else:
            try:
                # Adjust index to match encoder (subtract 1)
                p = label_encoder.classes_[k-1]
                result.append(p)
            except IndexError:
                result.append('°')
    
    # Join characters and remove duplicates
    text = "".join(result)
    cleaned_text = remove_duplicates(text)
    
    return cleaned_text

def visualize_prediction(image_path, prediction):
    """Display the image and prediction"""
    image = Image.open(image_path)
    plt.figure(figsize=(10, 5))
    plt.imshow(np.array(image), cmap='gray')
    plt.title(f"Prediction: {prediction}")
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='OCR Model Inference')
    parser.add_argument('--model_path', type=str, default='ocr_model_best.pth', 
                        help='Path to the saved model')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Path to the image for prediction')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize the image and prediction')
    
    args = parser.parse_args()
    
    # Check if model and image exist
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.image_path):
        print(f"Image file not found: {args.image_path}")
        return
    
    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Add LabelEncoder to safe globals
    torch.serialization.add_safe_globals([LabelEncoder])
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying with weights_only=False...")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    label_encoder = checkpoint['label_encoder']
    num_classes = checkpoint['num_classes']
    
    # Initialize model
    model = OCRModel(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Preprocess image
    image_tensor = preprocess_image(args.image_path)
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction, _ = model(image_tensor)
    
    # Decode prediction
    text_prediction = decode_prediction(prediction, label_encoder)
    
    # Print result
    print(f"Predicted text: {text_prediction}")
    
    # Visualize if requested
    if args.visualize:
        visualize_prediction(args.image_path, text_prediction)

if __name__ == "__main__":
    main()