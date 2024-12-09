import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_and_prep_image(image_path, SIZE=32):
    """
    Load and preprocess image for prediction
    """
    try:
        # Load image
        img = Image.open("/Users/aaditya/Desktop/skin cancer/data/HAM10000_images_part_1/ISIC_0029289.jpg")
        
        # Resize
        img = img.resize((SIZE, SIZE))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None

def predict_skin_cancer(model_path, image_path):
    """
    Make prediction on new image
    """
    # Class names (in order of label encoding)
    class_names = ['Actinic keratoses (akiec)',
                  'Basal cell carcinoma (bcc)',
                  'Benign keratosis-like lesions (bkl)',
                  'Dermatofibroma (df)',
                  'Melanoma (mel)',
                  'Melanocytic nevi (nv)',
                  'Vascular lesions (vas)']
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Load and preprocess image
    img_array, original_img = load_and_prep_image(image_path)
    if img_array is None:
        return
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Display results
    plt.figure(figsize=(12, 4))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Input Image")
    plt.axis('off')
    
    # Plot prediction probabilities
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(len(predictions[0])), predictions[0] * 100)
    plt.xticks(range(len(predictions[0])), class_names, rotation=45, ha='right')
    plt.ylabel('Probability (%)')
    plt.title('Prediction Probabilities')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print prediction
    print("\nPrediction Results:")
    print("-" * 50)
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Print all class probabilities
    print("\nProbabilities for all classes:")
    print("-" * 50)
    for class_name, prob in zip(class_names, predictions[0]):
        print(f"{class_name}: {prob*100:.2f}%")

if __name__ == "__main__":
    # Paths to model and image
    MODEL_PATH = "skin_cancer_model.h5"  # Path to your saved model
    
    # You can modify this to accept command line arguments or create a simple UI
    # For now, just change this path to test different images
    IMAGE_PATH = "path_to_your_test_image.jpg"  # Change this to your test image path
    
    predict_skin_cancer(MODEL_PATH, IMAGE_PATH)