import os
import sys
import numpy as np
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress annoying TensorFlow warnings

from tf_keras.models import load_model

class ModuleClassifier:
    def __init__(self, model_dir):
        """Loads both the .h5 model and the labels.txt file from the given directory."""
        self.model_path = os.path.join(model_dir, "keras_model.h5")
        self.labels_path = os.path.join(model_dir, "labels.txt")
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.labels_path):
            print(f"Error: Could not find model files in {model_dir}")
            sys.exit(1)
            
        print("Loading Keras model... (This may take a few seconds)")
        # compile=False is necessary because Teachable Machine models aren't meant to be retrained
        self.model = load_model(self.model_path, compile=False)
        
        # Load the labels into a list
        self.labels = []
        with open(self.labels_path, "r") as f:
            for line in f.readlines():
                # Teachable machine format is usually "0 ClassName"
                parts = line.strip().split(' ', 1)
                label = parts[1] if len(parts) > 1 else parts[0]
                self.labels.append(label)
                
        print(f"Model loaded successfully! Found {len(self.labels)} classes: {self.labels}")

    def predict(self, cv2_image):
        """
        Takes a raw OpenCV crop, preprocesses it to Google Teachable Machine specs,
        and returns the (Best Label, Confidence Score)
        """
        # 1. Resize to exactly 224x224
        # Note: cv2 resize ignores aspect ratio if you force it. If your crops are perfectly square, this is fine.
        image = cv2.resize(cv2_image, (224, 224), interpolation=cv2.INTER_AREA)

        # 2. Turn the image into a numpy array (Teachable machine expects float32)
        image_array = np.asarray(image, dtype=np.float32)

        # 3. Normalize the image array from [0-255] to [-1 to 1]
        normalized_image_array = (image_array / 127.5) - 1

        # 4. Create the batch tensor required by keras: shape becomes (1, 224, 224, 3)
        data = np.expand_dims(normalized_image_array, axis=0)

        # 5. Run inference
        prediction_array = self.model.predict(data, verbose=0)
        
        # 6. Extract the highest confidence result
        best_index = np.argmax(prediction_array[0])
        best_label = self.labels[best_index]
        confidence = float(prediction_array[0][best_index])
        
        return best_label, confidence

# If we run this script directly, test it with a sample image!
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(script_dir, "..", "models", "cw_keras")
    
    classifier = ModuleClassifier(model_folder)
    
    # We create a dummy test purely to warm up the model
    # (The first prediction is always slow, we want to see it work)
    dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)
    print("\nRunning test on dummy blank image to warm up GPU/CPU...")
    
    label, conf = classifier.predict(dummy_image)
    print(f"Result: {label} ({conf * 100:.1f}%)")
    
    print("\nTest passed. The inference module is ready to accept cropped images from cv_discovery.py!")
