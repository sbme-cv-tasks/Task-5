import os
import joblib
import cv2
import numpy as np
from core.pca_scratch import PCA_Scratch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class FaceRecognizer:
    def __init__(self):
        try:
            models_dir = os.path.join(BASE_DIR, "saved_models")
            self.pca = joblib.load(os.path.join(models_dir, "pca_model.pkl"))
            self.svm = joblib.load(os.path.join(models_dir, "svm_model.pkl"))
            self.le  = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found: {e}")
    
    def predict_face_with_match(self, img, train_path):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = cv2.resize(img, (100, 100))
        img_flat = img.flatten().reshape(1, -1)
        img_pca = self.pca.transform(img_flat)
        
        # SVM prediction
        pred = self.svm.predict(img_pca)
        prediction = str(self.le.inverse_transform(pred)[0])
        
        person_folder = os.path.join(train_path, prediction)
        best_match_path = None
        best_distance = float('inf')
        
        if os.path.exists(person_folder):
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                candidate = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                candidate = cv2.resize(candidate, (100, 100))
                candidate_pca = self.pca.transform(candidate.flatten().reshape(1, -1))
                
                distance = np.linalg.norm(img_pca - candidate_pca)
                if distance < best_distance:
                    best_distance = distance
                    best_match_path = img_path
        
        return prediction, best_match_path, best_distance