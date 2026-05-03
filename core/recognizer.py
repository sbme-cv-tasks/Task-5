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

    def predict_face(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = cv2.resize(img, (100, 100))
        img = img.flatten().reshape(1, -1)
        img_pca = self.pca.transform(img)
        pred = self.svm.predict(img_pca)
        return str(self.le.inverse_transform(pred)[0])