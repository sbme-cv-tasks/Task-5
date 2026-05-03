# resave_models.py
import sys
sys.path.insert(0, r"E:\Nodd\CV\Task-5")

from core.pca_scratch import PCA_Scratch  # لازم يتعرف عليه قبل الـ load

import joblib
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# ========== Load Data ==========
train_path = r"E:\Nodd\CV\Task-5\cropped_dataset\train"
test_path  = r"E:\Nodd\CV\Task-5\cropped_dataset\test"

X_train, y_train = [], []
X_test, y_test = [], []

for person in os.listdir(train_path):
    folder = os.path.join(train_path, person)
    for img_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100))
        X_train.append(img.flatten())
        y_train.append(person)

for person in os.listdir(test_path):
    folder = os.path.join(test_path, person)
    for img_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100))
        X_test.append(img.flatten())
        y_test.append(person)

X_train = np.array(X_train)
X_test  = np.array(X_test)

# ========== Label Encoder ==========
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# ========== PCA ==========
pca = PCA_Scratch(n_components=100)
X_train_pca = pca.fit_transform(X_train)

# ========== SVM ==========
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_pca, y_train_enc)

# ========== Save ==========
save_dir = r"E:\Nodd\CV\Task-5\saved_models"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(pca,       os.path.join(save_dir, "pca_model.pkl"))
joblib.dump(svm_model, os.path.join(save_dir, "svm_model.pkl"))
joblib.dump(le,        os.path.join(save_dir, "label_encoder.pkl"))

print("✓ Models saved successfully!")