import cv2
import numpy as np


class FaceDetector:
    def __init__(self):
        # Load OpenCV's pre-trained Haar cascade for frontal faces
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Standard ORL database image dimensions (width, height)
        self.target_size = (92, 112)

    def process_image(self, image_path):
        """
        Reads an image, detects a face, and prepares it for PCA.
        Returns: (display_image, prepared_face_matrix)
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read the image.")

        # Convert to grayscale for detection and PCA
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        display_img = img.copy()
        prepared_face = None

        # If a face is found, process the first one detected
        if len(faces) > 0:
            (x, y, w, h) = faces[0]

            # Draw a green bounding box on the color image for the UI
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the original grayscale image to just the face
            cropped_face = gray[y:y + h, x:x + w]

            # Resize to match the ORL dataset dimensions exactly
            prepared_face = cv2.resize(cropped_face, self.target_size)
            
        return display_img, prepared_face