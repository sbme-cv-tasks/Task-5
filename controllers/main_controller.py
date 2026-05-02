from core import FaceDetector

class MainController:
    def __init__(self):
        self.detector = FaceDetector()

    def handle_image_upload(self, file_path):
        """
        Passes the file path to the core detector.
        """
        display_img, prepared_face = self.detector.process_image(file_path)
        return display_img, prepared_face