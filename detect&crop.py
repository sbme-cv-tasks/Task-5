import os
import cv2
import shutil

from core.detector import FaceDetector

# ------------------------
# Paths
# ------------------------
train_path = "split_dataset/train"
test_path = "split_dataset/test"

output_path = "cropped_dataset"

train_out = os.path.join(output_path, "train")
test_out = os.path.join(output_path, "test")

# ------------------------
# Init detector
# ------------------------
detector = FaceDetector()

# ------------------------
# Helper function
# ------------------------
def process_and_save(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    for person in os.listdir(input_root):
        person_in_path = os.path.join(input_root, person)

        if not os.path.isdir(person_in_path):
            continue

        person_out_path = os.path.join(output_root, person)
        os.makedirs(person_out_path, exist_ok=True)

        for img_name in os.listdir(person_in_path):
            img_path = os.path.join(person_in_path, img_name)

            try:
                _, face = detector.process_image(img_path)


                if face is None:
                    continue


                save_path = os.path.join(person_out_path, img_name)

                cv2.imwrite(save_path, face)

            except Exception as e:
                print(f"Error in {img_path}: {e}")

    print(f"Done processing: {input_root}")


# ------------------------
# Run on Train + Test
# ------------------------
process_and_save(train_path, train_out)
process_and_save(test_path, test_out)

print("All done! Cropped dataset created successfully.")