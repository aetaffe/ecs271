import requests
import torch
import os
import cv2
import numpy as np
from pathlib import Path

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from alive_progress import alive_bar


# Name of repo on the hub or path to a local folder
model_name = "/home/alex/school/ECS271/project/ECS271Project/resnet50/detr-finetuned-cppe-5-10k-steps/checkpoint-8500"

image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(model_name)

# Load image for inference
# url = "https://images.pexels.com/photos/8413299/pexels-photo-8413299.jpeg?auto=compress&cs=tinysrgb&w=630&h=375&dpr=2"
# image = Image.open(requests.get(alt_file, stream=True).raw)


# alt_file = "/home/alex/school/ECS271/project/ECS271Project/Datasets/mst-e_data/images/PXL_20220922_131234792.jpg"
mst_folder = "/home/alex/school/ECS271/project/ECS271Project/Datasets/mst-e_data/images"


def save_image(image, filename):
    Path("images").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("images/" + filename, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))


def draw_bounding_box(image, box, color, thickness=6):
    image = np.array(image)
    x1, y1, x2, y2 = box
    color = tuple([int(c) for c in color])
    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), tuple(color), thickness)
    return image


def run_test():
    random_colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8")
    num_correct = 0
    num_total = 0
    files = [f for f in os.listdir(mst_folder) if f.endswith(".jpg")]
    with alive_bar(5, force_tty=True) as bar:
        for file in files:
            image = Image.open(f"{mst_folder}/{file}")

            # Prepare image for the model
            inputs = image_processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            # Post process model predictions
            # this include conversion to Pascal VOC format and filtering non confident boxes
            width, height = image.size
            target_sizes = torch.tensor([height, width]).unsqueeze(0)  # add batch dim
            results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

            if len(results["labels"]) != 0:
                num_correct += 1
            num_total += 1
            num_boxes = 0
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                confidence = round(score.item(), 3)
                if confidence <= 0.8:
                    continue
                box = [round(i, 2) for i in box.tolist()]
                image = draw_bounding_box(image, box, random_colors[num_boxes])
                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{confidence} at location {box}"
                )
                num_boxes += 1

            save_image(image, file)
            bar()

            if num_total == 5:
                break

    print(f"Accuracy: {num_correct / num_total}")


if __name__ == "__main__":
    run_test()
