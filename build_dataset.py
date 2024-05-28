from datasets import load_dataset_builder, get_dataset_split_names
import json
import os
from PIL import Image
from alive_progress import alive_bar
import shutil

dataset_name = "HuggingFaceM4/FairFace"
# dataset = load_dataset("")


def view_dataset_info():
    ds_builder = load_dataset_builder(dataset_name, '0.25')
    print(f"Description: {ds_builder.info.description}")
    print(f"Features: {ds_builder.info.features}")
    print(f"Split names: {get_dataset_split_names(dataset_name, '0.25')}")


def build_cppe_dataset(split_name: str = "train"):
    files = [f for f in os.listdir("../Datasets/fairface/" + split_name + "/person") if f.endswith(".jpg")]
    image_id = 0
    labels = []
    for file in tqdm(files):
        image = Image.open("../Datasets/fairface/" + split_name + "/person/" + file)
        width, height = image.size
        labels.append({
            "image_id": image_id,
            "width": width,
            "height": height,
            "file_name": file,
            "objects": {
                "id": [0],
                "area": [width * height],
                "bbox": [[0, 0, width - 1, height - 1]],
                "category": [0],
            }
        })
        image_id += 1

    with open("../Datasets/fairface/" + split_name + "/person/metadata.jsonl", "w") as f:
        for label in labels:
            json.dump(label, f)
            f.write("\n")


def build_crowdhuman_dataset(split_name: str = "train"):
    base_folder = "/home/alex/school/ECS271/project/ECS271Project/Datasets/CrowdHuman"
    ann_file = split_name if split_name == "train" else "val"
    bound_file = f"{base_folder}/annotations/annotation_{ann_file}.odgt"
    bounding_dict = {}
    annotation_id = 0
    with open(bound_file, "r") as f:
        for line in f:
            label = json.loads(line)
            filename = label["ID"]
            bounding_dict[filename + ".jpg"] = label

    image_path = f"{base_folder}/{split_name}/person"
    files = [f for f in os.listdir(image_path) if f.endswith(".jpg")]

    labels = []
    with alive_bar(300, force_tty=True) as bar:
        for idx in range(300):
            file = files[idx]

            # if split_name == "train":
            #     shutil.copyfile(f"{image_path}/{file}", f"{base_folder}/train/{file}")
            # else:
            #     shutil.copyfile(f"{image_path}/{file}", f"{base_folder}/test/{file}")

            image = Image.open(f"{image_path}/{file}")
            width, height = image.size
            label = bounding_dict[file]
            objects = label["gtboxes"]
            num_objects = len(objects)

            obj_dict = {
                "id": [],
                "category": [],
                "bbox": [],
                "area": []
            }
            annotation_id += num_objects
            num_boxes = 0
            for obj in objects:
                if obj["tag"] == "person":
                    bounding_boxes = [obj["fbox"], obj["vbox"], obj["hbox"]]
                    for box in bounding_boxes:
                        if box is not None:
                            obj_dict["bbox"].append([box[0], box[1], box[2], box[3]])
                            obj_dict["area"].append(box[2] * box[3])
                            num_boxes += 1
            obj_dict["id"] = list(annotation_id + i for i in range(num_boxes))
            obj_dict["category"] = [0] * num_boxes
            annotation_id += num_boxes
            labels.append({
                "image_id": idx,
                "width": width,
                "height": height,
                "file_name": file,
                "objects": obj_dict
            })
            bar()

    with open(f"{base_folder}/{split_name}/person/metadata.jsonl", "w") as f:
        for label in labels:
            json.dump(label, f)
            f.write("\n")


if __name__ == "__main__":
    # build_cppe_dataset()
    # build_cppe_dataset("test")
    # build_cppe_dataset("validation")\
    build_crowdhuman_dataset()
    build_crowdhuman_dataset("test")
