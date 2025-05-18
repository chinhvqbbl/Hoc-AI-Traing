import os
import json
import cv2

# Cấu hình đường dẫn
BASE_DIR = "dataset"
IMG_DIR = os.path.join(BASE_DIR, "img")
ANN_DIR = os.path.join(BASE_DIR, "ann")
OUTPUT_DIR = "output"

# Tạo thư mục output nếu chưa có
os.makedirs(os.path.join(OUTPUT_DIR, "with_mask"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "without_mask"), exist_ok=True)

def crop_faces_from_json(img_path, json_path, output_base):
    filename_prefix = os.path.splitext(os.path.basename(img_path))[0]

    # Load ảnh
    img = cv2.imread(img_path)
    if img is None:
        print(f"[⚠️] Không load được ảnh: {img_path}")
        return

    # Load annotation JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    count = {"with_mask": 0, "without_mask": 0}

    for obj in data.get("objects", []):
        label = obj.get("classTitle")
        if label not in count:
            continue

        (xmin, ymin), (xmax, ymax) = obj["points"]["exterior"]

        # Crop ảnh
        face_crop = img[ymin:ymax, xmin:xmax]

        # Tên ảnh lưu
        count[label] += 1
        output_filename = f"{filename_prefix}_{label}_{count[label]}.jpg"
        output_path = os.path.join(output_base, label, output_filename)

        # Lưu ảnh
        cv2.imwrite(output_path, face_crop)
        print(f"[✅] Đã lưu: {output_path}")

# Lặp tất cả file JSON
for json_file in os.listdir(ANN_DIR):
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(ANN_DIR, json_file)
    img_filename = os.path.splitext(json_file)[0]  # loại bỏ .json
    img_path = os.path.join(IMG_DIR, img_filename)

    if not os.path.exists(img_path):
        print(f"[⚠️] Không tìm thấy ảnh: {img_path}")
        continue

    crop_faces_from_json(img_path, json_path, OUTPUT_DIR)
