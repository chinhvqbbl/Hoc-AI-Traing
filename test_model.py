import torch
from torchvision import models, transforms
import cv2
from PIL import Image

# ==== CONFIG ====
MODEL_PATH = "resnet50_face_mask.pth"
CLASS_NAMES = ["with_mask", "without_mask"]

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== LOAD MODEL ====
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# ==== OPEN WEBCAM ====
cap = cv2.VideoCapture(0)  # 0 là webcam mặc định

if not cap.isOpened():
    print("❌ Không mở được webcam")
    exit()

print("🎥 Đang chạy webcam... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize nhỏ khung nhìn để predict nhanh
    input_frame = cv2.resize(frame, (224, 224))
    img = Image.fromarray(cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        label = CLASS_NAMES[pred.item()]

    # Vẽ lên frame gốc
    cv2.putText(frame, f"{label}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "with_mask" else (0, 0, 255), 2)

    cv2.imshow("Mask Detection (Press q to exit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
