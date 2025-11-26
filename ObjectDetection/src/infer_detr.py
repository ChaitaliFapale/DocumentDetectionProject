import os
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


MODEL_PATH = r"C:\Users\Lenovo\OneDrive\Desktop\AIproject\ObjectDetection\models\detr_passport"
IMAGE_PATH = r"C:\Users\Lenovo\OneDrive\Desktop\AIproject\ObjectDetection\src\test.png"


if not os.path.exists(os.path.join(MODEL_PATH, "preprocessor_config.json")):
    print("[INFO] preprocessor_config.json not found. Saving default processor to model folder...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    processor.save_pretrained(MODEL_PATH)
else:
    processor = DetrImageProcessor.from_pretrained(MODEL_PATH)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetrForObjectDetection.from_pretrained(MODEL_PATH).to(device)
model.eval()


image = Image.open(IMAGE_PATH).convert("RGB")
width, height = image.size


inputs = processor(images=image, return_tensors="pt").to(device)


with torch.no_grad():
    outputs = model(**inputs)


target_sizes = torch.tensor([[height, width]]).to(device)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]


plt.figure(figsize=(10, 8))
plt.imshow(image)
ax = plt.gca()

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.5: 
        x_min, y_min, x_max, y_max = box.cpu().numpy()

        
        corners = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max)
        ]

        poly = Polygon(corners, closed=True, edgecolor="red", linewidth=2, facecolor="none")
        ax.add_patch(poly)
        ax.text(x_min, y_min, f"passport: {score:.2f}", color="yellow", fontsize=12, weight="bold")


ax.add_patch(Rectangle((0, 0), width, height, linewidth=3, edgecolor='green', facecolor='none'))

plt.axis("off")
plt.show()
