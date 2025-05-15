from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torchvision.transforms as transforms

# إعداد Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# تحميل النموذج
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("fer_model.pth", map_location=device)
model.eval()

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# تحويل الصور
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# دالة التنبؤ
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="لم يتم رفع أي صورة")
        file = request.files["file"]
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        prediction = predict_image(path)
        image_url = url_for('static', filename=f"uploads/{filename}")
        return render_template("index.html", prediction=prediction, image_url=image_url)

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    prediction = predict_image(path)
    return jsonify({"prediction": prediction})

# تشغيل التطبيق
if __name__ == "__main__":
    app.run(debug=True)