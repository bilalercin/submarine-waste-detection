from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 22 sınıf terminal/nonterminal 
TERMINAL_CLASSES = ['animal_fish', 'animal_starfish', 'animal_shells', 'animal_crab', 'animal_eel', 'animal_etc']
ALL_CLASSES = ['rov', 'plant', 'animal_fish', 'animal_starfish', 'animal_shells', 'animal_crab', 'animal_eel', 'animal_etc',
        'trash_clothing', 'trash_pipe', 'trash_bottle', 'trash_bag', 'trash_snack_wrapper', 'trash_can',
        'trash_cup', 'trash_container', 'trash_unknown_instance', 'trash_branch', 'trash_wreckage',
        'trash_tarp', 'trash_rope', 'trash_net']
CLASS_NAMES = [
    ('terminal_' if c in TERMINAL_CLASSES else 'nonterminal_') + c for c in ALL_CLASSES
]

model = YOLO('/Users/bilalercin/userai/runs/trashcan/yolov8_trashcan2/weights/best.pt')


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def draw_gt_boxes(image_path, label_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    if not os.path.exists(label_path):
        return image, []
    gt_boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, bw, bh = map(float, parts)
            x1 = int((x_center - bw/2) * w)
            y1 = int((y_center - bh/2) * h)
            x2 = int((x_center + bw/2) * w)
            y2 = int((y_center + bh/2) * h)
            gt_boxes.append({
                'class': CLASS_NAMES[int(class_id)],
                'bbox': [x1, y1, x2, y2]
            })
            # Kırmızı kutu (gerçek)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(image, f"GT: {CLASS_NAMES[int(class_id)]}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return image, gt_boxes

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Label dosyasını bul
        label_name = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join('dataset/instance_version/train/labels', label_name)
        # Gerçek kutuları çiz
        gt_img, gt_boxes = draw_gt_boxes(filepath, label_path)
        gt_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'gt_{filename}')
        cv2.imwrite(gt_img_path, gt_img)
        # Model tahmini
        results = model(filepath)
        pred_img = results[0].plot()
        pred_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'pred_{filename}')
        cv2.imwrite(pred_img_path, pred_img)
        # Tahmin kutuları
        pred_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else f'class_{cls}'
            pred_boxes.append({
                'class': class_name,
                'confidence': f'{conf_score:.2f}',
                'bbox': [x1, y1, x2, y2]
            })
        return jsonify({
            'success': True,
            'gt_image': f'/static/uploads/gt_{filename}',
            'pred_image': f'/static/uploads/pred_{filename}',
            'gt_boxes': gt_boxes,
            'pred_boxes': pred_boxes
        })
    return jsonify({'error': 'Geçersiz dosya formatı'})

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
