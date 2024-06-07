from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import csv
import firebase_admin
from firebase_admin import credentials, db
import os

app = Flask(__name__)

# Inicializar Firebase
cred = credentials.Certificate("firebase_credentials.json")
try:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://proyecto-iot-887b4-default-rtdb.firebaseio.com/'  # Actualiza esto con la URL correcta
    })
except ValueError as e:
    print(f"Error initializing Firebase: {e}")

# Ruta del archivo CSV para guardar los recuentos
csv_file = 'object_counts.csv'

# Configuración de OpenCV y detección de objetos
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)  # Tamaño de entrada mayor para mejor precisión
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

camera = cv2.VideoCapture(0)  # 0 para la primera cámara conectada

# Variable global para almacenar el conteo de objetos
object_counts = {}

def gen_frames():
    global object_counts
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Count'])  # Escribir la primera fila con los encabezados

        while True:
            success, img = camera.read()
            if not success:
                break

            classIds, confs, bbox = net.detect(img, confThreshold=0.7)  # Umbral de confianza más alto
            object_counts = {}

            if len(classIds) != 0:
                indices = cv2.dnn.NMSBoxes(bbox, confs, score_threshold=0.7, nms_threshold=0.3)  # Umbral de NMS reducido

                # Ordenar las detecciones por el área del bounding box (prioridad a los objetos más cercanos)
                if len(indices) > 0:
                    indices = indices.flatten()
                    sorted_indices = sorted(indices, key=lambda i: bbox[i][2] * bbox[i][3], reverse=True)

                    for i in sorted_indices:
                        box = bbox[i]
                        classId = classIds[i]
                        confidence = confs[i]
                        className = classNames[classId - 1]
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                        cv2.putText(img, f'{className} ({confidence:.2f})', (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                        if className not in object_counts:
                            object_counts[className] = 1
                        else:
                            object_counts[className] += 1

                    for key, value in object_counts.items():
                        writer.writerow([key, value])

                    for idx, (key, value) in enumerate(object_counts.items()):
                        cv2.putText(img, f"{key}: {value}", (10, 50 + 30 * idx), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    try:
                        db.reference('object_counts').set(object_counts)
                    except Exception as e:
                        print(f"Error sending data to Firebase: {e}")

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/object_counts')
def object_counts_endpoint():
    global object_counts
    return jsonify(object_counts)

if __name__ == '__main__':
    app.run(debug=True)