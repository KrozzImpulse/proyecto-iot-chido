from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import csv

app = Flask(__name__)

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
net.setInputSize(320, 320)
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

            classIds, confs, bbox = net.detect(img, confThreshold=0.5)
            object_counts = {}

            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    className = classNames[classId - 1]
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                    cv2.putText(img, className, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    if className not in object_counts:
                        object_counts[className] = 1
                    else:
                        object_counts[className] += 1

                for key, value in object_counts.items():
                    writer.writerow([key, value])

                for idx, (key, value) in enumerate(object_counts.items()):
                    cv2.putText(img, f"{key}: {value}", (10, 50 + 30 * idx), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

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