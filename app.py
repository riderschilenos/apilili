from flask import Flask, render_template, request, Response
import cv2
import numpy as np

app = Flask(__name__)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("⚠️ Error: No se pudo cargar el clasificador de rostros.")
else:
    print("✅ Clasificador cargado correctamente.")

@app.route('/')
def index():
    return render_template('index.html')  # Archivo HTML con el visor de la cámara

@app.route('/video_feed', methods=['POST'])
def video_feed():
    if 'frame' not in request.files:
        return "No se recibió frame", 400

    file = request.files['frame']
    np_img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Convertir a escala de grises y detectar rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print(f"Rostros detectados: {len(faces)}")  # 👈 Agregamos esto para depuración

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
