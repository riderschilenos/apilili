from flask import Flask, render_template, Response
import cv2

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir la cámara")
except Exception as e:
    print(f"⚠️ Advertencia: {e}")
    cap = None  # No uses la cámara si hay error

app = Flask(__name__)

# Inicializar el clasificador de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Abrir la cámara (webcam)
camera = cv2.VideoCapture(0)

def generate_frames():
    if camera is None:
        print("⚠️ No hay cámara disponible. No se enviarán frames.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    # Renderiza la página principal que contiene la imagen del stream
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Ruta que sirve el stream de video
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
