import cv2
import argparse
import torch
import numpy as np
import warnings

def yolo_detection(source, weights):
    """
    Detección con YOLOv5 usando torch.hub.
    """
    # Cargar el modelo YOLO con autoshape para aceptar imágenes sin preprocesar
    model = torch.hub.load(
        'ultralytics/yolov5', 
        'custom', 
        path=weights, 
        trust_repo=True, 
        autoshape=True
    )
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("No se pudo abrir la fuente:", source)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar detección con YOLO
        results = model(frame)
        # Renderizar las detecciones sobre el frame
        annotated_frame = np.squeeze(results.render())
        
        cv2.imshow("Deteccion YOLO", annotated_frame)
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def face_detection(source):
    """
    Detección de rostros usando Haar Cascade de OpenCV.
    """
    # Cargar el clasificador Haar para rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error al cargar el clasificador de rostros.")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("No se pudo abrir la fuente:", source)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir el frame a escala de grises para el detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Dibujar un rectángulo alrededor de cada rostro detectado
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Deteccion de Rostros", frame)
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Aplicacion para deteccion en tiempo real")
    parser.add_argument("--source", type=str, default="0", 
                        help="Fuente de video: '0' para webcam o ruta a archivo de video/imagen")
    parser.add_argument("--mode", type=str, default="face", choices=["face", "yolo"],
                        help="Modo de deteccion: 'face' para deteccion de rostros o 'yolo' para deteccion con YOLO")
    parser.add_argument("--weights", type=str, default="yolov5s.pt",
                        help="Ruta a los pesos de YOLO (solo se usa en modo 'yolo')")
    args = parser.parse_args()
    
    # Determinar la fuente: si se ingresa '0' se interpreta como webcam
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    if args.mode == "face":
        print("Ejecutando deteccion de rostros...")
        face_detection(source)
    elif args.mode == "yolo":
        print("Ejecutando deteccion con YOLOv5...")
        yolo_detection(source, args.weights)

if __name__ == "__main__":
    # Opcional: ignorar avisos de deprecación de Torch (FutureWarnings)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
