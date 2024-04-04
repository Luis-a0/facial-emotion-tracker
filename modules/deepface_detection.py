from deepface import DeepFace
import cv2
from modules.face_detection import show_marked_faces
from modules.utilidades import calcular_tiempo_ejecucion

def region_extract(list_emotions):
    region = []
    for emotion in list_emotions:
        region.append([
            emotion['region']['x'],
            emotion['region']['y'],
            emotion['region']['w'],
            emotion['region']['h'],
            ])
    return region

@calcular_tiempo_ejecucion
def detect_faces_and_emotions(image_path, output_path):
    try:
        # Analizar las emociones en los rostros detectados
        analyzed_emotions = DeepFace.analyze(image_path, actions=['emotion'])
        faces = region_extract(analyzed_emotions)
        show_marked_faces(faces, cv2.imread(image_path), output_path)

    except Exception as e:
        print("Error:", str(e))