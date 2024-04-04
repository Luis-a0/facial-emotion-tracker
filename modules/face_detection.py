import cv2
from pathlib import Path
from modules.utilidades import calcular_tiempo_ejecucion
# import numpy as np

def file_exist(dir_file):
    return Path(dir_file).is_file()

def show_marked_faces(faces, image, output_path):
    marked_image = image.copy()
    for x, y, w, h in faces:
        cv2.rectangle(marked_image,
                      (x,y),
                      (x+w, y+h),
                      (0, 255, 0),
                      2)
    cv2.imwrite(output_path, marked_image)

@calcular_tiempo_ejecucion
def faces_image(imagen, pesos, output_path):
    if(file_exist(imagen) and file_exist(pesos)):
        face_clasif = cv2.CascadeClassifier(pesos)
        image = cv2.imread(imagen)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_clasif.detectMultiScale(
            gray,
            scaleFactor=1.06,
            minNeighbors=10,
            minSize=(30, 30),
            maxSize=(200,200)
        )
        show_marked_faces(faces, image, output_path)
    else:
        raise FileNotFoundError("El/Los archivo/s no existe/n")