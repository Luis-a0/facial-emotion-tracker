from modules import face_detection

IMAGEN_TEST = "data/SCREEN.png"
PESOS = "model/haarcascade_frontalface_default.xml"
OUTPUT_IMAGE = "data/SCREEN_MOD.png"

if __name__ == "__main__":
    face_detection.faces_image(IMAGEN_TEST, PESOS, OUTPUT_IMAGE)