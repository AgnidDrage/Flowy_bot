import argparse
import socketserver
import threading
import pickle
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from celery import Celery

load_dotenv()
BROKER_URL = os.getenv('BROKER_URL')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND')
celery_app = Celery('flowy_backend', broker=BROKER_URL, backend=CELERY_RESULT_BACKEND)
#celery_app.conf.task_serializer = 'pickle'
celery_app.conf.update(
    task_serializer='pickle',
    accept_content=['json','pickle'],
    broker_hearbeat=10,
)


@celery_app.task
def process_video(video_pickle):
    print("llego a celery el video")

def prepare_video(video_name):
    cap = cv2.VideoCapture(video_name) # abre el archivo de video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # obtiene el número de frames
    frames = [] # crea una lista vacía para guardar los frames

    ret = True # inicializa una variable booleana
    while ret: # mientras haya frames por leer
        ret, img = cap.read() # lee un frame del objeto cap; img es un np.array de dimensiones (x, y, c)
        if ret: # si se leyó correctamente el frame
            frames.append(img) # lo agrega a la lista de frames

    video = np.stack(frames, axis=0) # convierte la lista de frames en un np.array de dimensiones (frame, x, y, c)
    return video

class FlowyBackend(socketserver.ThreadingMixIn, socketserver.BaseRequestHandler):
    def handle(self):
        # Recive and save the video in a temporal file
        BUFFER_SIZE = 1024 * 1024  # Cambiar por el tamaño del fragmento de datos a recibir
        video_pickle = b''
        print('Recibiendo video...')
        while True:
            chunk = self.request.recv(BUFFER_SIZE)
            if not chunk:
                break
            video_pickle += chunk

        # Prepare video for Celery
        # print("Preparando video...")
        # video_name, video_bytes = pickle.loads(video_pickle)
        # with open(video_name, 'wb') as f:
        #     f.write(video_bytes)

        # del video_pickle, video_bytes

        # video = prepare_video(video_name)

        # Send the video pickle to Celery
        print("Enviando video a Celery...")
        result = process_video.delay(video_pickle)
        result = result.get()
        print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flowy backend')
    parser.add_argument(
        '--ip', type=str, default='127.0.0.1', help='IP address')
    parser.add_argument('-p', '--port', type=int,
                        default=8000, help='Port number')
    args = parser.parse_args()

    HOST, PORT = args.ip, args.port

    with socketserver.TCPServer((HOST, PORT), FlowyBackend) as server:
        print('Server running in', HOST, 'port', PORT)
        server.serve_forever()
