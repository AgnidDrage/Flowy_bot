import argparse
import socketserver
import pickle
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from celery import Celery
from concurrent.futures import ThreadPoolExecutor
import threading
import matplotlib.pyplot as plt
import requests

load_dotenv()
TOKEN = os.getenv('TOKEN')
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
    print("Video Received")
    # Read pickle and save video name and the video itself in a temporal file.
    video_name, video_bytes = pickle.loads(video_pickle)
    with open(video_name, 'wb') as f:
        f.write(video_bytes)
    del video_pickle, video_bytes

    chat_id = video_name.split('_')[0]

    # Prepare video for process converting it to a np.array
    cap = cv2.VideoCapture(video_name) # Open video file
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Read number of frames
    original_framerate = int(cap.get(cv2.CAP_PROP_FPS)) # Read framerate
    print(frame_count)
    frames = [] # List to store frames

    ret = True # Boolean flag 
    while ret: # while there are frames to read
        ret, img = cap.read() # read frame
        if ret: # if the frame was read correctly
            frames.append(img) # it is added to the list

    video = np.stack(frames, axis=0) # convert the frame list into a ndarray with shape = (frame, x, y, channel)
    print(video.shape)
    os.remove(video_name)
    
    # Divide the video in 3 chunks
    chunks = np.array_split(video, 3, axis=0)

    # Process each chunk in a different process
    print("Generating threads")
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(process_chunk, chunks)
    

    print("Rebuilding video")
    final_video = np.concatenate(list(results), axis=0)
    print(final_video.shape)
    print(final_video.min(), final_video.max()) 

    # Save the new video
    new_frame_rate = 60 * 2

    fourcc =  cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs('./processed_videos', exist_ok=True)
    output_video = cv2.VideoWriter('./processed_videos/'+video_name, fourcc, new_frame_rate, (video.shape[2], video.shape[1]), isColor=True)

    for frame in final_video:
        output_video.write(frame)

    output_video.release()

    processed_path = './processed_videos/'+video_name

    #Send video
    print("Sending video")
    url = f"https://api.telegram.org/bot{TOKEN}/sendVideo"
    files = {"video": open(processed_path, "rb")}
    data = {"chat_id": chat_id}
    response = requests.post(url, files=files, data=data)
    print(response)

    # # Send the video using the file ID
    # url = f"https://api.telegram.org/bot{TOKEN}/sendVideo"
    # data = {"chat_id": chat_id, "video": file_id}
    # response = requests.post(url, data=data)

def process_chunk(chunk):
    for i in range(2):
        frames, x, y, channel = chunk.shape
        alpha = 0.5

        # Create a new array for save the new chunk
        new_chunk = np.zeros((frames*2-1, x, y, channel), dtype=chunk.dtype)

        # Copy the original frames
        new_chunk[::2] = chunk

        # Interpolate the frames and normalize new frame
        for i in range(frames-1):
            new_frame = (1 - alpha) * chunk[i] + alpha * chunk[i+1]
            new_chunk[i*2+1] = new_frame

        # Interpolate the last frame
        new_chunk[-1] = chunk[-1]

        chunk = new_chunk
    

    return chunk

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
