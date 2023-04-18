import argparse
import socketserver
import threading
import pickle
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from celery import Celery
from concurrent.futures import ThreadPoolExecutor

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
    print("Video Received")
    # Read pickle and save video name and the video itself in a temporal file.
    video_name, video_bytes = pickle.loads(video_pickle)
    with open(video_name, 'wb') as f:
        f.write(video_bytes)
    del video_pickle, video_bytes

    # Prepare video for process converting it to a np.array
    cap = cv2.VideoCapture(video_name) # Open video file
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Read number of frames
    framerate = int(cap.get(cv2.CAP_PROP_FPS)) # Read framerate
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

    # Process the video

    # target_framerate = video.shape[0] * 2
    # new_axis = np.linspace(0, video.shape[0]-1, target_framerate)

    # # Interpolate the frame axis
    # for i in range(video.shape[1]):
    #     for j in range(video.shape[2]):
    #         for k in range(video.shape[3]):
    #             video[:, i, j, k] = np.interp(new_axis, np.arange(video.shape[0]), video[:, i, j, k])

    # return video.shape
    
    # Divide the video in 3 chunks
    chunks = np.array_split(video, 3, axis=0)

    # Process each chunk in a different process
    print("Generating threads")
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(process_chunk, chunks)
    
    print("Rebuilding video")
    final_video = np.concatenate(list(results), axis=0)

    # Normalize the video
    final_video = (final_video / 255).astype(np.uint8)

    # Save the video
    print("Saving video")
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), framerate*2, (final_video.shape[2], final_video.shape[1]))

    for i in range(final_video.shape[0]):
        out.write(final_video[i])

    out.release()



def process_chunk(chunk):
    target_frames = chunk.shape[0] * 2
    new_frame_axis = np.linspace(0, chunk.shape[0]-1, target_frames)
    new_chunk = np.empty((target_frames, chunk.shape[1], chunk.shape[2], chunk.shape[3]))

    # Interpolate the frame axis
    for i in range(chunk.shape[1]):
        for j in range(chunk.shape[2]):
            for k in range(chunk.shape[3]):
                new_chunk[:, i, j, k] = np.interp(new_frame_axis, np.arange(chunk.shape[0]), chunk[:, i, j, k])
    return new_chunk

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
