import argparse
import socketserver
import pickle
import os
import cv2
import numpy as np
import socket
from moviepy.editor import *
from dotenv import load_dotenv
from celery import Celery
from concurrent.futures import ThreadPoolExecutor
import subprocess
import requests

load_dotenv()
TOKEN = os.getenv('TOKEN')
BROKER_URL = os.getenv('BROKER_URL')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND')
FRAMES_MULTIPLIER = int(os.getenv('FRAMES_MULTIPLIER'))
celery_app = Celery('flowy_backend', broker=BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery_app.conf.update(
    task_serializer='pickle',
    accept_content=['json','pickle'],
    broker_hearbeat=10,
)


#@celery_app.task
def process_video(video_pickle):
    print("Video Received")
    # Read pickle and save video name and the video itself in a temporal file.
    video_name, video_bytes = pickle.loads(video_pickle)
    os.makedirs('./temp', exist_ok=True)
    with open('./temp/'+video_name, 'wb') as f:
        f.write(video_bytes)
    del video_pickle, video_bytes

    chat_id = video_name.split('_')[0]

    # Prepare video for process converting it to a np.array
    chunks, video_shape, audio, original_framerate, duration = prepare_video('./temp/'+video_name)

    os.remove('./temp/'+video_name)

    # Process each chunk in a different process
    print("Generating threads")
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(process_chunk, chunks)
    

    print("Rebuilding video")
    processed_path, temp_path = rebuild_video(results, video_name, video_shape, audio, original_framerate, duration)

    #Send video
    response = send_video(processed_path, chat_id)
    print(response)
    os.remove(processed_path)
    os.remove(temp_path)


def prepare_video(video_temp_path):
    cap = cv2.VideoCapture(video_temp_path) # Open video file
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Read number of frames
    original_framerate = int(cap.get(cv2.CAP_PROP_FPS)) # Read framerate
    duration = frame_count / original_framerate # Calculate duration
    frames = [] # List to store frames

    ret = True # Boolean flag 
    while ret: # while there are frames to read
        ret, img = cap.read() # read frame
        if ret: # if the frame was read correctly
            frames.append(img) # it is added to the list

    video = np.stack(frames, axis=0) # convert the frame list into a ndarray with shape = (frame, x, y, channel)
    
    # Divide the video in 3 chunks
    chunks = np.array_split(video, 3, axis=0)

    audio = VideoFileClip(video_temp_path).audio

    return chunks, video.shape, audio, original_framerate, duration

def process_chunk(chunk):
    for i in range(FRAMES_MULTIPLIER):
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

def rebuild_video(chunks, video_name, shape, audio, original_framerate, duration):
    os.makedirs('./processed_videos', exist_ok=True)
    processed_path =  './processed_videos/'+video_name
    temp_path = './temp/'+video_name

    final_video = np.concatenate(list(chunks), axis=0)

    # Save the new video
    new_framerate = original_framerate * (FRAMES_MULTIPLIER+3)
    #breakpoint()

    fourcc =  cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(temp_path, fourcc, new_framerate, (shape[2], shape[1]), isColor=True)

    for frame in final_video:
        output_video.write(frame)

    output_video.release()
    # Add metadata
    add_metadata_to_video(temp_path, processed_path, new_framerate)
    

    # Add audio
    video = VideoFileClip(processed_path)
    video = video.set_fps(new_framerate)
    audio = audio.set_duration(video.duration)
    audio = audio.set_fps(new_framerate)
    video = video.set_audio(audio)
    os.remove(processed_path)
    video.write_videofile(processed_path)

    return processed_path, temp_path

def add_metadata_to_video(temp_path, processed_path, frame_rate):
    metadata = {'frame_rate': str(frame_rate)}

    command = ['ffmpeg', '-i', temp_path]

    for key, value in metadata.items():
        command.extend(['-metadata', f'{key}={value}'])

    command.append(processed_path)

    subprocess.run(command)

def send_video(video_path, chat_id):
    print("Sending video")
    url = f"https://api.telegram.org/bot{TOKEN}/sendVideo"
    files = {"video": open(video_path, "rb")}
    data = {"chat_id": chat_id}
    response = requests.post(url, files=files, data=data)
    return response

class FlowyHandler(socketserver.ThreadingMixIn, socketserver.BaseRequestHandler):
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
        #result = process_video.delay(video_pickle)
        #result = result.get()
        process_video(video_pickle)
        


class FlowyBackend(socketserver.TCPServer):
    
    #address_family = socket.AF_UNSPEC  # allow both IPv4 and IPv6

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(self.server_address)
            self.server_address = self.socket.getsockname()
        except:
            self.address_family = socket.AF_INET6
            self.socket = socket.socket(self.address_family,
                                        self.socket_type)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(self.server_address)
            self.server_address = self.socket.getsockname()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flowy backend')
    parser.add_argument(
        '--ip', type=str, default='127.0.0.1', help='IP address')
    parser.add_argument('-p', '--port', type=int,
                        default=8000, help='Port number')
    args = parser.parse_args()

    HOST, PORT = args.ip, args.port

    with FlowyBackend((HOST, PORT), FlowyHandler) as server:
        server.socket_type = socket.AF_INET6
        print('Server running in', HOST, 'port', PORT)
        server.serve_forever()
