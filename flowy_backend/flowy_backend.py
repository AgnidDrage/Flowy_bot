import argparse
import logging
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
celery_app = Celery('flowy_backend', broker=BROKER_URL,
                    backend=CELERY_RESULT_BACKEND)
celery_app.conf.update(
    task_serializer='pickle',
    accept_content=['json', 'pickle'],
    broker_hearbeat=10,
)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


@celery_app.task
def process_video(video_pickle):
    """
        Processes a video received as a pickle and sends it to a chat.

        Reads the pickle and saves the video name and the video itself in a temporary file.
        Prepares the video for processing by converting it to a numpy array.
        Processes each chunk of the video in a different process using a ThreadPoolExecutor.
        Rebuilds the processed video and saves it in another temporary file.
        Sends the processed video to the corresponding chat using the send_video function.
        Deletes the temporary files created.

        :param video_pickle: The video encoded as a pickle that contains the video name and the video bytes.
        :type video_pickle: bytes
        :return: The response obtained when sending the processed video to the chat.
        :rtype: str
        :raises: Exception if any error occurs during the processing or sending of the video.
    """

    print("Video Received")
    # Read pickle and save video name and the video itself in a temporal file.
    video_name, video_bytes = pickle.loads(video_pickle)
    os.makedirs('./temp', exist_ok=True)
    with open('./temp/'+video_name, 'wb') as f:
        f.write(video_bytes)

    del video_pickle, video_bytes

    chat_id = video_name.split('_')[0]

    try:

        # Prepare video for process converting it to a np.array
        chunks, video_shape, audio, original_framerate, duration = prepare_video(
            './temp/'+video_name)

        os.remove('./temp/'+video_name)

        # Process each chunk in a different process
        print("Generating threads")
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_chunk, chunks)

        print("Rebuilding video")
        processed_path, temp_path = rebuild_video(
            results, video_name, video_shape, audio, original_framerate, duration)

        # Send video
        response = send_video(processed_path, chat_id)
        print(response)
        os.remove(processed_path)
        os.remove(temp_path)

    except:
        print("FATAL ERROR")
        if os.path.exists('./temp/'+video_name):
            os.remove('./temp/'+video_name)
        if os.path.exists('./processed_videos/'+video_name):
            os.remove('./processed_videos/'+video_name)
        for file in os.listdir('.'):
            if file.endswith('.mp3'):
                os.remove(file)
        handle_error(chat_id)


def prepare_video(video_temp_path):
    """
        Prepares a video for processing by converting it to a numpy array and dividing it into chunks.

        Opens the video file from the temporary path and reads the number of frames, the framerate and the duration.
        Stores each frame of the video in a list and converts the list into a numpy array with shape (frame, x, y, channel).
        Divides the video array into 4 chunks along the frame axis using np.array_split.
        Extracts the audio from the video file using VideoFileClip.

        :param video_temp_path: The path to the temp file
        :type video_temp_path: str
        :return: A tuple containing the chunks, the video shape, the audio, the original framerate and the duration.
        :rtype: tuple
    """

    cap = cv2.VideoCapture(video_temp_path)  # Open video file
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)
                      )  # Read number of frames
    original_framerate = int(cap.get(cv2.CAP_PROP_FPS))  # Read framerate
    duration = frame_count / original_framerate  # Calculate duration
    frames = []  # List to store frames

    ret = True  # Boolean flag
    while ret:  # while there are frames to read
        ret, img = cap.read()  # read frame
        if ret:  # if the frame was read correctly
            frames.append(img)  # it is added to the list

    # convert the frame list into a ndarray with shape = (frame, x, y, channel)
    video = np.stack(frames, axis=0)

    # Divide the video in 3 chunks
    chunks = np.array_split(video, 4, axis=0)
    audio = VideoFileClip(video_temp_path).audio

    return chunks, video.shape, audio, original_framerate, duration


def process_chunk(chunk):
    """
        Processes a chunk of a video by interpolating frames and increasing the framerate.

        For each iteration of the FRAMES_MULTIPLIER constant, the function doubles the number of frames in the chunk by interpolating new frames between the original ones using a linear combination with an alpha parameter. The last frame of the chunk is not interpolated.

        :param chunk: A chunk of a video as a numpy array with shape (frame, x, y, channel).
        :type chunk: np.ndarray
        :return: The processed chunk with more frames and higher framerate.
        :rtype: np.ndarray
    """

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


def rebuild_video(chunks, video_name, shape, audio, original_framerate):
    """
        Rebuilds a video from processed chunks and adds metadata and audio.

        Concatenates the chunks into a single numpy array representing the final video.
        Saves the final video in a temporary file using cv2.VideoWriter with a new framerate calculated from the original framerate and the FRAMES_MULTIPLIER constant.
        Adds metadata to the video using the add_metadata_to_video function and saves it in a processed file.
        Adds audio to the video using VideoFileClip and sets the duration and fps to match the video.
        Deletes the temporary file and returns the paths of the processed file and the temporary file.

        :param chunks: A list of processed chunks of a video as numpy arrays.
        :type chunks: list
        :param video_name: The name of the video file.
        :type video_name: str
        :param shape: The shape of the original video as a tuple (frame, x, y, channel).
        :type shape: tuple
        :param audio: The audio of the original video as a VideoFileClip object.
        :type audio: VideoFileClip
        :param original_framerate: The framerate of the original video in fps.
        :type original_framerate: int
        :return: A tuple containing the paths of the processed file and the temporary file.
        :rtype: tuple
    """
    os.makedirs('./processed_videos', exist_ok=True)
    processed_path = './processed_videos/'+video_name
    temp_path = './temp/'+video_name

    final_video = np.concatenate(list(chunks), axis=0)

    # Save the new video
    new_framerate = original_framerate * (FRAMES_MULTIPLIER+3)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(
        temp_path, fourcc, new_framerate, (shape[2], shape[1]), isColor=True)

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
    """
        Adds metadata to a video file using ffmpeg.

        Creates a dictionary with the metadata to add, in this case the frame rate.
        Builds a command list with the ffmpeg executable, the input file, the metadata key-value pairs and the output file.
        Runs the command using subprocess.run.

        :param temp_path: The path of the input video file.
        :type temp_path: str
        :param processed_path: The path of the output video file.
        :type processed_path: str
        :param frame_rate: The frame rate of the video in fps.
        :type frame_rate: int
    """
    metadata = {'frame_rate': str(frame_rate)}

    command = ['ffmpeg', '-i', temp_path]

    for key, value in metadata.items():
        command.extend(['-metadata', f'{key}={value}'])

    command.append(processed_path)

    subprocess.run(command)


def send_video(video_path, chat_id):
    """
        Sends a video file to a chat using the Telegram API.

        Opens the video file from the given path and creates a dictionary with the chat id.
        Makes a POST request to the Telegram API endpoint with the video file and the chat id as parameters.
        Returns the response object from the request.

        :param video_path: The path of the video file to send.
        :type video_path: str
        :param chat_id: The id of the chat to send the video to.
        :type chat_id: str
        :return: The response object from the request.
        :rtype: requests.Response
    """
    print("Sending video")
    url = f"https://api.telegram.org/bot{TOKEN}/sendVideo"
    files = {"video": open(video_path, "rb")}
    data = {"chat_id": chat_id}
    response = requests.post(url, files=files, data=data)
    return response


def handle_error(chat_id):
    """
        Sends an error message to a chat using the Telegram API.

        Creates a dictionary with the chat id and the error text.
        Makes a POST request to the Telegram API endpoint with the dictionary as data parameter.
        Returns the response object from the request.

        :param chat_id: The id of the chat to send the error message to.
        :type chat_id: str
        :return: The response object from the request.
        :rtype: requests.Response
    """
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": chat_id, "text": "An error ocurred, please try again later"}
    response = requests.post(url, data=data)
    return response


class FlowyHandler(socketserver.ThreadingMixIn, socketserver.BaseRequestHandler):
    def handle(self):
        # Recive and save the video in a temporal file
        BUFFER_SIZE = 1024 * 1024  # Cambiar por el tamaño del fragmento de datos a recibir
        video_pickle = b''
        logging.info("Recibiendo video...")
        while True:
            chunk = self.request.recv(BUFFER_SIZE)
            if not chunk:
                break
            video_pickle += chunk

        logging.info("Video recibido, procesando...")
        status = process_video.delay(video_pickle)
        logging.info("Esperando nuevo video...")


class FlowyBackend(socketserver.TCPServer):

    # address_family = socket.AF_UNSPEC  # allow both IPv4 and IPv6

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
