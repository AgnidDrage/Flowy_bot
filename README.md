
# Flowy Bot


Flowy is a project that allows processing videos sent by users through a Telegram bot. The processing consists of interpolating frames and increasing the framerate of the videos using image processing techniques with OpenCV and NumPy. The project uses sockets to communicate with a backend that performs the processing and returns the processed video to the user.




## Installation

To install the necessary dependencies to run the project, it is recommended to use a Python virtual environment and install the packages using the requirements.txt file:

python -m venv venv source venv/bin/activate pip install -r requirements.txt You also need to have ffmpeg installed on the operating system to add metadata and audio to the processed videos.
    
## Usage

To use the project, you must follow these steps:

Create a Telegram bot using BotFather and get the access token. Set the access token as an environment variable called TOKEN. Run the flowy.py script to start the Telegram bot. Run the flowy_backend.py script to start the server that processes the videos. Start Celery on the backend to process the videos. Send a video to the Telegram bot from the application and wait to receive the processed video.


## Author

- [@AgnidDrage](https://github.com/AgnidDrage)

