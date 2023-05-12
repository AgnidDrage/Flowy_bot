import logging
import os
import socket
import argparse
import pickle
from dotenv import load_dotenv
from telegram import Update, Sticker
from telegram.ext import filters, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler

load_dotenv()

TOKEN = os.getenv('TOKEN')
BUFFER_SIZE = int(os.getenv('BUFFER_SIZE'))

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
        This function sends a welcome message to the user and explains the bot's functionality.

        Args:
            update: An object that represents an incoming update from Telegram.
            context: An object that contains data about the chat and the bot.

        Returns:
            None
    """
    
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hi!, I'm Flowy, I can increase the framerate of your videos!! Send me a video in .mp4 (not as document) and I'll send you the result. High quality videos may generate problems. This bot is a WIP.")

async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
        This function sends a help message to the user and reminds them of the bot's functionality.

        Args:
            update: An object that represents an incoming update from Telegram.
            context: An object that contains data about the chat and the bot.

        Returns:
            None
    """
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Send me a video in .mp4 (not as document) and I'll send you the result. High quality videos may generate problems. This bot is a WIP.")


async def processRequest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
        This function processes a video request from a user using the Telegram API.

        It performs the following steps:
        - Gets the video from the update object and checks if its size is less than 20 MB.
        - Sends a message to the user and downloads the video to a local path using the context object.
        - Sends the video to the backend using the send_video function.

        Args:
            update: The update object that contains the user message and the video.
            context: The context object that provides access to the bot and its methods.
    """
    # Get video
    chat_id = update.effective_chat.id
    video = update.message.video

    # Check if the video is too big
    if video.file_size > 20 * 1024 * 1024:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="The video is too big. The maximum size is 20MB.")
        return

    # Send a message to the user and prepare for download the video
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Uploading...")
    video_file = await context.bot.get_file(video.file_id)
    name_file = f"{chat_id}_{video.file_unique_id}.{video.mime_type.split('/')[1]}"
    os.makedirs("./videos", exist_ok=True)
    video_path = "./videos/"+name_file

    # Download the video
    try:
        logging.info("Downloading video")
        await video_file.download_to_drive(video_path)
    except Exception as e:
        logging.error(e)
        raise Exception("Error downloading video")

    # Send the video to the backend
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Video uploaded for processing.\nThe video will be sent when it's done.")
    send_video(video_path=video_path, name_file=name_file)


def send_video(video_path: str, name_file: str):
    """
        This function sends a video file to the backend using a socket connection.

        It performs the following steps:
        - Creates a pickle with the name of the video and the video data by reading the file from the given path.
        - Tries to create a socket object with IPv6 protocol and connects to the backend server using the given ip and port arguments.
        - If the IPv6 connection fails, creates a socket object with IPv4 protocol and connects to the backend server.
        - Sends the pickle data in chunks of BUFFER_SIZE bytes using the socket object.
        - Closes the socket object and deletes the video file from the local path.

        Args:
            video_path: The path of the video file to send.
            name_file: The name of the video file.
    """

    # Create pickle with the name of the video and the video itself
    with open(video_path, 'rb') as f:
        video_data = f.read()
        video_pickle = pickle.dumps((name_file, video_data))
        f.close()
    # Send the video to the backend
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.connect((args.ip, args.port))
        logging.info("Connecting to backend")
    except:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((args.ip, args.port))
            logging.info("Connecting to backend")
        except:
            raise Exception("Error connecting to backend")
    for i in range(0, len(video_pickle), BUFFER_SIZE):
        s.send(video_pickle[i:i+BUFFER_SIZE])
    s.close()
    os.remove(video_path)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
        This function handles errors that occur in the bot.

        It performs the following steps:
        - Logs the error message and the traceback using the logging module.
        - Sends an error message to the user using the send_error function.

        Args:
            update: The update object that contains the user message and the video.
            context: The context object that provides access to the bot and its methods.
    """
    logging.error(context.error)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Error processing video. Please try again.")

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Flowy Bot')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='IP to send the video to')
    parser.add_argument('--port', '-p', type=int,
                        default=8000, help='Port to send the video to')
    args = parser.parse_args()

    # Create and run the bot
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', start))
    application.add_handler(MessageHandler(filters.VIDEO, processRequest))
    application.add_error_handler(error_handler)
    application.run_polling()
