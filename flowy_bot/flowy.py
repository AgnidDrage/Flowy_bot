import logging
import os
import socket
import argparse
import time
import pickle
import cv2
from dotenv import load_dotenv
from telegram import Update, Sticker
from telegram.ext import filters, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler

load_dotenv()

TOKEN = os.getenv('TOKEN')
BUFFER_SIZE = 1024 * 1024

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hi!, I'm Flowy, I can increase the framerate of your videos!! Send me a video in .mp4 and I'll send you the result.")


async def processRequest(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        await video_file.download_to_drive(video_path)
    except Exception as e:
        logging.error(e)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Error processing video.")
        # add more logic to handle errors
        return

    # Send the video to the backend
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Video uploaded for processing.\nThe video will be sent when it's done.")
    send_video(video_path=video_path, name_file=name_file)


def send_video(video_path, name_file):
    # Create pickle with the name of the video and the video itself
    with open(video_path, 'rb') as f:
        video_data = f.read()
        video_pickle = pickle.dumps((name_file, video_data))
        f.close()


    # Send the video to the backend
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.ip, args.port))
    for i in range(0, len(video_pickle), BUFFER_SIZE):
        s.send(video_pickle[i:i+BUFFER_SIZE])
    s.close()
    os.remove(video_path)


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
    application.add_handler(MessageHandler(filters.VIDEO, processRequest))
    application.run_polling()
