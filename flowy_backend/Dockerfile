FROM python:latest
RUN mkdir /flowy_backend
WORKDIR /flowy_backend
COPY ./.env /flowy_backend
COPY ./requirements.txt /flowy_backend
COPY . /flowy_backend
RUN apt-get update
RUN apt install -y ffmpeg
RUN pip install -r requirements.txt
RUN celery -A flowy_backend worker --detach
CMD ["python", "flowy_backend.py", "--ip", "0.0.0.0", "--port", "8000"]
