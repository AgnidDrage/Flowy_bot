FROM python:latest
RUN mkdir /flowy_bot
WORKDIR /flowy_bot
COPY ./.env /flowy_bot
COPY ./requirements.txt /flowy_bot
RUN pip install -r requirements.txt
COPY . /flowy_bot
CMD ["python", "flowy.py", "--ip", "127.0.0.1", "--port", "8000"]
