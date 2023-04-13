import argparse
import socketserver
import threading
import pickle
import os

class FlowyBackend(socketserver.ThreadingMixIn, socketserver.BaseRequestHandler):
    def handle(self):
        # Recive and save the video in a temporal file
        BUFFER_SIZE = 1024 * 1024 # Cambiar por el tamaño del fragmento de datos a recibir
        video_pickle = b''
        print('Recibiendo video...')
        while True:
            chunk = self.request.recv(BUFFER_SIZE)
            if not chunk:
                break
            video_pickle += chunk
        
        # Save video
        video_name, video_data = pickle.loads(video_pickle)
        os.makedirs("./videos_for_process", exist_ok=True)
        with open("./videos_for_process/"+video_name, 'wb') as f:
            f.write(video_data)
            print('Video guardado')
            f.close()
        
        # Procesar el archivo temporal (por ejemplo, reproducir el video)
        # ... ACA HAY QUE VER COMO ENCARAR CELERY PARA QUE HAGA EL TRABAJO



class ThreadedVideoServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flowy backend')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address')
    parser.add_argument('-p', '--port', type=int, default=8000, help='Port number')
    args = parser.parse_args()

    HOST, PORT = args.ip, args.port

    with socketserver.TCPServer((HOST, PORT), FlowyBackend) as server:
        print('Server running in', HOST, 'port', PORT)
        server.serve_forever()
    
