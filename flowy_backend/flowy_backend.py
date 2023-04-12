import argparse
import socketserver
import threading
import pickle

class FlowyBackend(socketserver.BaseRequestHandler):
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
        with open(video_name, 'wb') as f:
            f.write(video_data)
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

    HOST = args.ip
    PORT = args.port
    
    # Crear una instancia del servidor
    server = ThreadedVideoServer((HOST, PORT), FlowyBackend)
    
    # Poner el servidor en marcha en un hilo separado
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # Esperar a que se presione Ctrl+C para detener el servidor
    try:
        while True:
            continue
    except KeyboardInterrupt:
        server.shutdown()
