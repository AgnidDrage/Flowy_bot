Video Processor\n\n
Video Processor es un proyecto que permite procesar videos enviados por los usuarios a través de un bot de Telegram. El procesamiento consiste en interpolar frames y aumentar el framerate de los videos usando técnicas de procesamiento de imágenes con OpenCV y NumPy. El proyecto usa sockets para comunicarse con un backend que realiza el procesamiento y devuelve el video procesado al usuario.

Instalación
Para instalar las dependencias necesarias para ejecutar el proyecto, se recomienda usar un entorno virtual de Python e instalar los paquetes usando el archivo requirements.txt:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
También se necesita tener instalado ffmpeg en el sistema operativo para añadir metadatos y audio a los videos procesados.

Uso
Para usar el proyecto, se deben seguir los siguientes pasos:

Crear un bot de Telegram usando BotFather y obtener el token de acceso.
Establecer el token de acceso como una variable de entorno llamada TOKEN.
Ejecutar el script bot.py para iniciar el bot de Telegram.
Ejecutar el script backend.py para iniciar el servidor que procesa los videos.
Enviar un video al bot de Telegram desde la aplicación y esperar a recibir el video procesado.
