Visor de dos sensores: solo muestra la imagen de profundidad (o infrarroja) de cada sensor, sin detección,
sin GPIO y sin modo de disparo. Sirve para probar si los dos sensores pueden transmitir al mismo tiempo.

Ejecuta "dual_camera_viewer.py" e ingresa los dos números de serie (o déjalos en SERIALS).
En la consola se imprime cada paso con la hora y cuánto tardó (conectar, iniciar). Si un sensor falla,
su ventana muestra "SIN CONEXION" con el error y el otro sigue funcionando.

En cada ventana: fps, % de pixeles con lectura y la distancia al centro de la imagen.

Teclas:
- 1 / 2 : girar la imagen del sensor 1 / 2
- v : cambiar entre profundidad e infrarroja
- q : salir

Ajustes para probar (al inicio del archivo):
- CONNECTION = "gige": fuerza la conexión por red; "auto" usa USB si está disponible.
- PACKET_SIZE: tamaño de paquete de red con "gige" (prueba 1000 si se pierden cuadros).
- START_ALL_TOGETHER = False: conecta e inicia los sensores de a uno, en vez de conectar todos y luego iniciar.
