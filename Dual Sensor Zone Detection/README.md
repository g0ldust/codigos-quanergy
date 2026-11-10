Python version used: 3.14 (funciona con 3.12 o superior)

Objeto en área delimitada con dos sensores: los dos sensores miran la misma área (marcada con cinta en el piso)
desde lados distintos y verifican si hay un objeto y si está completamente dentro del área.
Cada sensor ve una cara distinta del objeto, así que entre los dos lo cubren completo.

Cómo mide:
- Usa la profundidad (Z) de cada punto, junto con X e Y, para ubicarlo en el piso en milímetros reales.
  Así algo que está delante o detrás del área (en la línea de vista del sensor) no se confunde con algo encima.
- Un punto es "objeto" si está a más de 5 cm del piso y su profundidad cambió respecto al área vacía
  (los muebles quietos no cuentan).
- Si algún sensor ve una parte del objeto fuera del borde: "OBJETO FUERA DEL AREA!".
- Si hay objeto y ningún sensor lo ve salirse: "Objeto dentro del area".
- Si un sensor queda tapado (por ejemplo, una persona delante), el otro sigue vigilando.

Pasos:
- Ejecuta "dual_zone_detection.py" e ingresa el número de serie de cada sensor
  (o déjalos fijos en SERIALS al inicio del archivo). Si dejas vacío el segundo, funciona con uno solo.
- Se abre una ventana por sensor. En CADA ventana, con el área VACÍA:
  - haz clic en las 4 esquinas de la cinta, en orden alrededor del área (la vista infrarroja muestra la cinta,
    cámbiala con 'v');
  - presiona 'f' para guardar la referencia. En la consola se imprimen los lados medidos en mm:
    revisa que coincidan con la cinta.
- Pon el objeto. Se pinta en verde lo que está dentro y en rojo lo que sale del área.

Las teclas actúan sobre la ventana donde está el mouse (tiene borde amarillo):
- clic izquierdo : marcar esquina (4 en total)
- clic derecho   : borrar las esquinas
- botón "Borrar marcas" (arriba a la derecha de cada ventana), tecla b o clic derecho: borrar las esquinas
  de ese sensor. La referencia del área vacía se conserva: al marcar las 4 esquinas nuevas mide de inmediato,
  sin presionar 'f' otra vez.
- f : guardar referencia con el área vacía
- n : reiniciar ese sensor
- v : cambiar vista: color (si el sensor tiene cámara a color), combinada, solo distancia, solo infrarroja
      La vista color es la imagen de la cámara a color alineada con la de profundidad: ahí se ve la cinta
      para marcar las esquinas. Si no aparece, el sensor no tiene cámara a color; prueba la vista infrarroja.
      Truco si la cinta no se ve: pon un objeto (vaso, botella) en cada esquina, haz clic donde toca el piso,
      retíralos y recién entonces presiona 'f'.
- m : cambiar el modo del sensor (Outdoor Dynamic, Medium Range Static, Short Range High Speed...), igual que
      en el visor de Quanergy. Cada modo da una imagen distinta: prueba cuál muestra mejor la cinta y el piso.
      Las esquinas se conservan, pero hay que presionar 'f' de nuevo con el área vacía. El modo actual se ve
      abajo en la ventana; para dejarlo fijo, cambia CAMERA_MODE al inicio del archivo (la consola dice el número).
- r : girar la imagen 90° (borra las esquinas y la referencia: hay que marcarlas de nuevo)
- q : salir

Salidas GPIO: APAGADAS por defecto (USE_GPIO = False), porque configurarlas cortaba la conexión de red
de los sensores. Con USE_GPIO = True (activas en 0, como los otros applets) las de ambos sensores muestran
el resultado combinado, así que puedes cablear cualquiera:
- OUT1 = 0 mientras el objeto está fuera del área.
- OUT2 = 0 mientras el objeto está completamente dentro.
- Con el área vacía ambas quedan en 1.

Ajustes (al inicio del archivo):
- SERIALS: números de serie de los sensores, por ejemplo ["3030031", "3030045"]. Vacío = se preguntan al iniciar.
- IMAGE_ROTATIONS: giros de 90° de la imagen de cada sensor (0 a 3). Sensor de cabeza = 2.
- HEIGHT_THRESHOLD_MM: altura mínima sobre el piso para considerar algo "objeto" (50 mm).
- OUTSIDE_TOLERANCE_MM: cuántos mm puede pasarse del borde sin alarma (20 mm).
- MAX_OUTSIDE_MM: puntos más lejos que esto del área se ignoran (300 mm): así una persona que pasa al lado
  no se cuenta como parte del objeto. Si el objeto puede sobresalir más que eso, súbelo.
- OUTSIDE_MIN_PIXELS / MIN_OBJECT_PIXELS: puntos necesarios para "fuera" (20) y para "hay objeto" (300).
- TEMPORAL_THRESHOLD: cuadros seguidos necesarios para confirmar el cambio (5).
- SMOOTHING_FRAMES / MEDIAN_FILTER_SIZE: filtro de ruido (5 y 5), igual que en los otros programas.
- CAMERA_MODE: modo de los sensores (5 = Medium Range Static Scene).
- USE_GPIO: usar las salidas OUT1/OUT2 (False por defecto).
- RECONNECT_SECONDS: si un sensor pierde la conexión, su ventana dice "SIN CONEXION" y se reintenta cada
  este tiempo (5 s). Mientras tanto el otro sensor sigue vigilando; la zona y la referencia se conservan.

Notas:
- Cada sensor corre en su propio proceso (se ven 3 python.exe en el Administrador de tareas). Con los dos
  sensores en el mismo proceso, el SDK perdía la conexión de ambos a los 30-60 s; separados funcionan sin cortes.
  Si un sensor se corta igual, su ventana dice "SIN CONEXION" y se reconecta solo, sin perder la calibración.
- Los dos sensores se miran de frente: con el área vacía, cada ventana debería decir "Area vacia" sin parpadear.
  Si aparecen puntos verdes o rojos sin objeto, puede ser interferencia entre ellos: sepáralos un poco en
  altura o en ángulo para que no se apunten directamente.
- Una persona parada justo al lado del objeto (a menos de 30 cm) se cuenta como parte de él y puede dar "fuera".
