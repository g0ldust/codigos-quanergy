Python version used: 3.12

Camión sobre balanza: el sensor (mirando desde arriba) verifica que el camión esté completamente dentro de la balanza.
Si alguna parte del camión queda fuera del borde, avisa en pantalla y por GPIO.

Maqueta en la oficina (escala 1:7 de una balanza de 20 x 3 m):
- Zona con cinta en el piso: 80 x 42 cm (la "balanza").
- Caja de 60 x 35 cm: el "camión". Margen de 3,5 cm a cada lado y 10 cm en cada extremo.
- Sensor a 2,15 m del piso, idealmente mirando hacia abajo sobre el centro de la zona
  (puede estar inclinado: el programa calcula el plano del piso).

Pasos:
- Ejecuta "truck_scale_detection.py" e ingresa el número de serie del sensor.
- Con la balanza VACÍA, haz clic en las 4 esquinas de la cinta, en orden alrededor de la zona
  (la vista de intensidad muestra la cinta; con 'v' cambias a la vista de profundidad).
- Presiona 'f' para guardar la referencia (toma 10 cuadros). En la consola se imprimen los lados medidos en mm:
  revisa que coincidan con la cinta (800 y 420 mm).
- Pon la caja. Se pinta en verde y aparece "Camion dentro: pesaje valido".
- Si una parte sale del borde se pinta en rojo: "CAMION FUERA DE LA BALANZA!", con cuántos mm sale.

Salidas GPIO (activas en 0, como los otros applets):
- OUT1 = 0 mientras el camión está fuera de la balanza.
- OUT2 = 0 mientras el camión está completamente dentro (pesaje válido).
- Con la balanza vacía ambas quedan en 1.

Teclas (con la ventana "Frame" seleccionada):
- clic izquierdo : marcar esquina (4 en total)
- clic derecho   : borrar las esquinas
- f : guardar referencia con la balanza vacía
- n : reiniciar todo
- v : cambiar vista intensidad / profundidad
- r : girar la imagen 90° (borra las esquinas y la referencia: hay que marcarlas de nuevo)
- q : salir

Ajustes (al inicio del archivo):
- HEIGHT_THRESHOLD_MM: altura mínima sobre el piso para considerar algo "camión" (50 mm).
- OUTSIDE_TOLERANCE_MM: cuántos mm puede pasarse del borde sin alarma (10 mm). Para un camión real, usar ~30.
- OUTSIDE_MIN_PIXELS: puntos fuera necesarios para dar la alarma (20), evita alarmas por ruido.
- TEMPORAL_THRESHOLD: cuadros seguidos necesarios para confirmar el cambio (5).
- CAMERA_MODE: modo del sensor (5 = Medium Range Static Scene).
- IMAGE_ROTATION: giros de 90° de la imagen al iniciar (0 a 3). Sensor de cabeza = 2.
- SMOOTHING_FRAMES: cuadros promediados por pixel para reducir el parpadeo (5). Más = imagen más estable
  pero reacciona más lento; 1 = apagado.
- MEDIAN_FILTER_SIZE: filtro para quitar pixeles sueltos con ruido (5; 3 = más suave, 0 = apagado).
- En la vista de profundidad, negro = pixeles sin lectura (muy lejos, superficies negras o brillantes).

Notas:
- Usa las coordenadas X, Y, Z reales del sensor: la parte alta de la caja no "parece" más afuera por perspectiva.
- Solo cuenta lo que toca la balanza: una persona caminando por otro lado se ignora,
  pero una persona pegada a la caja se suma al "camión".
- Si la vista de intensidad no aparece, el SDK no tiene FrameType.INTENSITY con ese nombre: se muestra la profundidad.
