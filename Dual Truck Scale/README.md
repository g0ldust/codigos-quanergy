Python version used: 3.14 (funciona con 3.12 o superior)

Camión en la balanza con dos sensores, uno a cada lado: cada sensor mide la distancia de las ruedas de su lado
al borde de la balanza (el margen). Entre los dos verifican que el camión tenga todas las ruedas dentro.

Maqueta a escala (la caja de 60 x 60 x 35 cm es el camión):
- Camión real 2.2 m de ancho = caja 60 cm, escala 1 : 3.67.
- La balanza real (3 m de ancho) en la maqueta mide 3.0 / 3.67 = 82 cm de ancho: marca la cinta así.
  El largo de la cinta puede ser cualquiera mayor que la caja (por ejemplo 1.2 m).
- Con la caja centrada quedan unos 11 cm de margen por lado (40 cm en el camión real).
- Los sensores van a los costados de la balanza (frente a los lados largos), mirando las ruedas.

Cómo mide:
- Cada sensor mide el lado de la balanza que tiene enfrente (se marca con una línea magenta al calibrar).
- Solo cuentan los puntos bajos (WHEEL_BAND_MM): las ruedas, no la carrocería ni la tolva, que pueden sobresalir.
- Margen = distancia en el piso desde la cara exterior de las ruedas hasta el borde. Se mide con X, Y, Z reales,
  así que el ángulo del sensor no cambia la medida.
- Ancho medido = ancho de la balanza - margen de un lado - margen del otro. Debe dar el ancho del camión
  (TRUCK_WIDTH_MM): si da otra cosa, lo medido no es el camión (una persona sobre el borde, un sensor tapado)
  y el resultado es "Medicion dudosa", sin dar dentro ni fuera.
- "RUEDA FUERA DE LA BALANZA!" si un margen es menor que MIN_MARGIN_MM o una rueda pasa un extremo de la balanza.
  En la consola se indica de qué lado (sensor 1 o 2).
- "CAMION DENTRO" si todo está bien. Si un sensor no ve el camión (tapado), se usa solo el otro, sin control de ancho.

Pasos:
- Ejecuta "dual_truck_scale.py" e ingresa el número de serie de cada sensor (o déjalos fijos en SERIALS).
- En CADA ventana, con la balanza VACÍA:
  - haz clic en las 4 esquinas de la cinta, en orden alrededor (la vista infrarroja muestra la cinta, cámbiala con 'v');
  - presiona 'f'. En la consola salen los lados medidos y el ancho: revisa que coincidan con la cinta.
    La línea magenta debe ser el lado de la balanza más cercano a ese sensor.
- Pon la caja. Se pinta en verde lo que el sensor mide como rueda y en rojo lo que pasa del borde.
  Arriba se ve el margen de ese sensor, el resultado y el ancho medido.

Teclas (actúan sobre la ventana donde está el mouse, que tiene borde amarillo):
- clic izquierdo / derecho: marcar esquina / borrar esquinas
- botón "Borrar marcas" (arriba a la derecha de cada ventana), tecla b o clic derecho: borrar las esquinas
  de ese sensor. La referencia del área vacía se conserva: al marcar las 4 esquinas nuevas mide de inmediato,
  sin presionar 'f' otra vez.
- f : guardar referencia con la balanza vacía
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
- r : girar la imagen 90° (hay que marcar las esquinas de nuevo)
- q : salir

Ajustes (al inicio del archivo). Valores de la maqueta; los del camión real están en el comentario:
- TRUCK_WIDTH_MM: ancho del camión (600 maqueta, 2200 real).
- WIDTH_TOLERANCE_MM: diferencia aceptada en el ancho medido (60 maqueta, 200 real).
- MIN_MARGIN_MM: margen mínimo por lado antes de dar alarma (0 = justo en el borde).
- WHEEL_BAND_MM: altura máxima de lo que se mide como rueda (300 maqueta, ~1100 real).
- END_TOLERANCE_MM: cuánto puede pasarse una rueda de los extremos delantero / trasero.
- MAX_OUTSIDE_MM: lo que esté más lejos que esto de la balanza se ignora (300 maqueta, ~1000 real).
- IMAGE_ROTATIONS, CAMERA_MODE, USE_GPIO, filtros: igual que en Dual Sensor Zone Detection.

Salidas GPIO (USE_GPIO = True, activas en 0): OUT1 = 0 con una rueda fuera, OUT2 = 0 con el camión dentro.
Con la balanza vacía o una medición dudosa, ambas en 1.

Notas:
- Cada sensor corre en su propio proceso (con los dos en el mismo proceso el SDK perdía la conexión).
- Un objeto más ancho que la balanza da "Medicion dudosa" (su ancho no es el del camión).
- En la balanza real (20 m de largo) un sensor por lado no alcanza a ver todas las ruedas: harían falta 2 por lado.
