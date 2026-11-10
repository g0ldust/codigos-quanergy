Python version used: 3.14 (funciona con 3.12 o superior)

Posición del camión en la balanza con dos sensores, enseñando la posición correcta (sin marcar la cinta):
cada sensor mide su distancia Z a la caja. Con la caja bien ubicada se guardan esas distancias (por ejemplo
80 y 78 cm). Si la caja se mueve hacia un sensor, su Z baja y la del otro sube; si se mueve más que la
tolerancia, está fuera de la balanza.

Maqueta a escala (la caja de 60 x 60 x 35 cm es el camión de 2.2 m, escala 1 : 3.67; balanza de 82 cm de ancho):
- Tolerancia = (ancho balanza - ancho camión) / 2 = (82 - 60) / 2 = 11 cm (en el camión real, 40 cm).
- Los sensores van uno a cada lado, mirándose, con la caja entre ellos (como en el boceto).

Pasos:
1. Ejecuta "dual_truck_position.py" e ingresa el número de serie de cada sensor (o déjalos fijos en SERIALS).
2. En CADA ventana, arrastra con el mouse un rectángulo alrededor de donde puede estar la caja
   (más grande que la caja, que alcance a cubrirla aunque se corra). No hace falta ver la cinta.
3. Con la balanza VACÍA presiona 'f' (se guarda en todos los sensores a la vez).
4. Pon la caja BIEN UBICADA y presiona 't' (todos los sensores guardan su distancia a la caja).
5. Listo. Cada ventana muestra:
   - "Z = 79.0 cm (posicion 80.0 cm, cambio -1.0 cm)": la distancia actual a la caja y cuánto cambió.
   - El resultado: DENTRO DE LA BALANZA / FUERA DE LA BALANZA! / Medicion dudosa / Balanza vacia.
   - "corrida 1.0 cm hacia S2 (tolerancia 11 cm)": cuánto se movió la caja y hacia qué sensor.
   En verde se pinta la caja y en magenta la cara que el sensor mide (la más cercana a él).

Cómo decide:
- Corrimiento = (cambio sensor 1 - cambio sensor 2) / 2. Si pasa la tolerancia: FUERA.
- Con dos sensores enfrentados, cuando la caja se mueve una distancia baja lo mismo que la otra sube.
  Si no es así (por ejemplo, una persona entre un sensor y la caja), da "Medicion dudosa".
- Si un sensor no ve la caja (tapado), decide solo con el otro.
- Un sensor que mira hacia abajo en diagonal mide la distancia sobre el piso (FLOOR_CORRECTION): así la caja que
  se corre 10 cm da 10 cm en los dos sensores. En la consola se ve el ángulo detectado de cada sensor.
- Moverse a lo largo de la balanza (no hacia un sensor) no cambia las distancias: no se cuenta como fuera.

Teclas:
- arrastrar con el mouse: rectángulo donde puede estar la caja
- botón "Borrar marcas" (arriba a la derecha), tecla b o clic derecho: borrar el rectángulo de ese sensor
  (se conserva la balanza vacía; con el rectángulo nuevo, presiona 't' de nuevo)
- f : guardar la balanza vacía (todos los sensores)
- t : enseñar la posición correcta con la caja bien ubicada (todos los sensores)
- v : cambiar vista: color (si el sensor la tiene), combinada, distancia, infrarroja
- m : cambiar el modo del sensor (hay que repetir 'f' y 't'; para dejarlo fijo, CAMERA_MODE)
- n : reiniciar ese sensor
- r : girar la imagen 90° (hay que repetir todo en ese sensor)
- q : salir
v, m, b, n y r actúan sobre la ventana donde está el mouse (borde amarillo).

Ajustes (al inicio del archivo; valores de la maqueta, los del camión real en el comentario):
- TOLERANCE_MM: cuánto puede correrse el camión desde la posición enseñada (110 maqueta, 400 real).
- CONSISTENCY_MM: diferencia aceptada entre lo que baja un sensor y lo que sube el otro (60 maqueta, 200 real).
- CHANGE_THRESHOLD_MM: cuánto debe cambiar un punto respecto a la balanza vacía para ser "caja" (50).
- MIN_OBJECT_PIXELS: puntos necesarios para considerar que el sensor ve la caja (150).
- FLOOR_CORRECTION: medir sobre el piso (True) o en el eje Z del sensor tal cual (False).
- IMAGE_ROTATIONS, CAMERA_MODE, USE_GPIO, filtros: igual que en los otros programas de dos sensores.

Salidas GPIO (USE_GPIO = True, activas en 0): OUT1 = 0 con el camión fuera, OUT2 = 0 con el camión dentro.

Notas:
- Cada sensor corre en su propio proceso (con los dos en el mismo proceso el SDK perdía la conexión).
- La posición enseñada debe ser la correcta (caja centrada en la balanza): todo se mide respecto a ella.
