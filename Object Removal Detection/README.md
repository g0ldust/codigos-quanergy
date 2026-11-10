Python version used: 3.12

Detección de objeto retirado: el sensor memoriza la profundidad de un área con el objeto en su lugar
y avisa cuando el objeto sale de ahí (y cuando vuelve).

Pasos:
- Ejecuta "object_removal_detection.py" e ingresa el número de serie del sensor.
- Con el mouse, arrastra un rectángulo AJUSTADO alrededor del objeto (que el objeto ocupe la mayor parte del rectángulo).
- Con el objeto en su lugar y quieto, presiona 'f' para guardar la referencia (toma 10 cuadros).
- El rectángulo se pone verde: "Objeto en su lugar".
- Si retiras el objeto, el rectángulo se pone rojo: "OBJETO RETIRADO!", se imprime la hora en la consola
  y la salida GPIO OUT1 baja a 0. Cuando el objeto vuelve, OUT1 regresa a 1.

Teclas (con la ventana "Frame" seleccionada):
- f : guardar referencia con el objeto en su lugar
- n : borrar el área y elegir otra
- + / - : subir / bajar el umbral (qué % del área debe cambiar para considerar el objeto retirado, por defecto 30%)
- r : girar la imagen 90° (borra el área y la referencia: hay que elegirlas de nuevo)
- v : cambiar la vista: combinada (colores de distancia + detalle de la imagen infrarroja, por defecto),
      solo distancia, o solo imagen infrarroja (en gris, como una cámara normal). La detección no cambia.
- c : girar 90° la imagen de la cámara a color (ventana "Camara")

Cámara a color: si el sensor la tiene (modelos F540-W-C), se abre la ventana "Camara" con la imagen a color.
Es solo para ver; la detección usa la imagen 3D. Se puede achicar o agrandar con el mouse.
Para no mostrarla, pon SHOW_COLOR_CAMERA = False. COLOR_ROTATION deja fijo el giro de esa imagen.
- q : salir

Ajustes (al inicio del archivo):
- DIFF_THRESHOLD_MM: cuántos mm debe cambiar un pixel para contar como cambio (50 mm por defecto).
  Para objetos muy delgados (menos de 5 cm de alto) bájalo, por ejemplo a 20.
- TEMPORAL_THRESHOLD: cuadros seguidos necesarios para confirmar el cambio (5 por defecto), evita falsas alarmas.
- CAMERA_MODE: modo del sensor (5 = Medium Range Static Scene).
- IMAGE_ROTATION: giros de 90° de la imagen al iniciar (0 a 3). Sensor de cabeza = 2.
- SMOOTHING_FRAMES: cuadros promediados por pixel para reducir el parpadeo (5). Más = imagen más estable
  pero reacciona más lento; 1 = apagado.
- MEDIAN_FILTER_SIZE: filtro para quitar pixeles sueltos con ruido (5; 3 = más suave, 0 = apagado).
- Al iniciar se imprimen en la consola los modos del sensor: si la imagen tiene mucho ruido, prueba otro CAMERA_MODE.
- En pantalla, negro = pixeles sin lectura (muy lejos, superficies negras o brillantes).

Consejos:
- En pantalla se ve "Cambio: X%". Con el objeto en su lugar debe estar bajo (menos de 10%); al retirarlo debe superar el umbral.
  Si no es así, ajusta el rectángulo o el umbral con + / -.
- Ponerse entre el sensor y el objeto también cambia la profundidad y se detecta como "retirado".
