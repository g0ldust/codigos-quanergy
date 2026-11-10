# Códigos Quanergy

Programas en Python para los sensores 3D ToF Quanergy Q-Vision F540 (SDK `chronoptics.tof`).
Cada carpeta es un programa independiente, con su propio README (cómo se usa, teclas y ajustes)
y su `requirements.txt`.

## Programas

### Un sensor

| Carpeta | Qué hace |
|---|---|
| [Box Filling](Box%20Filling) | Detecta una caja abierta, mide sus dimensiones, su ángulo y el porcentaje de llenado. |
| [Intrusion Detection](Intrusion%20Detection) | Detecta intrusiones en un área y activa una salida GPIO. |
| [Object Removal Detection](Object%20Removal%20Detection) | Avisa cuando un objeto se retira de su lugar (y cuando vuelve). |
| [Truck Scale Detection](Truck%20Scale%20Detection) | Maqueta de balanza de camiones: verifica que la caja (el camión) quede dentro de la zona marcada. |

### Dos sensores

| Carpeta | Qué hace |
|---|---|
| [Dual Camera Viewer](Dual%20Camera%20Viewer) | Solo muestra la imagen de los dos sensores; sirve para probar la conexión. |
| [Dual Sensor Zone Detection](Dual%20Sensor%20Zone%20Detection) | Dos sensores vigilan un área marcada y avisan si el objeto está dentro o se sale. |
| [Dual Truck Scale](Dual%20Truck%20Scale) | Balanza: cada sensor mide el margen entre las ruedas de su lado y el borde de la balanza. |
| [Dual Truck Position](Dual%20Truck%20Position) | Balanza sin marcar la cinta: se enseña la posición correcta y cada sensor vigila su distancia Z al camión. |

## Requisitos

- Python 3.12 o superior.
- El SDK de los sensores instalado (`C:\Program Files\tof`), con estas variables de entorno:
  - `PYTHONPATH` = `C:\Program Files\tof\lib\python`
  - `TOF_DIR` = `C:\Program Files\tof\bin`
- Las librerías de cada programa: `pip install -r requirements.txt` dentro de su carpeta.

## Notas

- Los programas de dos sensores abren cada sensor en su propio proceso: con los dos en el mismo
  proceso el SDK perdía la conexión de ambos a los 30-60 s.
- `Dual Sensor Zone Detection/configurar_red_sensores.ps1` ajusta la tarjeta de red del PC
  (búferes de recepción, ahorro de energía). Se ejecuta como administrador.
