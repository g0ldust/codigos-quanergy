import time
import traceback
import cv2
import numpy as np
from datetime import datetime

SERIALS = []                 # Serial numbers of the sensors, e.g. ["3030031", "3030034"]. Empty = asked at start
IMAGE_ROTATIONS = [0, 0]     # Extra 90° clockwise turns of each sensor's image (2 = upside down). Keys '1'/'2' change it
CAMERA_MODE = 5              # Medium Range Static Scene
CONNECTION = "auto"          # "auto" = USB if available, else network. "gige" = force network with PACKET_SIZE
PACKET_SIZE = 1472           # Network packet size for CONNECTION = "gige" (smaller, e.g. 1000, if the network drops packets)
START_ALL_TOGETHER = True    # True = connect every sensor first, then start them. False = connect and start one by one

# Only shows the images: no trigger, no GPIO, no detection. To test if both sensors can stream at the same time.


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def frames_by_type(frames):
    ''' Frames indexed by their type: the camera doesn't always return them in the order they were requested '''
    return {int(f.frameType()): f for f in frames}


def get_frame_array(frame, rotation=0):
    ''' Frame as a float array rotated to the sensor's display orientation plus `rotation` extra 90° turns '''
    return np.ascontiguousarray(np.rot90(np.asarray(frame).astype(np.float32), -(1 + rotation)))


def depth_to_color(depth):
    ''' Colored depth, invalid pixels (no reading) in black '''
    valid = depth > 0
    if not np.any(valid):
        return np.zeros(depth.shape + (3,), dtype=np.uint8)
    low, high = np.percentile(depth[valid], [1, 99])
    normalized = np.clip((depth - low) / max(high - low, 1), 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def intensity_to_gray(intensity):
    ''' Intensity (infrared image) as 0-255 gray, ignoring the brightest 1% (reflections) '''
    high = np.percentile(intensity, 99)
    return (np.clip(intensity / max(high, 1e-6), 0, 1) * 255).astype(np.uint8)


class Viewer:
    ''' One sensor and its window. If something fails, the error is kept and shown, the other sensor keeps going '''

    def __init__(self, index, serial, rotation):
        self.index = index
        self.serial = serial
        self.rotation = rotation
        self.window = f"Sensor {index + 1} ({serial})"
        self.cam = None
        self.error = None
        self.view = 0
        self.frames = 0
        self.fps = 0.0
        self.fps_time = time.time()
        self.start_time = time.time()
        self.last_frame_time = None

    def connect(self):
        start = time.time()
        log(f"{self.window}: connecting ({CONNECTION})...")
        try:
            if CONNECTION == "gige":
                self.cam = tof.createTuiCameraGige(self.serial, PACKET_SIZE)
            else:
                self.cam = tof.TuiCamera(serial=self.serial)
            self.has_intensity = hasattr(tof.FrameType, "INTENSITY")
            self.frame_types = [tof.FrameType.Z]
            if self.has_intensity:
                self.frame_types.append(tof.FrameType.INTENSITY)
            self.cam.setOutputFrameTypes(self.frame_types)
            log(f"{self.window}: connected in {time.time() - start:.1f} s, firmware {self.cam.version()}")
        except Exception as e:
            self.fail("connecting", e, start)

    def start(self):
        if self.error:
            return
        start = time.time()
        log(f"{self.window}: starting...")
        try:
            self.cam.start()
            self.cam.switchConfig(CAMERA_MODE)
            self.start_time = time.time()
            log(f"{self.window}: streaming (started in {time.time() - start:.1f} s)")
        except Exception as e:
            self.fail("starting", e, start)

    def fail(self, step, e, start):
        self.error = f"Error {step}: {e}"
        log(f"{self.window}: ERROR {step} after {time.time() - start:.1f} s: {e}")
        traceback.print_exc()

    def update(self):
        if self.error:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "SIN CONEXION", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            for i in range(0, len(self.error), 50):
                cv2.putText(img, self.error[i:i + 50], (20, 110 + 15 * i // 50 * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(self.window, img)
            return
        try:
            if not self.cam.hasFrames():
                # No image yet: say so instead of leaving the window gray
                if self.last_frame_time is None or time.time() - self.last_frame_time > 2:
                    since = self.last_frame_time or self.start_time
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(img, f"SIN IMAGENES hace {time.time() - since:.0f} s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    cv2.putText(img, f"conectado={self.cam.isConnected()}  transmitiendo={self.cam.isStreaming()}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow(self.window, img)
                return
            self.last_frame_time = time.time()
            frames = self.cam.getFrames()
            while self.cam.hasFrames():
                frames = self.cam.getFrames()
        except Exception as e:
            self.fail("reading frames", e, time.time())
            return
        frames = frames_by_type(frames)
        if int(tof.FrameType.Z) not in frames:
            return
        depth = get_frame_array(frames[int(tof.FrameType.Z)], self.rotation)
        intensity = frames.get(int(tof.FrameType.INTENSITY)) if self.has_intensity else None

        if self.view == 1 and intensity is not None:
            img = cv2.cvtColor(intensity_to_gray(get_frame_array(intensity, self.rotation)), cv2.COLOR_GRAY2BGR)
        else:
            img = depth_to_color(depth)

        self.frames += 1
        if time.time() - self.fps_time >= 1:
            self.fps = self.frames / (time.time() - self.fps_time)
            self.frames, self.fps_time = 0, time.time()
        valid = depth > 0
        center = depth[depth.shape[0] // 2, depth.shape[1] // 2]
        text = f"{self.fps:.1f} fps  validos {100 * valid.mean():.0f}%  centro {center:.0f} mm"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(self.window, img)

    def stop(self):
        if self.cam is not None and not self.error:
            try:
                self.cam.stop()
            except Exception:
                pass


if __name__ == "__main__":
    import chronoptics.tof as tof

    serials = list(SERIALS)
    if not serials:
        serials = [input("Enter sensor 1 serial number: ").strip()]
        second = input("Enter sensor 2 serial number (empty = only one sensor): ").strip()
        if second:
            serials.append(second)

    viewers = [Viewer(i, s, IMAGE_ROTATIONS[i] if i < len(IMAGE_ROTATIONS) else 0) for i, s in enumerate(serials)]
    for v in viewers:
        cv2.namedWindow(v.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(v.window, 640, 540)

    if START_ALL_TOGETHER:
        for v in viewers:
            v.connect()
        for v in viewers:
            v.start()
    else:
        for v in viewers:
            v.connect()
            v.start()

    print("Keys: 1 / 2 = rotate sensor 1 / 2 image, v = depth / infrared, q = quit.")
    while True:
        for v in viewers:
            v.update()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('1'), ord('2')) and int(chr(key)) <= len(viewers):
            v = viewers[int(chr(key)) - 1]
            v.rotation = (v.rotation + 1) % 4
        elif key == ord('v'):
            for v in viewers:
                v.view = 1 - v.view

    for v in viewers:
        v.stop()
    cv2.destroyAllWindows()
