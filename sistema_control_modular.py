# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  you may obtain a copy of the License at
#
#      http:# www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************

"""
Sistema de Monitoreo y Metrología - Modo Monitor (Lazo Abierto)
Hardware: Jetson Nano + Cámaras Orbbec Femto Bolt + ToF VL53L1X
Display: TV por HDMI (optimizado para visualización a distancia)

NOTA: Este sistema NO controla el robot/impresora. Solo mide y muestra recomendaciones.
"""

import sys
import time
from datetime import datetime
import csv
import cv2 as cv
import numpy as np
from collections import deque

# -----------------------------
# IMPORTS: cámaras (separado)
# -----------------------------
try:
    from pyorbbecsdk import *  # noqa: F401,F403
    from utils import frame_to_bgr_image
except ImportError:
    print("Warning: 'pyorbbecsdk' / 'utils' not found. Camera functions will fail.")

# -----------------------------
# IMPORTS: ToF (separado)
# -----------------------------
try:
    import smbus2  # noqa: F401
    import VL53L1X
except ImportError:
    smbus2 = None
    VL53L1X = None
    print("Warning: VL53L1X libs not found. ToF functions will fail.")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Central configuration for the application."""

    # -------- ToF ----------
    TOF_ENABLED = True
    TOF_I2C_BUS = 1            # Jetson Nano normalmente usa I2C-1 (pines 3/5)
    TOF_I2C_ADDRESS = 0x29     # VL53L1X default
    TOF_TIMING_BUDGET_MS = 50  # 20–50ms típico
    TOF_INTER_MEASUREMENT_MS = 60

    TARGET_DISTANCE_MM = 200.0     # tu objetivo de distancia
    TOF_FILTER_WINDOW = 5          # suavizado

    # System Limits
    MAX_DEVICES = 2
    MAX_QUEUE_SIZE = 5

    # Control Constants
    PRINT_INTERVAL = 2  # seconds

    # Depth Filtering
    MIN_DEPTH = 20  # 20mm
    MAX_DEPTH = 1000  # 1000mm

    # Robot Safety Limits (para cálculo teórico de recomendaciones)
    MAX_SPEED = 0.15   # m/s - Maximum allowed speed
    MIN_SPEED = 0.002  # m/s - Minimum allowed speed

    # Image Processing / Metrology
    FACTOR_DISTANCIA = 0.324  # mm/pixel
    BORD_H = 1380
    BORD_V = 0
    BOQUILLA1 = 155
    BOQUILLA2 = 155

    # Analisis Circular
    CIRCLE_ANALYSIS_RADIUS = 140
    CENTER_X = 270
    CENTER_Y = 270

    # Default Control Params (para simulación de recomendaciones)
    DEFAULT_KP = 0.00025
    DEFAULT_KI = 0.00000
    DEFAULT_KD = 0.00000

    # GUI para TV
    HUD_PANEL_WIDTH = 500
    HUD_PANEL_HEIGHT = 260   # <-- aumentado para texto del ToF
    FONT_SCALE_LARGE = 2.0
    FONT_SCALE_MEDIUM = 1.5
    FONT_THICKNESS = 3


class DataLogger:
    """Handles logging of telemetry data to CSV."""
    def __init__(self):
        self.file = None
        self.writer = None
        self.filename = ""
        self.start_log()

    def start_log(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filename = f"log_monitor_{current_time}.csv"
        self.file = open(self.filename, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)

        # Columnas (incluye ToF)
        self.writer.writerow([
            'Timestamp',
            'Ancho_Medido_mm', 'SetPoint_mm', 'Error_Ancho_mm',
            'Velocidad_Recomendada_mps', 'Accion_Ancho',
            'P_term', 'I_term', 'D_term',
            'ToF_mm', 'Error_Distancia_mm', 'Accion_Altura'
        ])
        self.file.flush()

    def log(self, width, setpoint, error_w, velocity_recommended, action_w,
            p_term, i_term, d_term, tof_mm, ed_mm, action_h):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.writer.writerow([
            timestamp,
            round(width, 3),
            float(setpoint),
            round(error_w, 3),
            round(velocity_recommended, 6),
            action_w,
            round(p_term, 6),
            round(i_term, 6),
            round(d_term, 6),
            (round(tof_mm, 1) if tof_mm == tof_mm else ""),   # NaN -> vacío
            round(ed_mm, 1),
            action_h
        ])
        self.file.flush()

    def close(self):
        if self.file:
            self.file.flush()
            self.file.close()
            print(f"Datos guardados en: {self.filename}")


class RecommendationSimulator:
    """
    Simulador de Consejos - Calcula qué haría un controlador PID
    pero NO envía comandos. Solo genera recomendaciones para mostrar.
    """
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_accum = [0] * 6

    def calculate_recommendation(self, target, measured, base_speed):
        error = measured - target

        self.error_accum.append(error)
        if len(self.error_accum) > 10:
            self.error_accum.pop(0)

        prev_error = self.error_accum[-2] if len(self.error_accum) >= 2 else 0

        p_term = self.kp * error
        i_term = self.ki * np.sum(self.error_accum)
        d_term = self.kd * (error - prev_error)

        control_signal = p_term + i_term + d_term
        new_speed = base_speed + control_signal

        final_speed = max(Config.MIN_SPEED, min(Config.MAX_SPEED, new_speed))

        if abs(error) < 1.0:
            action = "OPTIMO"
        elif error > 1.0:
            action = "ACELERAR"
        else:
            action = "FRENAR"

        return final_speed, error, action, p_term, i_term, d_term


class VisionSystem:
    """Handles image processing and metrology."""
    def __init__(self):
        self.umbral_binarizacion = 100
        cv.namedWindow("Calibracion", cv.WINDOW_NORMAL)
        cv.createTrackbar("Umbral", "Calibracion", self.umbral_binarizacion, 255, lambda x: None)

        self.kernel_morph = np.array([[0,0,1,1,1,0,0],
                                      [0,1,1,1,1,1,0],
                                      [1,1,1,1,1,1,1],
                                      [1,1,1,1,1,1,1],
                                      [1,1,1,1,1,1,1],
                                      [0,1,1,1,1,1,0],
                                      [0,0,1,1,1,0,0] ]).astype(np.uint8)

    def process_images(self, color_image0, color_image1, setpoint, error, action_recommended):
        self.umbral_binarizacion = cv.getTrackbarPos("Umbral", "Calibracion")

        # Crop
        y_slice = slice(Config.BORD_V, 1079 - Config.BORD_V)

        x_slice1 = slice(Config.BORD_H - Config.BOQUILLA1, 1919 - Config.BOQUILLA1)
        image1_proco = color_image1[y_slice, x_slice1, :]

        x_slice2 = slice(Config.BORD_H - Config.BOQUILLA2, 1919 - Config.BOQUILLA2)
        image2_proco = color_image0[y_slice, x_slice2, :]

        image1_proc = image1_proco[:, :, 0]
        image2_proc = image2_proco[:, :, 0]

        # Binarization
        _, thres1 = cv.threshold(image1_proc, self.umbral_binarizacion, 255, cv.THRESH_BINARY)
        _, thres2 = cv.threshold(image2_proc, self.umbral_binarizacion, 255, cv.THRESH_BINARY)

        # Morphology
        thres1 = cv.morphologyEx(thres1, cv.MORPH_OPEN, self.kernel_morph)
        thres2 = cv.morphologyEx(thres2, cv.MORPH_OPEN, self.kernel_morph)

        thres1_rgb = np.dstack((thres1, thres1, thres1))
        thres2_rgb = np.dstack((thres2, thres2, thres2))

        # Resize
        width1 = color_image1.shape[1]
        height1 = color_image1.shape[0]
        target_size = (width1 // 2 - Config.BORD_H // 2, height1 // 2 - Config.BORD_V)

        if target_size[0] > 0 and target_size[1] > 0:
            thres1_rgb = cv.resize(thres1_rgb, target_size)
            image1_proco = cv.resize(image1_proco, target_size)
            thres2_rgb = cv.resize(thres2_rgb, target_size)
            image2_proco = cv.resize(image2_proco, target_size)

        # Rectification
        thres2_rgb = np.flip(thres2_rgb, axis=0)
        image2_proco = np.flip(image2_proco, axis=0)
        thres2_rgb = np.flip(thres2_rgb, axis=1)
        image2_proco = np.flip(image2_proco, axis=1)

        img_fil = np.concatenate((thres1_rgb, thres2_rgb), axis=1)
        img_ori = np.concatenate((image1_proco, image2_proco), axis=1)
        img_ori2 = img_ori.copy()

        img_hud = img_ori.copy()

        img_fil[:, :, 0] = cv.dilate(img_fil[:, :, 0], self.kernel_morph, iterations=2)

        # Circular mask
        white_mask = np.ones_like(img_fil[:, :, 1]) * 255
        r_analisis = cv.circle(white_mask.copy(),
                               (Config.CENTER_X, Config.CENTER_Y),
                               Config.CIRCLE_ANALYSIS_RADIUS,
                               (0, 0, 0), -1)

        img_fil[:, :, 0] = cv.add(img_fil[:, :, 0], r_analisis)
        img_fil[:, :, 0] = cv.subtract(white_mask, img_fil[:, :, 0])

        r_analisis_small = cv.circle(np.ones_like(img_fil[:, :, 1]) * 150,
                                     (Config.CENTER_X, Config.CENTER_Y),
                                     Config.CIRCLE_ANALYSIS_RADIUS,
                                     (0, 0, 0), -1)
        img_fil[:, :, 1] = cv.subtract(img_fil[:, :, 1], r_analisis_small)
        img_fil[:, :, 2] = cv.subtract(img_fil[:, :, 2], r_analisis_small)

        contours, _ = cv.findContours(img_fil[:, :, 0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        distancia = 0.0

        if len(contours) > 0:
            sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
            largest = sorted_contours[0]

            cv.drawContours(img_hud, largest, -1, (0, 255, 0), 3)

            # Medición
            radius_h = 4500
            radius_l = 4000

            bordes_circulo = []
            for i in largest:
                pt = i[0]
                radius_sq = ((pt[0] - 270) ** 2 + (pt[1] - 270) ** 2)
                if radius_l < radius_sq < radius_h:
                    bordes_circulo.append(pt)

            if len(bordes_circulo) > 1:
                p1 = bordes_circulo[0]
                p2 = bordes_circulo[-1]

                cv.line(img_hud, tuple(p1), tuple(p2), (0, 255, 255), 4)

                distancia_pix = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                distancia = np.round(distancia_pix * Config.FACTOR_DISTANCIA, 1)

        # HUD panel
        cv.rectangle(img_hud, (0, 0), (Config.HUD_PANEL_WIDTH, Config.HUD_PANEL_HEIGHT), (0, 0, 0), -1)

        if abs(error) < 1.0:
            color_texto = (0, 255, 0)
            status_text = "OPTIMO"
        elif error > 1.0:
            color_texto = (255, 200, 0)
            status_text = "MUY ANCHO - ACELERAR"
        else:
            color_texto = (0, 0, 255)
            status_text = "MUY DELGADO - FRENAR"

        cv.putText(img_hud, f"Ancho: {distancia} mm", (20, 60),
                   cv.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE_LARGE, color_texto, Config.FONT_THICKNESS)

        cv.putText(img_hud, f"Meta: {setpoint} mm", (20, 110),
                   cv.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE_MEDIUM, (255, 255, 255), Config.FONT_THICKNESS)

        cv.putText(img_hud, status_text, (20, 160),
                   cv.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE_LARGE, color_texto, Config.FONT_THICKNESS)

        cv.drawMarker(img_hud, (Config.CENTER_X, Config.CENTER_Y), (100, 100, 100),
                      cv.MARKER_CROSS, 30, 2)
        cv.circle(img_fil, (Config.CENTER_X, Config.CENTER_Y), 6, (102, 117, 179), -1)

        return distancia, img_fil, img_hud, img_ori2


class ToFSensor:
    def __init__(self):
        self.enabled = Config.TOF_ENABLED and (VL53L1X is not None)
        self.tof = None
        self.buf = deque(maxlen=Config.TOF_FILTER_WINDOW)

    def start(self):
        if not self.enabled:
            return False
        try:
            self.tof = VL53L1X.VL53L1X(i2c_bus=Config.TOF_I2C_BUS, i2c_address=Config.TOF_I2C_ADDRESS)
            self.tof.open()
            try:
                self.tof.set_timing_budget(Config.TOF_TIMING_BUDGET_MS * 1000)  # a veces es us
            except Exception:
                pass
            try:
                self.tof.start_ranging(1)
            except Exception:
                self.tof.start_ranging()
            return True
        except Exception as e:
            print(f"ToF init error: {e}")
            self.enabled = False
            return False

    def read_mm(self):
        if not self.enabled or self.tof is None:
            return None
        try:
            d = self.tof.get_distance()  # mm
            if d is None or d <= 0:
                return None
            self.buf.append(float(d))
            return sum(self.buf) / len(self.buf)
        except Exception:
            return None

    def stop(self):
        try:
            if self.tof:
                try:
                    self.tof.stop_ranging()
                except Exception:
                    pass
                self.tof.close()
        except Exception:
            pass


def height_recommendation(ed_mm, base_speed_mps):
    """
    Recomendación simple por altura/distancia:
    - ed_mm > 0: estás más lejos que la meta (ToF mide mayor distancia)
    - ed_mm < 0: estás más cerca que la meta
    """
    if abs(ed_mm) < 2.0:
        return base_speed_mps, "ALTURA OK"
    elif ed_mm > 2.0:
        return max(Config.MIN_SPEED, base_speed_mps - 0.005), "ALTURA: BAJAR VEL"
    else:
        return min(Config.MAX_SPEED, base_speed_mps + 0.005), "ALTURA: SUBIR VEL"


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    print("=" * 70)
    print(" Sistema de Monitoreo y Metrología - Modo Monitor")
    print(" (Lazo Abierto - Solo Medición y Visualización)")
    print("=" * 70)

    # ----------------------------------------------------
    # 1. INITIAL SETUP
    # ----------------------------------------------------
    try:
        raw_setpoint = input("Coloque el ancho objetivo del filamento [mm]:\n")
        set_point = float(raw_setpoint) if raw_setpoint else 30.0
        print()
        print("Velocidad base de referencia (para cálculo teórico de recomendaciones)")
        raw_vel = input("Velocidad base [mm/s]:\n")
        velocidad_base = float(raw_vel) / 1000 if raw_vel else 0.02  # a m/s
    except ValueError:
        print("Entrada inválida. Usando valores por defecto.")
        set_point = 30.0
        velocidad_base = 0.02

    logger = DataLogger()
    recommendation_sim = RecommendationSimulator(Config.DEFAULT_KP, Config.DEFAULT_KI, Config.DEFAULT_KD)
    vision = VisionSystem()

    tof = ToFSensor()
    tof_ok = tof.start()
    print(f"ToF sensor: {'OK' if tof_ok else 'OFF'}")

    # ----------------------------------------------------
    # 2. CAMERA CONNECTION
    # ----------------------------------------------------
    print("\nConectando con cámaras Orbbec...")
    try:
        ctx = Context()
        device_list = ctx.query_devices()
        curr_device_cnt = device_list.get_count()

        if curr_device_cnt != 2:
            print(f"ERROR: Se requieren exactamente 2 cámaras. Detectadas: {curr_device_cnt}")
            if curr_device_cnt == 0:
                print("No device connected")
            return

        device0 = device_list.get_device_by_index(0)
        device1 = device_list.get_device_by_index(1)

        config0 = pyorbbecsdk.Config()
        config1 = pyorbbecsdk.Config()

        pipeline0 = Pipeline(device0)
        pipeline1 = Pipeline(device1)

        # Depth
        profile_list0 = pipeline0.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile0 = profile_list0.get_default_video_stream_profile()
        config0.enable_stream(depth_profile0)

        profile_list1 = pipeline1.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile1 = profile_list1.get_default_video_stream_profile()
        config1.enable_stream(depth_profile1)

        print("Depth profiles configured.")

        # Color
        profile_list0 = pipeline0.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        profile_list1 = pipeline1.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

        try:
            color_profile0 = profile_list0.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
            color_profile1 = profile_list1.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(f"Warning: Specific profile not found, using default. {e}")
            color_profile0 = profile_list0.get_default_video_stream_profile()
            color_profile1 = profile_list1.get_default_video_stream_profile()

        config0.enable_stream(color_profile0)
        config1.enable_stream(color_profile1)

        pipeline0.start(config0)
        pipeline1.start(config1)
        print("Streams iniciados correctamente.")

    except Exception as e:
        print(f"Critical Camera Error: {e}")
        return

    # Windows (crear una vez, no en cada loop)
    cv.namedWindow("Visualizacion completa", cv.WINDOW_NORMAL)
    cv.setWindowProperty("Visualizacion completa", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.namedWindow("Filtro", cv.WINDOW_NORMAL)

    # ----------------------------------------------------
    # 3. MONITORING LOOP
    # ----------------------------------------------------
    time.sleep(1)
    last_print_time = time.time()

    print("\n" + "=" * 70)
    print(" Iniciando monitoreo... (Presione 'q' o ESC para salir)")
    print(" NOTA: Este sistema NO controla el robot. Solo mide y muestra recomendaciones.")
    print("=" * 70)

    try:
        while True:
            # A. ACQUIRE FRAMES
            frames0 = pipeline0.wait_for_frames(200)
            frames1 = pipeline1.wait_for_frames(200)
            if frames0 is None or frames1 is None:
                continue

            depth_frame0 = frames0.get_depth_frame()
            depth_frame1 = frames1.get_depth_frame()
            color_frame0 = frames0.get_color_frame()
            color_frame1 = frames1.get_color_frame()

            if not depth_frame0 or not depth_frame1 or not color_frame0 or not color_frame1:
                continue

            # B. CONVERT TO IMAGES
            color_image0 = frame_to_bgr_image(color_frame0)
            color_image1 = frame_to_bgr_image(color_frame1)
            if color_image0 is None or color_image1 is None:
                print("Failed to convert frame to image")
                continue

            # C. FIRST PASS (solo para medir ancho)
            measured_width, img_fil, img_hud, img_ori2 = vision.process_images(
                color_image0, color_image1, set_point, 0.0, ""
            )

            # D. PID SIM (recomendación por ancho)
            velocity_recommended, error_w, action_w, p_term, i_term, d_term = \
                recommendation_sim.calculate_recommendation(set_point, measured_width, velocidad_base)

            # E. ToF
            tof_mm = tof.read_mm()
            if tof_mm is None:
                tof_mm = float("nan")

            ed = (tof_mm - Config.TARGET_DISTANCE_MM) if (tof_mm == tof_mm) else 0.0  # NaN check
            vel_h, action_h = height_recommendation(ed, velocity_recommended)

            # F. SECOND PASS (HUD con recomendación de ancho)
            measured_width, img_fil, img_hud, img_ori2 = vision.process_images(
                color_image0, color_image1, set_point, error_w, action_w
            )

            # G. Dibujar ToF en HUD (sin tocar process_images)
            cv.putText(
                img_hud,
                f"ToF: {tof_mm:.1f} mm  ed: {ed:.1f}  {action_h}",
                (20, 220),
                cv.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2
            )

            # Print periodic status
            current_time = time.time()
            if current_time - last_print_time >= Config.PRINT_INTERVAL:
                print(f"Ancho: {measured_width:.1f} mm | ew: {error_w:+.2f} mm | {action_w} | "
                      f"Vel(w): {velocity_recommended*1000:.2f} mm/s")
                print(f"ToF: {tof_mm:.1f} mm | ed: {ed:+.1f} mm | {action_h} | Vel(h): {vel_h*1000:.2f} mm/s")
                last_print_time = current_time

            # H. LOGGING
            logger.log(measured_width, set_point, error_w, velocity_recommended, action_w,
                       p_term, i_term, d_term, tof_mm, ed, action_h)

            # I. VISUALIZATION
            final_view = np.concatenate((img_fil, img_hud), axis=1)
            cv.imshow("Visualizacion completa", final_view)
            cv.imshow("Filtro", img_ori2)

            key = cv.waitKey(1)
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\nDeteniendo por usuario...")
    except Exception as e:
        print(f"\nError inesperado en bucle: {e}")
    finally:
        print("Cerrando recursos...")
        try:
            tof.stop()
        except Exception:
            pass
        try:
            logger.close()
        except Exception:
            pass
        try:
            pipeline0.stop()
            pipeline1.stop()
        except Exception:
            pass
        cv.destroyAllWindows()
        print("Sistema detenido.")


if __name__ == "__main__":
    main()
