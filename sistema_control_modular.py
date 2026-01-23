# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  you may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to   in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************

"""
Sistema de Monitoreo y Metrología - Modo Monitor (Lazo Abierto)
Hardware: Jetson Nano + Cámaras Orbbec Femto Bolt
Display: TV por HDMI (optimizado para visualización a distancia)

NOTA: Este sistema NO controla el robot/impresora. Solo mide y muestra recomendaciones.
"""

import sys
import time
from datetime import datetime
import csv
import cv2 as cv
import numpy as np

try:
    from pyorbbecsdk import *
    from utils import frame_to_bgr_image
except ImportError:
    print("Warning: 'pyorbbecsdk' library not found. Camera functions will fail.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Central configuration for the application."""
    # System Limits
    MAX_DEVICES = 2
    MAX_QUEUE_SIZE = 5
    
    # Control Constants
    PRINT_INTERVAL = 2  # seconds
    
    # Depth Filtering
    MIN_DEPTH = 20  # 20mm
    MAX_DEPTH = 1000  # 1000mm
    
    # Robot Safety Limits (para cálculo teórico de recomendaciones)
    MAX_SPEED = 0.15  # m/s - Maximum allowed speed
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
    
    # GUI para TV (tamaños aumentados)
    HUD_PANEL_WIDTH = 500  # Doble del original (250 -> 500)
    HUD_PANEL_HEIGHT = 180  # Doble del original (90 -> 180)
    FONT_SCALE_LARGE = 2.0  # Para texto principal (era 0.8)
    FONT_SCALE_MEDIUM = 1.5  # Para texto secundario (era 0.6)
    FONT_THICKNESS = 3  # Grosor aumentado para mejor visibilidad


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
        # Columnas: Timestamp, Ancho_Medido, SetPoint, Error, Velocidad_Recomendada, Accion_Recomendada, P_term, I_term, D_term
        self.writer.writerow(['Timestamp', 'Ancho_Medido', 'SetPoint', 'Error', 
                              'Velocidad_Recomendada', 'Accion_Recomendada',
                              'P_term', 'I_term', 'D_term'])
        self.file.flush()

    def log(self, width, setpoint, error, velocity_recommended, action_recommended, p_term, i_term, d_term):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.writer.writerow([
            timestamp,
            round(width, 3),
            setpoint,
            round(error, 3),
            round(velocity_recommended, 6),
            action_recommended,
            round(p_term, 6),
            round(i_term, 6),
            round(d_term, 6)
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
        self.error_accum = [0] * 6  # Initialize buffer
        
    def calculate_recommendation(self, target, measured, base_speed):
        """
        Calcula recomendación teórica del controlador PID.
        
        Returns:
            tuple: (velocidad_recomendada, error, accion_texto, p_term, i_term, d_term)
        """
        error = measured - target
        
        # Corrección FIFO: eliminar el dato más antiguo
        self.error_accum.append(error)
        if len(self.error_accum) > 10:
            self.error_accum.pop(0) 
            
        prev_error = self.error_accum[-2] if len(self.error_accum) >= 2 else 0

        p_term = self.kp * error
        i_term = self.ki * np.sum(self.error_accum)
        d_term = self.kd * (error - prev_error)
        
        control_signal = p_term + i_term + d_term
        new_speed = base_speed + control_signal
        
        # Safety Saturations
        final_speed = max(Config.MIN_SPEED, min(Config.MAX_SPEED, new_speed))
        
        # Determinar acción recomendada basada en el error
        if abs(error) < 1.0:
            action = "OPTIMO"
        elif error > 1.0:  # Filamento muy ancho
            action = "ACELERAR"
        else:  # error < -1.0, filamento muy delgado
            action = "FRENAR"
        
        return final_speed, error, action, p_term, i_term, d_term


class VisionSystem:
    """Handles image processing and metrology."""
    def __init__(self):
        # GUI Setup
        self.umbral_binarizacion = 100
        cv.namedWindow("Calibracion", cv.WINDOW_NORMAL)
        cv.createTrackbar("Umbral", "Calibracion", self.umbral_binarizacion, 255, lambda x: None)
        
        # Kernel Definition
        self.kernel_morph = np.array([[0,0,1,1,1,0,0],
                                      [0,1,1,1,1,1,0],
                                      [1,1,1,1,1,1,1],
                                      [1,1,1,1,1,1,1],
                                      [1,1,1,1,1,1,1],
                                      [0,1,1,1,1,1,0],
                                      [0,0,1,1,1,0,0] ]).astype(np.uint8)

    def process_images(self, color_image0, color_image1, setpoint, error, action_recommended):
        """Main processing pipeline optimized for TV display."""
        
        # 1. Update Threshold from GUI
        self.umbral_binarizacion = cv.getTrackbarPos("Umbral", "Calibracion")

        # 2. Crop Images
        y_slice = slice(Config.BORD_V, 1079 - Config.BORD_V)
        
        x_slice1 = slice(Config.BORD_H - Config.BOQUILLA1, 1919 - Config.BOQUILLA1)
        image1_proco = color_image1[y_slice, x_slice1, :]
        
        x_slice2 = slice(Config.BORD_H - Config.BOQUILLA2, 1919 - Config.BOQUILLA2)
        image2_proco = color_image0[y_slice, x_slice2, :]
        
        image1_proc = image1_proco[:, :, 0]
        image2_proc = image2_proco[:, :, 0]

        # 3. Binarization
        _, thres1 = cv.threshold(image1_proc, self.umbral_binarizacion, 255, cv.THRESH_BINARY)
        _, thres2 = cv.threshold(image2_proc, self.umbral_binarizacion, 255, cv.THRESH_BINARY)

        # 4. Morphology
        thres1 = cv.morphologyEx(thres1, cv.MORPH_OPEN, self.kernel_morph)
        thres2 = cv.morphologyEx(thres2, cv.MORPH_OPEN, self.kernel_morph)

        # 5. Prepare for Merge
        thres1_rgb = np.dstack((thres1, thres1, thres1))
        thres2_rgb = np.dstack((thres2, thres2, thres2))

        # 6. Resize
        width1 = color_image1.shape[1]
        height1 = color_image1.shape[0]
        
        target_size = (width1 // 2 - Config.BORD_H // 2, height1 // 2 - Config.BORD_V)
        
        if target_size[0] > 0 and target_size[1] > 0:
            thres1_rgb = cv.resize(thres1_rgb, target_size)
            image1_proco = cv.resize(image1_proco, target_size)
            thres2_rgb = cv.resize(thres2_rgb, target_size)
            image2_proco = cv.resize(image2_proco, target_size)

        # 7. Rectification (Flip)
        thres2_rgb = np.flip(thres2_rgb, axis=0)
        image2_proco = np.flip(image2_proco, axis=0)
        thres2_rgb = np.flip(thres2_rgb, axis=1)
        image2_proco = np.flip(image2_proco, axis=1)

        img_fil = np.concatenate((thres1_rgb, thres2_rgb), axis=1)
        img_ori = np.concatenate((image1_proco, image2_proco), axis=1)
        img_ori2 = img_ori.copy()
        
        # 8. Detection HUD Logic
        img_hud = img_ori.copy()
        
        img_fil[:,:,0] = cv.dilate(img_fil[:,:,0], self.kernel_morph, iterations=2)

        # --- Circular Analysis / Masking ---
        white_mask = np.ones_like(img_fil[:,:,1]) * 255
        r_analisis = cv.circle(white_mask.copy(), (Config.CENTER_X, Config.CENTER_Y), 
                               Config.CIRCLE_ANALYSIS_RADIUS, (0,0,0), -1)
        
        img_fil[:,:,0] = cv.add(img_fil[:,:,0], r_analisis)
        img_fil[:,:,0] = cv.subtract(white_mask, img_fil[:,:,0])
        
        r_analisis_small = cv.circle(np.ones_like(img_fil[:,:,1])*150, (Config.CENTER_X, Config.CENTER_Y), 
                                     Config.CIRCLE_ANALYSIS_RADIUS, (0,0,0), -1)
        img_fil[:,:,1] = cv.subtract(img_fil[:,:,1], r_analisis_small)
        img_fil[:,:,2] = cv.subtract(img_fil[:,:,2], r_analisis_small)

        # 9. Find Contours
        contours, hierarchy = cv.findContours(img_fil[:,:,0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        distancia = 0.0
        
        if len(contours) > 0:
            sorted_contours = sorted(contours, key = cv.contourArea, reverse=True)
            largest = sorted_contours[0]

            # Dibujar contorno del concreto (Verde)
            cv.drawContours(img_hud, largest, -1, (0, 255, 0), 3)
            
            # --- MEDICIÓN SIMPLIFICADA ---
            x, y, w, h = cv.boundingRect(largest)
            ancho_pixeles = min(w, h)
            
            # Recuperando el cálculo de distancia promedio para mantener calibración:
            bordes_circulo = []
            radius_h = 4500
            radius_l = 4000
            
            for i in largest:
                pt = i[0]
                radius_sq = ((pt[0]-270)**2 + (pt[1]-270)**2)
                if radius_l < radius_sq < radius_h:
                    bordes_circulo.append(pt)
            
            if len(bordes_circulo) > 1:
                p1 = bordes_circulo[0]
                p2 = bordes_circulo[-1]
                
                # Dibujar LINEA DE MEDICIÓN (Amarillo, más gruesa para TV)
                cv.line(img_hud, tuple(p1), tuple(p2), (0, 255, 255), 4)
                
                # Distancia Euclidiana
                distancia_pix = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                distancia = np.round(distancia_pix * Config.FACTOR_DISTANCIA, 1)

        # 10. Draw HUD Info - OPTIMIZADO PARA TV (GRANDE)
        # Crear un rectángulo negro más grande
        cv.rectangle(img_hud, (0, 0), (Config.HUD_PANEL_WIDTH, Config.HUD_PANEL_HEIGHT), (0, 0, 0), -1) 
        
        # SEMÁFORO VISUAL según el error
        if abs(error) < 1.0:
            # OPTIMO - VERDE
            color_texto = (0, 255, 0)
            status_text = "OPTIMO"
        elif error > 1.0:
            # MUY ANCHO - ACELERAR - AZUL
            color_texto = (255, 200, 0)  # Azul cian brillante
            status_text = "MUY ANCHO - ACELERAR"
        else:  # error < -1.0
            # MUY DELGADO - FRENAR - ROJO
            color_texto = (0, 0, 255)
            status_text = "MUY DELGADO - FRENAR"
        
        # Texto 1: ANCHO (Grande)
        cv.putText(img_hud, f"Ancho: {distancia} mm", (20, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE_LARGE, color_texto, Config.FONT_THICKNESS)
        
        # Texto 2: META (Mediano)
        cv.putText(img_hud, f"Meta: {setpoint} mm", (20, 110), 
                   cv.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE_MEDIUM, (255, 255, 255), Config.FONT_THICKNESS)

        # Texto 3: ESTADO/RECOMENDACIÓN (Grande y con color del semáforo)
        cv.putText(img_hud, status_text, (20, 160), 
                   cv.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE_LARGE, color_texto, Config.FONT_THICKNESS)

        # Markers (más grandes para TV)
        cv.drawMarker(img_hud, (Config.CENTER_X, Config.CENTER_Y), (100, 100, 100), cv.MARKER_CROSS, 30, 2)
        cv.circle(img_fil, (Config.CENTER_X, Config.CENTER_Y), 6, (102,117,179), -1)
        
        return distancia, img_fil, img_hud, img_ori2


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
    
    # Inputs
    try:
        raw_setpoint = input("Coloque el ancho objetivo del filamento [mm]:\n")
        set_point = float(raw_setpoint) if raw_setpoint else 30.0
        print()
        print("Velocidad base de referencia (para cálculo teórico de recomendaciones)")
        raw_vel = input("Velocidad base [mm/s]:\n")
        velocidad_base = float(raw_vel)/1000 if raw_vel else 0.02  # Convertir a m/s
    except ValueError:
        print("Entrada inválida. Usando valores por defecto.")
        set_point = 30.0
        velocidad_base = 0.02

    # Initialize Modules
    logger = DataLogger()
    recommendation_sim = RecommendationSimulator(Config.DEFAULT_KP, Config.DEFAULT_KI, Config.DEFAULT_KD)
    vision = VisionSystem()
    
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
            if curr_device_cnt == 0: print("No device connected")
            return

        device0 = device_list.get_device_by_index(0)
        device1 = device_list.get_device_by_index(1)

        config0 = pyorbbecsdk.Config()
        config1 = pyorbbecsdk.Config()

        pipeline0 = Pipeline(device0)
        pipeline1 = Pipeline(device1)

        # ENABLE STREAMS
        profile_list0 = pipeline0.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile0 = profile_list0.get_default_video_stream_profile()
        config0.enable_stream(depth_profile0)
        
        profile_list1 = pipeline1.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile1 = profile_list1.get_default_video_stream_profile()
        config1.enable_stream(depth_profile1)
        
        print("Depth profiles configured.")

        # Color Profiles
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
        
        # Start Streams
        pipeline0.start(config0)
        pipeline1.start(config1)
        print("Streams iniciados correctamente.")
        
    except Exception as e:
        print(f"Critical Camera Error: {e}")
        return

    # ----------------------------------------------------
    # 3. MONITORING LOOP
    # ----------------------------------------------------
    time.sleep(1)  # Warmup
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

            # C. PROCESS VISION
            measured_width, img_fil, img_hud, img_ori2 = vision.process_images(
                color_image0, color_image1, set_point, 0, ""  # error y action se calculan después
            )
            
            # D. CALCULAR RECOMENDACIÓN (Simulación teórica)
            velocity_recommended, error, action_recommended, p_term, i_term, d_term = \
                recommendation_sim.calculate_recommendation(set_point, measured_width, velocidad_base)
            
            # Actualizar HUD con la recomendación calculada
            measured_width, img_fil, img_hud, img_ori2 = vision.process_images(
                color_image0, color_image1, set_point, error, action_recommended
            )
            
            # Print periodic status
            current_time = time.time()
            if current_time - last_print_time >= Config.PRINT_INTERVAL:
                print(f"Ancho: {measured_width} mm | Error: {error:.2f} mm | Recomendación: {action_recommended}")
                print(f"Velocidad Recomendada: {round(velocity_recommended*1000,2)} mm/s")
                last_print_time = current_time

            # E. LOGGING (Sin comunicación con impresora)
            logger.log(measured_width, set_point, error, velocity_recommended, 
                      action_recommended, p_term, i_term, d_term)
            
            # F. VISUALIZATION
            final_view = np.concatenate((img_fil, img_hud), axis=1)
            
            # Mostrar en pantalla completa para TV
            cv.namedWindow("Visualizacion completa", cv.WINDOW_NORMAL)
            cv.setWindowProperty("Visualizacion completa", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            cv.imshow("Visualizacion completa", final_view)
            cv.imshow("Filtro", img_ori2)

            key = cv.waitKey(1)
            if key == ord('q') or key == 27:  # q or ESC
                break
                
    except KeyboardInterrupt:
        print("\nDeteniendo por usuario...")
    except Exception as e:
        print(f"\nError inesperado en bucle: {e}")
    finally:
        # 4. CLEANUP
        print("Cerrando recursos...")
        logger.close()
        try:
            pipeline0.stop()
            pipeline1.stop()
        except: pass
        cv.destroyAllWindows()
        print("Sistema detenido.")

if __name__ == "__main__":
    main()
