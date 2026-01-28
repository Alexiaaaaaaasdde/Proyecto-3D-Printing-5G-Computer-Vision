#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Monitoreo/Metrología de filamento (Jetson Nano + Orbbec + ToF)

- Problema principal en Jetson Nano: el stream de COLOR suele llegar como MJPEG 1080p.
  Eso implica decodificar JPEG en CPU para cada frame (y peor si son 2 cámaras).
  En una Nano 4GB, ese paso domina el tiempo → típicamente ~1–2 FPS.

- Optimización aplicada: decodificar directo a escala de grises (más barato) y
  procesar a resolución reducida (640x480) ANTES del análisis de contornos.

- Arquitectura de captura: hilos persistentes por cámara para evitar overhead de
  crear/join threads en cada loop. El loop principal consume “el último frame”
  disponible (modelo tipo “latest sample” usado en sistemas en tiempo real).


"""

import cv2
import numpy as np
import time
import sys
import os
import threading
from collections import deque
from datetime import datetime

# ===========================
# FIX PARA JETSON NANO
# ===========================
# Fix para error eglMakeCurrent en Jetson Nano
os.environ['QT_X11_NO_MITSHM'] = '1'

# ===========================
# IMPORTS DE SENSORES
# ===========================
try:
    import VL53L1X
    TOF_DISPONIBLE = True
except ImportError:
    print("Advertencia: VL53L1X no disponible. Sensor ToF deshabilitado.")
    TOF_DISPONIBLE = False

try:
    from pyorbbecsdk import Context, Pipeline, Config, OBSensorType, OBFormat, VideoStreamProfile
    from utils import frame_to_bgr_image, frame_to_gray_image
    ORBBEC_DISPONIBLE = True
except ImportError:
    print("ERROR: pyorbbecsdk no disponible.")
    ORBBEC_DISPONIBLE = False
    sys.exit(1)

# ===========================
# CONFIGURACIÓN OPTIMIZADA
# ===========================
class ConfigSistema:
    # ===========================
    # PARÁMETROS CLAVE 
    # ===========================
    # - CAMERA_*: “objetivo” de configuración. Ojo: la Femto Bolt puede ignorarlo y
    #   entregar su perfil default (ej. 1920x1080 MJPEG).
    # - USE_COLOR: True usa COLOR; False fuerza DEPTH (suele ser más liviano).
    # - PIXELS_TO_MM: calibración (depende de óptica/ROI/distancia). Ajustar con patrón.
    #
    # Cámaras - objetivo 640x480 @ 30fps (como código base)
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    USE_COLOR = True              # Usar cámara de color en vez de depth
    
    # Visualización
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480
    
    # Procesamiento de imagen
    UMBRAL_BINARIO = 80          # Umbral para binarización
    MIN_AREA_CONTORNO = 50       # Área mínima del contorno en píxeles (reducido para detectar filamentos delgados)
    
    # Medición
    PIXELS_TO_MM = 0.075         # Factor para 640x480 (ajustar según calibración)
    FILTER_WINDOW = 5            # Ventana para suavizado de mediciones
    
    # ToF
    TOF_FILTER_WINDOW = 5
    
    # Control
    TOLERANCIA_ERROR = 0.5       # mm de tolerancia

    # ===========================
    # ANTI-"DISTRACCIÓN" (ROI + TRACKING)
    # ===========================
    USE_ROI = True

    # ROI en porcentajes sobre 640x480 (ajusta si tu filamento está en otra zona)
    ROI_X1 = 0.25
    ROI_X2 = 0.75
    ROI_Y1 = 0.20
    ROI_Y2 = 0.80

    HOLD_LAST_N_MISSES = 2   # cuántos frames aguanta sin contorno (anti-jitter)


# ===========================
# CLASE SENSOR TOF
# ===========================
class SensorToF:
    def __init__(self):
        self.tof = None
        self.disponible = TOF_DISPONIBLE
        self.buffer_distancias = deque(maxlen=ConfigSistema.TOF_FILTER_WINDOW)
        
    def inicializar(self):
        if not self.disponible:
            return False
        try:
            self.tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
            self.tof.open()
            self.tof.start_ranging(2)
            print("✓ ToF sensor: OK")
            return True
        except Exception as e:
            print(f"✗ Error ToF: {e}")
            self.disponible = False
            return False
    
    def leer_distancia(self):
        # Telecom: filtro temporal simple (promedio móvil) para reducir ruido.
        # Esto equivale a un low-pass FIR de ventana fija.
        if not self.disponible or self.tof is None:
            return None
        try:
            distancia_mm = self.tof.get_distance()
            if distancia_mm > 0:
                distancia_cm = distancia_mm / 10.0
                self.buffer_distancias.append(distancia_cm)
                # Retornar promedio suavizado
                if len(self.buffer_distancias) > 0:
                    return np.mean(self.buffer_distancias)
        except Exception as e:
            print(f"Error leyendo ToF: {e}")
        return None
    
    def cerrar(self):
        if self.tof is not None:
            try:
                self.tof.stop_ranging()
                self.tof.close()
            except:
                pass

# ===========================
# CLASE MEDICIÓN DE ANCHO
# ===========================
class MedidorAncho:
    def __init__(self):
        self.buffer_anchos = deque(maxlen=ConfigSistema.FILTER_WINDOW)
        self.umbral = ConfigSistema.UMBRAL_BINARIO
        
        # Kernel morfológico optimizado
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Tracking simple para evitar que se "distraiga"
        self.last_center = None
        self.last_ancho = None
        self.miss_count = 0


    def procesar_frame(self, depth_frame, color_frame=None):
        """
        Procesa un frame y calcula el ancho del filamento
        Usa COLOR si está disponible, sino DEPTH
        Retorna: ancho_mm, imagen_procesada
        """
        # Telecom: Selección de fuente.
        # - COLOR: más “rico” visualmente, pero puede venir comprimido (MJPEG) y ser caro.
        # - DEPTH: suele venir raw uint16 (más barato), pero depende del escenario.
        if color_frame is not None and ConfigSistema.USE_COLOR:
            return self._procesar_color(color_frame)
        elif depth_frame is not None:
            return self._procesar_depth(depth_frame)
        return None, None

    def _procesar_color(self, color_frame):
        """Procesa frame de color para detectar filamento.

        Pipeline (rápido y robusto):
        1) Decodificar a GRAY (si viene MJPEG) para ahorrar CPU.
        2) Downscale a 640x480 antes del procesamiento.
        3) OTSU + morfología + contornos + minAreaRect.
        """
        try:
            # Telecom: el cuello de botella es la decodificación MJPEG.
            # Decodificar a GRAY reduce ~3x memoria vs BGR y baja el costo.
            gray = frame_to_gray_image(color_frame)
            if gray is None:
                return None, None

            # Telecom: bajar resolución reduce el costo O(N) de filtros/contornos.
            # 1920x1080 → 640x480 ≈ 6.75x menos píxeles (y suele sentirse “>5x”).

            # --- ROI (para que no se distraiga con cosas fuera del filamento) ---
            PROC_W, PROC_H = 640, 480
            h_orig, w_orig = gray.shape[:2]
            if w_orig > PROC_W or h_orig > PROC_H:
                gray = cv2.resize(gray, (PROC_W, PROC_H))

            x_off, y_off = 0, 0
            work = gray
            if ConfigSistema.USE_ROI:
                x1 = int(PROC_W * ConfigSistema.ROI_X1)
                x2 = int(PROC_W * ConfigSistema.ROI_X2)
                y1 = int(PROC_H * ConfigSistema.ROI_Y1)
                y2 = int(PROC_H * ConfigSistema.ROI_Y2)
                work = gray[y1:y2, x1:x2]
                x_off, y_off = x1, y1

            # Umbral (usa tu self.umbral y OTSU)
            ret, binary = cv2.threshold(work, self.umbral, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Telecom: morfología para limpiar ruido impulsivo y cerrar huecos (mejora contorno).
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)

            # Telecom: contornos = extracción de “blobs” (ROI implícita).
            result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = result[0] if len(result) == 2 else result[1]

            # Imagen de salida para UI (BGR porque OpenCV imshow espera BGR/Gray)
            vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if ConfigSistema.USE_ROI:
                cv2.rectangle(vis_image, (x_off, y_off),
                      (x_off + work.shape[1], y_off + work.shape[0]),
                      (255, 0, 0), 2)


            ancho_mm = None

            if len(contours) > 0:
                # Filtrar contornos por área
                contornos_validos = [c for c in contours
                                    if cv2.contourArea(c) > ConfigSistema.MIN_AREA_CONTORNO]

                if contornos_validos:
                    # Tomar contorno principal:
                    # - Si no hay historial: el más grande (como antes)
                    # - Si hay historial: el más cercano al centro anterior (evita "distracciones")
                    if self.last_center is None:
                        contorno_principal = max(contornos_validos, key=cv2.contourArea)
                    else:
                        lx, ly = self.last_center
                        def dist2(c):
                            x, y, w, h = cv2.boundingRect(c)
                            cx = x_off + x + w / 2.0
                            cy = y_off + y + h / 2.0
                            dx = cx - lx
                            dy = cy - ly
                            return dx*dx + dy*dy
                        contorno_principal = min(contornos_validos, key=dist2)

                    # Desplazar contorno del ROI a coordenadas globales para dibujar
                    cont_shift = contorno_principal.copy()
                    cont_shift[:, 0, 0] += x_off
                    cont_shift[:, 0, 1] += y_off
                    cv2.drawContours(vis_image, [cont_shift], -1, (0, 255, 0), 2)

                    # Telecom: minAreaRect da orientación + dimensiones. El “ancho”
                    # se asume como el lado menor del rectángulo (filamento ~cilindro).
                    rect = cv2.minAreaRect(cont_shift)  # usar el contorno ya desplazado
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    # Dibujar rectángulo
                    cv2.drawContours(vis_image, [box], 0, (0, 0, 255), 2)

                    # Calcular ancho (el lado más corto)
                    width_rect, height_rect = rect[1]
                    ancho_pixels = min(width_rect, height_rect)

                    # Telecom: conversión pixel→mm (calibración). Ajustar PIXELS_TO_MM.
                    ancho_mm = ancho_pixels * ConfigSistema.PIXELS_TO_MM

                    # Guardar historial para tracking y hold
                    self.last_center = rect[0]
                    self.last_ancho = ancho_mm
                    self.miss_count = 0


                    # Añadir al buffer
                    self.buffer_anchos.append(ancho_mm)

                    # Telecom: filtro temporal (promedio móvil) para suavizar jitter.
                    if len(self.buffer_anchos) > 0:
                        ancho_mm = np.mean(self.buffer_anchos)

                    # Dibujar centro
                    center = tuple(np.int0(rect[0]))
                    cv2.circle(vis_image, center, 5, (255, 0, 255), -1)

                    # Dibujar línea de medición (azul)
                    width_rect, height_rect = rect[1]
                    if width_rect < height_rect:
                        # El ancho es horizontal
                        x_offset = int(width_rect / 2)
                        pt1 = (center[0] - x_offset, center[1])
                        pt2 = (center[0] + x_offset, center[1])
                    else:
                        # El ancho es vertical
                        y_offset = int(height_rect / 2)
                        pt1 = (center[0], center[1] - y_offset)
                        pt2 = (center[0], center[1] + y_offset)
                    cv2.line(vis_image, pt1, pt2, (255, 255, 0), 3)

                    # Texto con medición (MÁS GRANDE y en posición fija)
                    cv2.putText(vis_image, f"ANCHO: {ancho_mm:.2f}mm",
                               (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                else:
                    self.miss_count += 1
                    if self.last_ancho is not None and self.miss_count <= ConfigSistema.HOLD_LAST_N_MISSES:
                        return self.last_ancho, vis_image

            else:
                self.miss_count += 1
                if self.last_ancho is not None and self.miss_count <= ConfigSistema.HOLD_LAST_N_MISSES:
                    return self.last_ancho, vis_image

            return ancho_mm, vis_image

        except Exception as e:
            print(f"Error procesando color: {e}")
            return None, None

    def _procesar_depth(self, depth_frame):
        """Procesa frame de depth (método original)"""
        try:
            # Convertir depth a array numpy
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
            
            # Normalizar para procesamiento
            depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Threshold simple OTSU (más rápido)
            ret, binary = cv2.threshold(depth_norm, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Operaciones morfológicas para limpiar
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
            
            # Encontrar contornos (compatible con OpenCV 3 y 4)
            result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = result[0] if len(result) == 2 else result[1]
            
            # Crear imagen de visualización en color
            vis_image = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)

            ancho_mm = None
            
            if len(contours) > 0:
                # Filtrar contornos por área
                contornos_validos = [c for c in contours 
                                    if cv2.contourArea(c) > ConfigSistema.MIN_AREA_CONTORNO]
                
                if contornos_validos:
                    # Tomar el contorno más grande
                    contorno_principal = max(contornos_validos, key=cv2.contourArea)

                    # Dibujar contorno
                    cv2.drawContours(vis_image, [contorno_principal], -1, (0, 255, 0), 2)

                    # Calcular ancho mediante rectángulo rotado
                    rect = cv2.minAreaRect(contorno_principal)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # Dibujar rectángulo
                    cv2.drawContours(vis_image, [box], 0, (0, 0, 255), 2)
                    
                    # Calcular ancho (el lado más corto del rectángulo)
                    width, height = rect[1]
                    ancho_pixels = min(width, height)
                    
                    # Convertir a mm
                    ancho_mm = ancho_pixels * ConfigSistema.PIXELS_TO_MM

                    # Añadir al buffer para suavizado
                    self.buffer_anchos.append(ancho_mm)
                    
                    # Calcular promedio suavizado
                    if len(self.buffer_anchos) > 0:
                        ancho_mm = np.mean(self.buffer_anchos)
                    
                    # Dibujar punto central
                    center = tuple(np.int0(rect[0]))
                    cv2.circle(vis_image, center, 5, (255, 0, 255), -1)
                    
                    # Dibujar línea de medición (azul)
                    width_rect, height_rect = rect[1]
                    if width_rect < height_rect:
                        # El ancho es horizontal
                        x_offset = int(width_rect / 2)
                        pt1 = (center[0] - x_offset, center[1])
                        pt2 = (center[0] + x_offset, center[1])
                    else:
                        # El ancho es vertical
                        y_offset = int(height_rect / 2)
                        pt1 = (center[0], center[1] - y_offset)
                        pt2 = (center[0], center[1] + y_offset)
                    cv2.line(vis_image, pt1, pt2, (255, 255, 0), 3)
                    
                    # Texto con medición (MÁS GRANDE y en posición fija)
                    cv2.putText(vis_image, f"ANCHO: {ancho_mm:.2f}mm", 
                               (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                else:
                    self.miss_count += 1
                    if self.last_ancho is not None and self.miss_count <= ConfigSistema.HOLD_LAST_N_MISSES:
                        return self.last_ancho, vis_image

            return ancho_mm, vis_image
            
        except Exception as e:
            print(f"Error procesando frame: {e}")
            return None, None

# Clase SistemaVisualizacion eliminada - ya no se necesita

# ===========================
# MAIN
# ===========================
def main():
    print("=" * 70)
    print(" Sistema de Monitoreo y Metrología - OPTIMIZADO")
    print(" Jetson Nano + Orbbec Femto Bolt + VL53L1X ToF")
    print("=" * 70)
    
    # Entrada de parámetros
    try:
        ancho_objetivo = float(input("Ancho objetivo del filamento [mm]: "))
        velocidad_base = float(input("Velocidad base de referencia [mm/s]: "))
    except ValueError:
        print("Error: valores no válidos")
        return
    
    # Inicializar componentes
    sensor_tof = SensorToF()
    sensor_tof.inicializar()
    
    medidor0 = MedidorAncho()
    medidor1 = MedidorAncho()
    
    # Variables
    pipeline0 = None
    pipeline1 = None
    
    if not ORBBEC_DISPONIBLE:
        print("ERROR: pyorbbecsdk no disponible")
        return
    
    print("\n[1/3] Conectando con cámaras Orbbec...")
    
    try:
        ctx = Context()
        device_list = ctx.query_devices()
        curr_device_cnt = device_list.get_count()

        if curr_device_cnt == 0:
            print("ERROR: No se detectaron cámaras")
            print("\nSoluciones:")
            print("  1. lsusb | grep -i orbbec")
            print("  2. Desconectar y reconectar USB")
            print("  3. sudo chmod 666 /dev/video*")
            return
        
        print(f"✓ Detectadas {curr_device_cnt} cámara(s)")

        # Esperar un momento antes de abrir
        time.sleep(0.5)
        
        try:
            device0 = device_list.get_device_by_index(0)
            print(f"  Cámara 0: {device0.get_device_info().get_name()}")
        except Exception as e:
            print(f"\n✗ Error abriendo cámara: {e}")
            print("\nCámara ocupada o sin permisos.")
            print("Ejecuta: sudo chmod 666 /dev/video*")
            print("O cierra otros programas que usen la cámara")
            return
        
        # Segunda cámara si está disponible
        device1 = None
        if curr_device_cnt >= 2:
            try:
                device1 = device_list.get_device_by_index(1)
                print(f"  Cámara 1: {device1.get_device_info().get_name()}")
            except Exception as e:
                print(f"  ⚠ No se pudo abrir cámara 1: {e}")
                device1 = None

        print("\n[2/3] Iniciando pipeline...")
        
        # Crear pipelines
        pipeline0 = Pipeline(device0)
        
        # Pipeline para segunda cámara si está disponible
        if device1 is not None:
            pipeline1 = Pipeline(device1)
        else:
            pipeline1 = None
        
        # Intentar configurar resolución menor para mejor rendimiento
        def configurar_camara(pipeline, nombre):
            """Intenta configurar la cámara a resolución baja"""
            config = Config()
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            
            # Intentar diferentes resoluciones (de menor a mayor)
            resoluciones = [
                (640, 480, 30),   # 480p - más rápido
                (640, 480, 15),
                (1280, 720, 30),  # 720p
                (1280, 720, 15),
            ]
            
            perfil_elegido = None
            for w, h, fps in resoluciones:
                try:
                    # Intentar MJPEG primero (OBFormat.MJPG)
                    perfil = profile_list.get_video_stream_profile(w, h, OBFormat.MJPG, fps)
                    perfil_elegido = perfil
                    print(f"  {nombre}: Encontrado {w}x{h} @ {fps}fps MJPEG")
                    break
                except:
                    pass
                
                try:
                    # Intentar cualquier formato
                    perfil = profile_list.get_video_stream_profile(w, 0, OBFormat.ANY, fps)
                    perfil_elegido = perfil
                    print(f"  {nombre}: Encontrado {w}x? @ {fps}fps")
                    break
                except:
                    pass
            
            if perfil_elegido:
                config.enable_stream(perfil_elegido)
                pipeline.start(config)
                print(f"✓ {nombre}: {perfil_elegido.get_width()}x{perfil_elegido.get_height()} @ {perfil_elegido.get_fps()}fps")
                return True
            else:
                # Usar default
                pipeline.start()
                print(f"✓ {nombre}: usando configuración por defecto")
                return False
        
        configurar_camara(pipeline0, "Cam0")
        
        if pipeline1:
            # Delay para evitar conflicto uvc_open failed
            time.sleep(0.5)
            print("  Esperando antes de iniciar segunda cámara...")
            configurar_camara(pipeline1, "Cam1")
        
        # Esperar estabilización
        time.sleep(1.0)
        
        print("\n[3/3] Sistema iniciado correctamente")
        print("=" * 70)
        print("CONTROLES:")
        print("  'q' - Salir")
        print("  '+' - Aumentar umbral de binarización")
        print("  '-' - Disminuir umbral de binarización")
        print("=" * 70 + "\n")
        
        # Variables de control
        frame_count = 0
        start_time = time.time()
        last_print_time = time.time()
        
        # Variables compartidas para threading
        result0 = {'ancho': None, 'image': None, 'ts': 0.0}
        result1 = {'ancho': None, 'image': None, 'ts': 0.0}
        lock0 = threading.Lock()
        lock1 = threading.Lock()
        stop_event = threading.Event()
        
        def worker_cam0():
            """Hilo persistente para cámara 0 (siempre el último frame)"""
            while not stop_event.is_set():
                try:
                    frameset = pipeline0.wait_for_frames(50)
                    if not frameset:
                        continue
                    color_frame = frameset.get_color_frame()
                    depth_frame = frameset.get_depth_frame()
                    if not (color_frame or depth_frame):
                        continue
                    ancho, vis = medidor0.procesar_frame(depth_frame, color_frame)
                    with lock0:
                        result0['ancho'] = ancho
                        result0['image'] = vis
                        result0['ts'] = time.time()
                except:
                    continue
        
        def worker_cam1():
            """Hilo persistente para cámara 1"""
            if not pipeline1:
                return
            while not stop_event.is_set():
                try:
                    frameset = pipeline1.wait_for_frames(50)
                    if not frameset:
                        continue
                    color_frame = frameset.get_color_frame()
                    depth_frame = frameset.get_depth_frame()
                    if not (color_frame or depth_frame):
                        continue
                    ancho, vis = medidor1.procesar_frame(depth_frame, color_frame)
                    with lock1:
                        result1['ancho'] = ancho
                        result1['image'] = vis
                        result1['ts'] = time.time()
                except:
                    continue

        # Lanzar hilos persistentes (evita overhead de crear threads por frame)
        th0 = threading.Thread(target=worker_cam0, daemon=True)
        th0.start()
        th1 = None
        if pipeline1:
            th1 = threading.Thread(target=worker_cam1, daemon=True)
            th1.start()
        
        # ===========================
        # CREAR VENTANAS (Fix para OpenGL en Jetson)
        # ===========================
        cv2.namedWindow("Sistema de Medicion - Vista Dual", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Sistema de Medicion", cv2.WINDOW_NORMAL)
        
        # ===========================
        # LOOP PRINCIPAL CON THREADING
        # ===========================
        while True:
            frame_count += 1
            loop_start = time.time()
            
            # Leer distancia ToF
            distancia_tof = sensor_tof.leer_distancia()
            
            # Obtener el ÚLTIMO resultado disponible (sin bloquear)
            with lock0:
                ancho0 = result0['ancho']
                vis_image0 = result0['image']
            with lock1:
                ancho1 = result1['ancho']
                vis_image1 = result1['image']
            
            # Calcular FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Visualización lado a lado - TAMAÑO FIJO 640x480 por cámara
            DISPLAY_W = 640
            DISPLAY_H = 480
            
            if vis_image0 is not None or vis_image1 is not None:
                # SIEMPRE redimensionar a tamaño fijo
                if vis_image0 is not None:
                    vis_image0 = cv2.resize(vis_image0, (DISPLAY_W, DISPLAY_H))
                    cv2.putText(vis_image0, "Camara 1", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if ancho0 is not None:
                        cv2.putText(vis_image0, f"{ancho0:.2f}mm", (10, DISPLAY_H - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if vis_image1 is not None:
                    vis_image1 = cv2.resize(vis_image1, (DISPLAY_W, DISPLAY_H))
                    cv2.putText(vis_image1, "Camara 2", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if ancho1 is not None:
                        cv2.putText(vis_image1, f"{ancho1:.2f}mm", (10, DISPLAY_H - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Mostrar según disponibilidad
                if vis_image0 is not None and vis_image1 is not None and pipeline1:
                    # Ambas cámaras: lado a lado (mismo tamaño garantizado)
                    combined = np.hstack((vis_image0, vis_image1))
                    
                    # Línea separadora vertical
                    cv2.line(combined, (DISPLAY_W, 0), (DISPLAY_W, DISPLAY_H), (0, 255, 255), 2)
                    
                    # Barra superior con info
                    cv2.rectangle(combined, (0, 0), (DISPLAY_W * 2, 50), (0, 0, 0), -1)
                    
                    # FPS
                    cv2.putText(combined, f"FPS: {fps:.1f}", (20, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Promedio si ambas cámaras tienen medición
                    if ancho0 is not None and ancho1 is not None:
                        ancho_prom = (ancho0 + ancho1) / 2
                        error = ancho_prom - ancho_objetivo
                        
                        # Color según error
                        if abs(error) <= ConfigSistema.TOLERANCIA_ERROR:
                            color = (0, 255, 0)  # Verde
                            estado = "OK"
                        elif error > 0:
                            color = (0, 165, 255)  # Naranja
                            estado = "ANCHO"
                        else:
                            color = (0, 0, 255)  # Rojo
                            estado = "DELGADO"
                        
                        cv2.putText(combined, f"Prom: {ancho_prom:.2f}mm ({estado})", 
                                   (DISPLAY_W - 130, 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Objetivo
                    cv2.putText(combined, f"Obj: {ancho_objetivo:.2f}mm", 
                               (DISPLAY_W * 2 - 180, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow("Sistema de Medicion - Vista Dual", combined)
                    
                elif vis_image0 is not None:
                    # Solo cámara 0
                    cv2.putText(vis_image0, f"FPS: {fps:.1f}", (DISPLAY_W - 120, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow("Sistema de Medicion", vis_image0)
                    
                elif vis_image1 is not None:
                    # Solo cámara 1
                    cv2.putText(vis_image1, f"FPS: {fps:.1f}", (DISPLAY_W - 120, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow("Sistema de Medicion", vis_image1)
            
            # Imprimir estadísticas cada 2 segundos
            if time.time() - last_print_time >= 2.0:
                print(f"\n--- {datetime.now().strftime('%H:%M:%S')} | Frame {frame_count} | FPS: {fps:.1f} ---")
                
                if distancia_tof:
                    print(f"  ToF: {distancia_tof:.1f} cm")
                
                if ancho0:
                    error0 = ancho0 - ancho_objetivo
                    print(f"  Cam0: {ancho0:.2f} mm (error: {error0:+.2f} mm)")
                
                if ancho1:
                    error1 = ancho1 - ancho_objetivo
                    print(f"  Cam1: {ancho1:.2f} mm (error: {error1:+.2f} mm)")
                
                # Promedio si ambas cámaras tienen medición
                if ancho0 and ancho1:
                    ancho_prom = (ancho0 + ancho1) / 2
                    error_prom = ancho_prom - ancho_objetivo
                    print(f"  Promedio: {ancho_prom:.2f} mm (error: {error_prom:+.2f} mm)")
                    
                    # Recomendación
                    if abs(error_prom) <= ConfigSistema.TOLERANCIA_ERROR:
                        print(f"  Estado: ✓ OPTIMO")
                    elif error_prom > 0:
                        porcentaje = (error_prom / ancho_objetivo) * 100
                        print(f"  Estado: ⚠ MUY ANCHO (+{porcentaje:.1f}%) - ACELERAR")
                    else:
                        porcentaje = (abs(error_prom) / ancho_objetivo) * 100
                        print(f"  Estado: ⚠ MUY DELGADO (-{porcentaje:.1f}%) - FRENAR")
                
                last_print_time = time.time()
            
            # Manejo de teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[SALIENDO] Deteniendo sistema...")
                stop_event.set()
                break
            elif key == ord('+'):
                medidor0.umbral = min(255, medidor0.umbral + 5)
                medidor1.umbral = medidor0.umbral
                print(f"Umbral: {medidor0.umbral}")
            elif key == ord('-'):
                medidor0.umbral = max(0, medidor0.umbral - 5)
                medidor1.umbral = medidor0.umbral
                print(f"Umbral: {medidor0.umbral}")
            
            # Control de framerate
            loop_time = time.time() - loop_start
            if loop_time < 1.0 / ConfigSistema.CAMERA_FPS:
                time.sleep(1.0 / ConfigSistema.CAMERA_FPS - loop_time)
    
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] {e}")
        import traceback
        traceback.print_exc()
        print("\nSugerencias:")
        print("  1. Verifica conexiones USB: lsusb | grep -i orbbec")
        print("  2. Verifica permisos: groups | grep video")
        print("  3. Ejecuta: sudo jetson_clocks")
        print("  4. Verifica RAM: free -h")
        print("  5. Buffer USB: cat /sys/module/usbcore/parameters/usbfs_memory_mb")
    
    finally:
        # Cleanup
        print("\n[LIMPIEZA] Cerrando recursos...")
        try:
            stop_event.set()
        except:
            pass
        if pipeline0:
            try:
                pipeline0.stop()
                print("✓ Pipeline 0 cerrado")
            except:
                pass
        if pipeline1:
            try:
                pipeline1.stop()
                print("✓ Pipeline 1 cerrado")
            except:
                pass
        sensor_tof.cerrar()
        cv2.destroyAllWindows()
        print("✓ Sistema cerrado correctamente")

if __name__ == "__main__":
    main()
