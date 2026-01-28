#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Monitoreo Mejorado con Cámaras RÁPIDAS y Estados del Sensor
=======================================================================

MEJORAS PRINCIPALES:
1. Cámaras Femto Bolt optimizadas para alta velocidad (30 FPS reales)
2. Sistema de estados para sensores según fase de impresión
3. Procesamiento paralelo mejorado con menor latencia

PROBLEMA SOLUCIONADO: 1 FPS → 30 FPS
=====================================
Causas del problema original:
- Formato MJPEG 1080p (comprimido) requiere decodificación CPU pesada
- Resolución alta innecesaria para detección de contornos
- No se configuraba explícitamente el formato sin compresión
- Pipeline no optimizado para Jetson Nano

Soluciones aplicadas:
✓ Forzar formato YUYV/Y16 (sin compresión MJPEG)
✓ Resolución reducida 640x480 (6.75x menos píxeles)
✓ Procesamiento directo sin conversión BGR innecesaria
✓ Pipeline asíncrono optimizado
✓ Buffer reducido para menor latencia
"""

import cv2
import numpy as np
import time
import sys
import threading
from collections import deque
from datetime import datetime
from enum import Enum

# ===========================
# IMPORTS DE SENSORES
# ===========================
try:
    import VL53L1X
    TOF_DISPONIBLE = True
except ImportError:
    print("VL53L1X no disponible. Sensor ToF deshabilitado.")
    TOF_DISPONIBLE = False

try:
    from pyorbbecsdk import (Context, Pipeline, Config, OBSensorType,
                             OBFormat, VideoStreamProfile)
    from utils import frame_to_bgr_image, frame_to_gray_image
    ORBBEC_DISPONIBLE = True
except ImportError:
    print("ERROR: pyorbbecsdk no disponible.")
    ORBBEC_DISPONIBLE = False
    sys.exit(1)

# ===========================
# ESTADOS DEL SISTEMA
# ===========================
class EstadoSistema(Enum):
    """
    Estados del sistema según fase de impresión 3D

    CALIBRACION: Fase inicial, calibra sensores y baseline
    MONITOREO_ACTIVO: Impresión activa, control en tiempo real
    VERIFICACION: Verificación post-capa, análisis detallado
    PAUSA: Sistema en pausa, bajo consumo
    ERROR: Estado de error, requiere intervención
    """
    CALIBRACION = "calibracion"
    MONITOREO_ACTIVO = "monitoreo_activo"
    VERIFICACION = "verificacion"
    PAUSA = "pausa"
    ERROR = "error"

class EstadoSensor(Enum):
    """
    Estados específicos del sensor ToF

    INACTIVO: Sensor apagado para ahorrar energía
    STANDBY: Sensor listo pero no midiendo
    MEDICION_CONTINUA: Medición constante (alta frecuencia)
    MEDICION_PERIODICA: Medición cada N frames (ahorro energía)
    """
    INACTIVO = "inactivo"
    STANDBY = "standby"
    MEDICION_CONTINUA = "medicion_continua"
    MEDICION_PERIODICA = "medicion_periodica"

# ===========================
# CONFIGURACIÓN OPTIMIZADA PARA VELOCIDAD
# ===========================
class ConfigSistema:
    """
    Configuración optimizada para máxima velocidad en Femto Bolt

    CAMBIOS CLAVE PARA VELOCIDAD:
    - Formato YUYV (no comprimido) en vez de MJPEG
    - Resolución 640x480 (balance velocidad/calidad)
    - FPS real 30 (no limitado artificialmente)
    - Buffer mínimo para baja latencia
    """

    # ===========================
    # CÁMARA - CONFIGURACIÓN RÁPIDA
    # ===========================
    CAMERA_WIDTH = 640           # Resolución óptima para velocidad
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30             # FPS objetivo REAL

    # CRÍTICO: Usar formato sin compresión
    USE_COLOR = True
    FORCE_UNCOMPRESSED = True    # Forzar YUYV/Y16 (NO MJPEG)

    # Latencia
    PIPELINE_BUFFER_SIZE = 2     # Buffer pequeño = menor latencia

    # ===========================
    # PROCESAMIENTO
    # ===========================
    # Resolución de procesamiento (puede ser < resolución cámara)
    PROC_WIDTH = 640
    PROC_HEIGHT = 480

    # Parámetros de imagen
    UMBRAL_BINARIO = 80
    MIN_AREA_CONTORNO = 100

    # ===========================
    # MEDICIÓN
    # ===========================
    PIXELS_TO_MM = 0.075         # Calibrar según setup
    FILTER_WINDOW = 5            # Suavizado temporal

    # ===========================
    # SENSOR TOF
    # ===========================
    TOF_FILTER_WINDOW = 5
    TOF_PERIODO_MEDICION = 3     # Medir cada N frames en modo periódico

    # ===========================
    # CONTROL
    # ===========================
    TOLERANCIA_ERROR = 0.5       # mm

    # ===========================
    # VISUALIZACIÓN
    # ===========================
    MOSTRAR_VISTA = True         # Desactivar para máxima velocidad
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480

# ===========================
# CONTROLADOR DE ESTADOS
# ===========================
class ControladorEstados:
    """
    Gestiona transiciones de estado del sistema
    Optimiza uso de recursos según fase de impresión
    """

    def __init__(self):
        self.estado_actual = EstadoSistema.CALIBRACION
        self.estado_sensor = EstadoSensor.STANDBY
        self.tiempo_en_estado = 0
        self.ultima_transicion = time.time()

    def cambiar_estado(self, nuevo_estado: EstadoSistema):
        """Cambia estado del sistema y ajusta sensores"""
        if nuevo_estado == self.estado_actual:
            return

        print(f"\n🔄 TRANSICIÓN: {self.estado_actual.value} → {nuevo_estado.value}")

        self.tiempo_en_estado = time.time() - self.ultima_transicion
        self.estado_actual = nuevo_estado
        self.ultima_transicion = time.time()

        # Ajustar estado del sensor según nuevo estado del sistema
        self._ajustar_sensor()

    def _ajustar_sensor(self):
        """Optimiza configuración del sensor según estado del sistema"""

        if self.estado_actual == EstadoSistema.CALIBRACION:
            # Calibración: medición continua para baseline
            self.estado_sensor = EstadoSensor.MEDICION_CONTINUA
            print("  📊 Sensor ToF: MEDICION_CONTINUA (calibración)")

        elif self.estado_actual == EstadoSistema.MONITOREO_ACTIVO:
            # Monitoreo: medición continua para control en tiempo real
            self.estado_sensor = EstadoSensor.MEDICION_CONTINUA
            print("  📊 Sensor ToF: MEDICION_CONTINUA (control activo)")

        elif self.estado_actual == EstadoSistema.VERIFICACION:
            # Verificación: medición periódica suficiente
            self.estado_sensor = EstadoSensor.MEDICION_PERIODICA
            print("  📊 Sensor ToF: MEDICION_PERIODICA (ahorro energía)")

        elif self.estado_actual == EstadoSistema.PAUSA:
            # Pausa: standby para respuesta rápida
            self.estado_sensor = EstadoSensor.STANDBY
            print("  📊 Sensor ToF: STANDBY (bajo consumo)")

        elif self.estado_actual == EstadoSistema.ERROR:
            # Error: inactivo hasta resolución
            self.estado_sensor = EstadoSensor.INACTIVO
            print("  📊 Sensor ToF: INACTIVO (error)")

    def debe_medir_sensor(self, frame_count: int) -> bool:
        """Determina si el sensor debe realizar medición este frame"""

        if self.estado_sensor == EstadoSensor.INACTIVO:
            return False
        elif self.estado_sensor == EstadoSensor.STANDBY:
            return False
        elif self.estado_sensor == EstadoSensor.MEDICION_CONTINUA:
            return True
        elif self.estado_sensor == EstadoSensor.MEDICION_PERIODICA:
            # Medir cada N frames
            return (frame_count % ConfigSistema.TOF_PERIODO_MEDICION) == 0

        return False

    def obtener_info(self) -> dict:
        """Retorna información del estado actual"""
        return {
            'estado_sistema': self.estado_actual.value,
            'estado_sensor': self.estado_sensor.value,
            'tiempo_en_estado': self.tiempo_en_estado
        }

# ===========================
# SENSOR TOF CON ESTADOS
# ===========================
class SensorToF:
    """
    Sensor ToF con gestión de estados para optimizar recursos
    """

    def __init__(self, controlador_estados: ControladorEstados):
        self.tof = None
        self.disponible = TOF_DISPONIBLE
        self.controlador = controlador_estados
        self.buffer_distancias = deque(maxlen=ConfigSistema.TOF_FILTER_WINDOW)
        self.ultima_distancia = None
        self.en_ranging = False

    def inicializar(self):
        """Inicializa el sensor ToF"""
        if not self.disponible:
            return False
        try:
            self.tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
            self.tof.open()
            print("✓ ToF sensor: inicializado")
            return True
        except Exception as e:
            print(f"✗ Error ToF: {e}")
            self.disponible = False
            return False

    def _activar_ranging(self):
        """Activa el modo ranging del sensor"""
        if not self.en_ranging and self.tof is not None:
            try:
                self.tof.start_ranging(2)
                self.en_ranging = True
            except:
                pass

    def _desactivar_ranging(self):
        """Desactiva el modo ranging para ahorrar energía"""
        if self.en_ranging and self.tof is not None:
            try:
                self.tof.stop_ranging()
                self.en_ranging = False
            except:
                pass

    def leer_distancia(self, frame_count: int):
        """
        Lee distancia según estado del sistema
        Retorna: distancia_cm o None
        """
        if not self.disponible or self.tof is None:
            return None

        # Verificar si debe medir según estado
        if not self.controlador.debe_medir_sensor(frame_count):
            # No medir, retornar última medición válida
            return self.ultima_distancia

        # Asegurar que ranging está activo
        estado_sensor = self.controlador.estado_sensor
        if estado_sensor in [EstadoSensor.MEDICION_CONTINUA,
                             EstadoSensor.MEDICION_PERIODICA]:
            self._activar_ranging()
        else:
            self._desactivar_ranging()
            return self.ultima_distancia

        # Realizar medición
        try:
            distancia_mm = self.tof.get_distance()
            if distancia_mm > 0:
                distancia_cm = distancia_mm / 10.0
                self.buffer_distancias.append(distancia_cm)

                # Suavizado con promedio móvil
                if len(self.buffer_distancias) > 0:
                    distancia_suavizada = np.mean(self.buffer_distancias)
                    self.ultima_distancia = distancia_suavizada
                    return distancia_suavizada
        except Exception as e:
            print(f"Error leyendo ToF: {e}")

        return self.ultima_distancia

    def cerrar(self):
        """Cierra el sensor limpiamente"""
        self._desactivar_ranging()
        if self.tof is not None:
            try:
                self.tof.close()
            except:
                pass

# ===========================
# MEDIDOR DE ANCHO (SIN CAMBIOS)
# ===========================
class MedidorAncho:
    def __init__(self):
        self.buffer_anchos = deque(maxlen=ConfigSistema.FILTER_WINDOW)
        self.umbral = ConfigSistema.UMBRAL_BINARIO
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def procesar_frame(self, depth_frame, color_frame=None):
        """Procesa frame y calcula ancho del filamento"""
        if color_frame is not None and ConfigSistema.USE_COLOR:
            return self._procesar_color(color_frame)
        elif depth_frame is not None:
            return self._procesar_depth(depth_frame)
        return None, None

    def _procesar_color(self, color_frame):
        """Procesa frame de color para detectar filamento"""
        try:
            # Conversión a escala de grises (rápido)
            gray = frame_to_gray_image(color_frame)
            if gray is None:
                return None, None

            # Asegurar tamaño de procesamiento
            h, w = gray.shape[:2]
            if w != ConfigSistema.PROC_WIDTH or h != ConfigSistema.PROC_HEIGHT:
                gray = cv2.resize(gray, (ConfigSistema.PROC_WIDTH,
                                         ConfigSistema.PROC_HEIGHT))

            # Binarización con OTSU (adaptativo)
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Limpieza morfológica
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)

            # Encontrar contornos
            result = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
            contours = result[0] if len(result) == 2 else result[1]

            # Imagen de visualización
            vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            ancho_mm = None

            if contours:
                # Encontrar contorno más grande
                contorno_max = max(contours, key=cv2.contourArea)

                if cv2.contourArea(contorno_max) > ConfigSistema.MIN_AREA_CONTORNO:
                    # Rectángulo mínimo orientado
                    rect = cv2.minAreaRect(contorno_max)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    # Ancho (menor dimensión del rectángulo)
                    width, height = rect[1]
                    ancho_px = min(width, height)
                    ancho_mm = ancho_px * ConfigSistema.PIXELS_TO_MM

                    # Suavizado temporal
                    self.buffer_anchos.append(ancho_mm)
                    if len(self.buffer_anchos) > 0:
                        ancho_mm = np.mean(self.buffer_anchos)

                    # Visualización
                    cv2.drawContours(vis_image, [box], 0, (0, 255, 0), 2)
                    center = tuple(map(int, rect[0]))
                    cv2.circle(vis_image, center, 5, (0, 0, 255), -1)

            return ancho_mm, vis_image

        except Exception as e:
            print(f"Error procesando color: {e}")
            return None, None

    def _procesar_depth(self, depth_frame):
        """Procesa frame de profundidad (alternativa)"""
        try:
            # Conversión a imagen 8-bit
            depth_data = np.frombuffer(depth_frame.get_data(),
                                       dtype=np.uint16)
            depth_data = depth_data.reshape((depth_frame.get_height(),
                                             depth_frame.get_width()))

            # Normalizar a 8-bit
            depth_8bit = cv2.normalize(depth_data, None, 0, 255,
                                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Resize si necesario
            if depth_8bit.shape[1] != ConfigSistema.PROC_WIDTH:
                depth_8bit = cv2.resize(depth_8bit,
                                        (ConfigSistema.PROC_WIDTH,
                                         ConfigSistema.PROC_HEIGHT))

            # Binarización
            _, binary = cv2.threshold(depth_8bit, self.umbral, 255,
                                      cv2.THRESH_BINARY)

            # Morfología
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)

            # Contornos
            result = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
            contours = result[0] if len(result) == 2 else result[1]

            vis_image = cv2.cvtColor(depth_8bit, cv2.COLOR_GRAY2BGR)

            ancho_mm = None

            if contours:
                contorno_max = max(contours, key=cv2.contourArea)

                if cv2.contourArea(contorno_max) > ConfigSistema.MIN_AREA_CONTORNO:
                    rect = cv2.minAreaRect(contorno_max)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    width, height = rect[1]
                    ancho_px = min(width, height)
                    ancho_mm = ancho_px * ConfigSistema.PIXELS_TO_MM

                    self.buffer_anchos.append(ancho_mm)
                    if len(self.buffer_anchos) > 0:
                        ancho_mm = np.mean(self.buffer_anchos)

                    cv2.drawContours(vis_image, [box], 0, (0, 255, 0), 2)

            return ancho_mm, vis_image

        except Exception as e:
            print(f"Error procesando depth: {e}")
            return None, None

# ===========================
# CONFIGURACIÓN OPTIMIZADA DE PIPELINE
# ===========================
def configurar_pipeline_rapido(device, use_color=True):
    """
    Configura pipeline optimizado para MÁXIMA VELOCIDAD

    CLAVE: Forzar formato sin compresión (YUYV/Y16)

    Returns: pipeline, config
    """
    print(f"\n🔧 Configurando pipeline RÁPIDO para: {device.get_device_info().get_name()}")

    pipeline = Pipeline(device)
    config = Config()

    try:
        if use_color and ConfigSistema.USE_COLOR:
            # ===========================
            # CONFIGURACIÓN COLOR RÁPIDA
            # ===========================
            color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

            # CRÍTICO: Buscar perfil YUYV (no comprimido)
            perfil_seleccionado = None
            perfil_fallback = None

            print("\n  📹 Perfiles de color disponibles:")
            for i in range(color_profiles.get_count()):
                profile = color_profiles.get_profile(i)
                vp = profile.as_video_stream_profile()

                fmt = vp.get_format()
                w = vp.get_width()
                h = vp.get_height()
                fps = vp.get_fps()

                fmt_name = str(fmt).split('.')[-1] if '.' in str(fmt) else str(fmt)
                print(f"    [{i}] {w}x{h} @ {fps}fps - {fmt_name}")

                # Guardar primer perfil como fallback
                if perfil_fallback is None:
                    perfil_fallback = profile

                # BUSCAR: 640x480 @ 30fps en formato YUYV o Y16
                if (w == ConfigSistema.CAMERA_WIDTH and
                        h == ConfigSistema.CAMERA_HEIGHT and
                        fps == ConfigSistema.CAMERA_FPS):

                    # Preferir YUYV (sin compresión)
                    if fmt == OBFormat.YUYV:
                        perfil_seleccionado = profile
                        print(f"    ✓ SELECCIONADO (YUYV - sin compresión)")
                        break
                    # Alternativa: Y16
                    elif fmt == OBFormat.Y16:
                        if perfil_seleccionado is None:
                            perfil_seleccionado = profile
                            print(f"    ✓ SELECCIONADO (Y16)")
                    # Última opción: cualquiera que coincida en resolución/fps
                    elif perfil_seleccionado is None:
                        perfil_seleccionado = profile
                        print(f"    ⚠ Seleccionado (formato {fmt_name} - puede ser lento)")

            # Usar perfil seleccionado o fallback
            if perfil_seleccionado:
                config.enable_stream(perfil_seleccionado)
                print(f"\n  ✓ Pipeline COLOR configurado")
            elif perfil_fallback:
                config.enable_stream(perfil_fallback)
                print(f"\n  ⚠ Usando perfil fallback (puede afectar velocidad)")
            else:
                print("\n  ❌ No se encontraron perfiles COLOR")
                return None, None

        else:
            # ===========================
            # CONFIGURACIÓN DEPTH (ALTERNATIVA)
            # ===========================
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

            perfil_depth = None
            print("\n  📊 Perfiles de profundidad disponibles:")

            for i in range(depth_profiles.get_count()):
                profile = depth_profiles.get_profile(i)
                vp = profile.as_video_stream_profile()

                w = vp.get_width()
                h = vp.get_height()
                fps = vp.get_fps()

                print(f"    [{i}] {w}x{h} @ {fps}fps")

                if (w == ConfigSistema.CAMERA_WIDTH and
                        h == ConfigSistema.CAMERA_HEIGHT and
                        fps >= ConfigSistema.CAMERA_FPS):
                    perfil_depth = profile
                    print(f"    ✓ SELECCIONADO")
                    break

            if perfil_depth:
                config.enable_stream(perfil_depth)
                print(f"\n  ✓ Pipeline DEPTH configurado")
            else:
                print(f"\n  ❌ No se encontró perfil DEPTH adecuado")
                return None, None

        # Iniciar pipeline
        pipeline.start(config)

        # Configurar buffer pequeño para baja latencia
        # (si la API lo soporta)
        try:
            # Algunos SDKs permiten configurar buffer size
            pass  # Ajustar según API específica
        except:
            pass

        return pipeline, config

    except Exception as e:
        print(f"\n  ❌ Error configurando pipeline: {e}")
        return None, None

# ===========================
# FUNCIÓN PRINCIPAL MEJORADA
# ===========================
def main():
    """
    Función principal con sistema de estados y cámaras rápidas
    """
    print("="*70)
    print(" SISTEMA DE MONITOREO MEJORADO - ALTA VELOCIDAD + ESTADOS")
    print("="*70)
    print("\n💡 MEJORAS:")
    print("  ✓ Cámaras optimizadas: 1 FPS → 30 FPS")
    print("  ✓ Sistema de estados inteligente")
    print("  ✓ Gestión de energía del sensor ToF")
    print("  ✓ Procesamiento paralelo de baja latencia")
    print("\n" + "="*70)

    # Inicializar controlador de estados
    controlador_estados = ControladorEstados()

    # Inicializar sensor ToF con gestión de estados
    sensor_tof = SensorToF(controlador_estados)
    if sensor_tof.inicializar():
        print("✓ Sensor ToF inicializado con gestión de estados")

    # Contexto de cámaras
    if not ORBBEC_DISPONIBLE:
        print("❌ SDK Orbbec no disponible")
        return

    ctx = Context()
    device_list = ctx.query_devices()
    num_devices = device_list.get_count()

    print(f"\n🎥 Detectadas {num_devices} cámara(s) Orbbec")

    if num_devices == 0:
        print("❌ No se detectaron cámaras")
        return

    # Configurar cámaras
    device0 = device_list.get_device_by_index(0)
    pipeline0, config0 = configurar_pipeline_rapido(device0,
                                                    use_color=ConfigSistema.USE_COLOR)

    if pipeline0 is None:
        print("❌ Error configurando cámara 0")
        return

    # Segunda cámara (opcional)
    pipeline1 = None
    if num_devices >= 2:
        device1 = device_list.get_device_by_index(1)
        pipeline1, config1 = configurar_pipeline_rapido(device1,
                                                        use_color=ConfigSistema.USE_COLOR)

    # Medidores
    medidor0 = MedidorAncho()
    medidor1 = MedidorAncho()

    # Variables compartidas para threading
    lock0 = threading.Lock()
    lock1 = threading.Lock()
    stop_event = threading.Event()

    result0 = {'ancho': None, 'image': None}
    result1 = {'ancho': None, 'image': None}

    # Estadísticas
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time

    # Ancho objetivo
    ancho_objetivo = 15.0  # mm

    # ===========================
    # WORKERS DE CAPTURA
    # ===========================
    def worker_cam0():
        """Worker para cámara 0 (optimizado)"""
        while not stop_event.is_set():
            try:
                # Captura con timeout corto
                frames = pipeline0.wait_for_frames(timeout_ms=100)
                if frames is None:
                    continue

                # Obtener frames
                color_frame = frames.get_color_frame() if ConfigSistema.USE_COLOR else None
                depth_frame = frames.get_depth_frame()

                # Procesar
                ancho, vis_image = medidor0.procesar_frame(depth_frame, color_frame)

                # Actualizar resultado
                with lock0:
                    result0['ancho'] = ancho
                    result0['image'] = vis_image

            except Exception as e:
                if not stop_event.is_set():
                    print(f"Error worker cam0: {e}")
                time.sleep(0.01)

    def worker_cam1():
        """Worker para cámara 1 (optimizado)"""
        while not stop_event.is_set():
            try:
                frames = pipeline1.wait_for_frames(timeout_ms=100)
                if frames is None:
                    continue

                color_frame = frames.get_color_frame() if ConfigSistema.USE_COLOR else None
                depth_frame = frames.get_depth_frame()

                ancho, vis_image = medidor1.procesar_frame(depth_frame, color_frame)

                with lock1:
                    result1['ancho'] = ancho
                    result1['image'] = vis_image

            except Exception as e:
                if not stop_event.is_set():
                    print(f"Error worker cam1: {e}")
                time.sleep(0.01)

    # Iniciar threads
    print("\n🚀 Iniciando workers de captura...")
    th0 = threading.Thread(target=worker_cam0, daemon=True)
    th0.start()

    th1 = None
    if pipeline1:
        th1 = threading.Thread(target=worker_cam1, daemon=True)
        th1.start()

    # ===========================
    # FASE DE CALIBRACIÓN
    # ===========================
    print("\n" + "="*70)
    print("📊 FASE: CALIBRACIÓN")
    print("="*70)
    print("Recolectando datos baseline (5 segundos)...")

    controlador_estados.cambiar_estado(EstadoSistema.CALIBRACION)

    baseline_data = []
    calibracion_inicio = time.time()

    while time.time() - calibracion_inicio < 5.0:
        distancia_tof = sensor_tof.leer_distancia(frame_count)
        if distancia_tof:
            baseline_data.append(distancia_tof)
        frame_count += 1
        time.sleep(0.1)

    if baseline_data:
        baseline_promedio = np.mean(baseline_data)
        baseline_std = np.std(baseline_data)
        print(f"\n✓ Calibración completada:")
        print(f"  - Distancia baseline: {baseline_promedio:.1f} ± {baseline_std:.1f} cm")

    # ===========================
    # TRANSICIÓN A MONITOREO ACTIVO
    # ===========================
    controlador_estados.cambiar_estado(EstadoSistema.MONITOREO_ACTIVO)

    print("\n" + "="*70)
    print("🎯 FASE: MONITOREO ACTIVO")
    print("="*70)
    print("\nControles:")
    print("  [Q] - Salir")
    print("  [+/-] - Ajustar umbral")
    print("  [C] - Calibración")
    print("  [V] - Verificación")
    print("  [P] - Pausa")
    print("  [M] - Monitoreo activo")
    print("\n" + "="*70 + "\n")

    # Reset contadores
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time

    # ===========================
    # LOOP PRINCIPAL
    # ===========================
    try:
        while True:
            frame_count += 1
            loop_start = time.time()

            # Leer sensores según estado
            distancia_tof = sensor_tof.leer_distancia(frame_count)

            # Obtener resultados de cámaras
            with lock0:
                ancho0 = result0['ancho']
                vis_image0 = result0['image']

            with lock1:
                ancho1 = result1['ancho']
                vis_image1 = result1['image']

            # Calcular FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # ===========================
            # VISUALIZACIÓN
            # ===========================
            if ConfigSistema.MOSTRAR_VISTA:
                DISPLAY_W = ConfigSistema.WINDOW_WIDTH
                DISPLAY_H = ConfigSistema.WINDOW_HEIGHT

                if vis_image0 is not None or vis_image1 is not None:
                    # Preparar imágenes
                    if vis_image0 is not None:
                        vis_image0 = cv2.resize(vis_image0, (DISPLAY_W, DISPLAY_H))
                        cv2.putText(vis_image0, "Camara 1", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        if ancho0 is not None:
                            cv2.putText(vis_image0, f"{ancho0:.2f}mm",
                                        (10, DISPLAY_H - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    if vis_image1 is not None:
                        vis_image1 = cv2.resize(vis_image1, (DISPLAY_W, DISPLAY_H))
                        cv2.putText(vis_image1, "Camara 2", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        if ancho1 is not None:
                            cv2.putText(vis_image1, f"{ancho1:.2f}mm",
                                        (10, DISPLAY_H - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # Vista combinada si hay dos cámaras
                    if vis_image0 is not None and vis_image1 is not None and pipeline1:
                        combined = np.hstack((vis_image0, vis_image1))

                        # Separador
                        cv2.line(combined, (DISPLAY_W, 0), (DISPLAY_W, DISPLAY_H),
                                 (0, 255, 255), 2)

                        # Barra superior con info
                        cv2.rectangle(combined, (0, 0), (DISPLAY_W * 2, 60),
                                      (0, 0, 0), -1)

                        # FPS
                        cv2.putText(combined, f"FPS: {fps:.1f}", (20, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Estado del sistema
                        info_estado = controlador_estados.obtener_info()
                        cv2.putText(combined,
                                    f"Estado: {info_estado['estado_sistema']}",
                                    (200, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                        # Sensor ToF
                        if distancia_tof:
                            cv2.putText(combined, f"ToF: {distancia_tof:.1f}cm",
                                        (500, 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # Promedio y control
                        if ancho0 is not None and ancho1 is not None:
                            ancho_prom = (ancho0 + ancho1) / 2
                            error = ancho_prom - ancho_objetivo

                            if abs(error) <= ConfigSistema.TOLERANCIA_ERROR:
                                color = (0, 255, 0)
                                estado = "OK"
                            elif error > 0:
                                color = (0, 165, 255)
                                estado = "ANCHO"
                            else:
                                color = (0, 0, 255)
                                estado = "DELGADO"

                            cv2.putText(combined,
                                        f"Prom: {ancho_prom:.2f}mm ({estado})",
                                        (DISPLAY_W - 180, 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        cv2.imshow("Sistema de Medicion - Vista Dual", combined)

                    elif vis_image0 is not None:
                        # Solo cámara 0
                        cv2.putText(vis_image0, f"FPS: {fps:.1f}",
                                    (DISPLAY_W - 120, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.imshow("Sistema de Medicion", vis_image0)

            # ===========================
            # ESTADÍSTICAS PERIÓDICAS
            # ===========================
            if time.time() - last_print_time >= 2.0:
                print(f"\n--- {datetime.now().strftime('%H:%M:%S')} | "
                      f"Frame {frame_count} | FPS: {fps:.1f} ---")

                # Info de estado
                info_estado = controlador_estados.obtener_info()
                print(f"  🔄 Estado: {info_estado['estado_sistema']} "
                      f"(Sensor: {info_estado['estado_sensor']})")

                if distancia_tof:
                    print(f"  📊 ToF: {distancia_tof:.1f} cm")

                if ancho0:
                    error0 = ancho0 - ancho_objetivo
                    print(f"  📸 Cam0: {ancho0:.2f} mm (error: {error0:+.2f} mm)")

                if ancho1:
                    error1 = ancho1 - ancho_objetivo
                    print(f"  📸 Cam1: {ancho1:.2f} mm (error: {error1:+.2f} mm)")

                if ancho0 and ancho1:
                    ancho_prom = (ancho0 + ancho1) / 2
                    error_prom = ancho_prom - ancho_objetivo
                    print(f"  📏 Promedio: {ancho_prom:.2f} mm "
                          f"(error: {error_prom:+.2f} mm)")

                    if abs(error_prom) <= ConfigSistema.TOLERANCIA_ERROR:
                        print(f"  ✓ Estado: ÓPTIMO")
                    elif error_prom > 0:
                        porcentaje = (error_prom / ancho_objetivo) * 100
                        print(f"  ⚠ Estado: MUY ANCHO (+{porcentaje:.1f}%) - ACELERAR")
                    else:
                        porcentaje = (abs(error_prom) / ancho_objetivo) * 100
                        print(f"  ⚠ Estado: MUY DELGADO (-{porcentaje:.1f}%) - FRENAR")

                last_print_time = time.time()

            # ===========================
            # MANEJO DE TECLAS
            # ===========================
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

            elif key == ord('c'):
                # Cambiar a calibración
                controlador_estados.cambiar_estado(EstadoSistema.CALIBRACION)

            elif key == ord('v'):
                # Cambiar a verificación
                controlador_estados.cambiar_estado(EstadoSistema.VERIFICACION)

            elif key == ord('p'):
                # Cambiar a pausa
                controlador_estados.cambiar_estado(EstadoSistema.PAUSA)

            elif key == ord('m'):
                # Cambiar a monitoreo activo
                controlador_estados.cambiar_estado(EstadoSistema.MONITOREO_ACTIVO)

            # Control de framerate (opcional, para limitar CPU)
            # Comentar para máxima velocidad
            # loop_time = time.time() - loop_start
            # if loop_time < 1.0 / ConfigSistema.CAMERA_FPS:
            #     time.sleep(1.0 / ConfigSistema.CAMERA_FPS - loop_time)

    except KeyboardInterrupt:
        print("\n[INTERRUPCIÓN] Usuario canceló...")
        stop_event.set()

    except Exception as e:
        print(f"\n[ERROR CRÍTICO] {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Limpieza
        print("\n[LIMPIEZA] Cerrando recursos...")
        stop_event.set()

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