"""
Utilidades para conversión de frames de cámaras Orbbec
Compatible con pyorbbecsdk
"""

import numpy as np
import cv2

def frame_to_gray_image(frame):
    """
    Convierte un frame Orbbec a escala de grises (uint8).
    Optimizado para MJPEG: decodifica directamente a GRAY (más rápido que COLOR).
    """
    try:
        width = frame.get_width()
        height = frame.get_height()

        data = np.asanyarray(frame.get_data())

        if len(data.shape) == 1:
            total_pixels = width * height
            expected_rgb = total_pixels * 3

            # Caso típico: MJPEG/JPEG comprimido (muy por debajo del tamaño raw)
            if len(data) < expected_rgb * 0.5:
                gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                return gray

            # Raw: deducir bytes por pixel
            bytes_per_pixel = len(data) // total_pixels if total_pixels > 0 else 0
            if bytes_per_pixel >= 3:
                # RGB/BGR: tomar un canal y listo
                image = data.reshape((height, width, bytes_per_pixel))
                return image[:, :, 0].astype(np.uint8, copy=False)
            elif bytes_per_pixel == 2:
                image = data.reshape((height, width, 2))
                return image[:, :, 0].astype(np.uint8, copy=False)
            else:
                # Fallback: intentar decodificar como comprimido
                gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                return gray

        if len(data.shape) == 3:
            # (H, W, C): tomar canal 0
            return data[:, :, 0].astype(np.uint8, copy=False)

        if len(data.shape) == 2:
            return data.astype(np.uint8, copy=False)

        return None
    except Exception as e:
        print(f"Error en frame_to_gray_image: {e}")
        return None

def frame_to_bgr_image(frame):
    """
    Convierte un frame de Orbbec a formato BGR de OpenCV.
    
    Args:
        frame: Frame de pyorbbecsdk (ColorFrame o DepthFrame)
        
    Returns:
        numpy.ndarray: Imagen en formato BGR lista para OpenCV, o None si hay error
    """
    try:
        # Obtener dimensiones del frame
        width = frame.get_width()
        height = frame.get_height()
        
        # Obtener datos del frame como array numpy
        data = np.asanyarray(frame.get_data())
        
        # Determinar el formato y convertir
        if len(data.shape) == 1:
            # Datos planos, necesitan ser reformateados
            total_pixels = width * height
            expected_rgb = total_pixels * 3
            
            # Detectar si es MJPEG/JPEG comprimido
            if len(data) < expected_rgb * 0.5:
                # Datos comprimidos (MJPEG) - decodificar con OpenCV
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                return image
            
            # Calcular número de canales para datos raw
            bytes_per_pixel = len(data) // total_pixels
            
            if bytes_per_pixel == 3:
                # RGB: reshape y convertir a BGR
                image = data.reshape((height, width, 3))
                image = image[:, :, ::-1].copy()  # RGB -> BGR
            elif bytes_per_pixel == 2:
                # Puede ser YUV o formato de 16 bits
                image = data.reshape((height, width, 2))
                # Tomar solo el primer canal o convertir según formato
                image = np.stack([image[:,:,0]] * 3, axis=2)  # Convertir a 3 canales
            elif bytes_per_pixel >= 1:
                # Caso general: intentar reshape básico
                image = data.reshape((height, width, bytes_per_pixel))
            else:
                # bytes_per_pixel es 0, datos comprimidos - intentar decodificar
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is None:
                    return None
                
        elif len(data.shape) == 3:
            # Ya tiene forma (height, width, channels)
            image = data
            
            # Si es RGB, convertir a BGR
            if image.shape[2] == 3:
                # Verificar si necesita conversión RGB -> BGR
                # OpenCV usa BGR por defecto
                image = image[:, :, ::-1].copy()
            elif image.shape[2] == 4:
                # RGBA -> BGR (descarta canal alpha)
                image = image[:, :, [2, 1, 0]].copy()
                
        elif len(data.shape) == 2:
            # Imagen de un solo canal (ej: depth o infrarrojo)
            # Convertir a 3 canales duplicando
            image = np.stack([data] * 3, axis=2)
            
        else:
            print(f"Formato de frame no soportado: shape={data.shape}")
            return None
        
        # Asegurar que el tipo de datos es uint8
        if image.dtype != np.uint8:
            # Normalizar y convertir
            if image.max() > 255:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        return image
        
    except Exception as e:
        print(f"Error en frame_to_bgr_image: {e}")
        import traceback
        traceback.print_exc()
        return None


def frame_to_depth_image(depth_frame, min_depth=20, max_depth=10000):
    """
    Convierte un depth frame a imagen visualizable.
    
    Args:
        depth_frame: DepthFrame de pyorbbecsdk
        min_depth: Profundidad mínima en mm
        max_depth: Profundidad máxima en mm
        
    Returns:
        numpy.ndarray: Imagen de profundidad normalizada (0-255)
    """
    try:
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        
        # Obtener datos de profundidad
        depth_data = np.asanyarray(depth_frame.get_data())
        
        # Reshape si es necesario
        if len(depth_data.shape) == 1:
            depth_data = depth_data.reshape((height, width))
        
        # Filtrar valores fuera de rango
        depth_data = np.clip(depth_data, min_depth, max_depth)
        
        # Normalizar a 0-255
        depth_image = ((depth_data - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
        
        return depth_image
        
    except Exception as e:
        print(f"Error en frame_to_depth_image: {e}")
        return None


def print_frame_info(frame):
    """
    Imprime información de debug sobre un frame.
    Útil para diagnosticar problemas de formato.
    """
    try:
        print(f"\n=== Frame Info ===")
        print(f"Width: {frame.get_width()}")
        print(f"Height: {frame.get_height()}")
        
        data = np.asanyarray(frame.get_data())
        print(f"Data shape: {data.shape}")
        print(f"Data dtype: {data.dtype}")
        print(f"Data min: {data.min()}")
        print(f"Data max: {data.max()}")
        print(f"Data size: {data.size}")
        
        if hasattr(frame, 'get_format'):
            print(f"Format: {frame.get_format()}")
            
    except Exception as e:
        print(f"Error mostrando info de frame: {e}")


# Compatibilidad con versiones anteriores
def frame_to_rgb_image(frame):
    """
    Convierte frame a RGB (en lugar de BGR).
    Menos común en OpenCV pero útil para algunas aplicaciones.
    """
    bgr_image = frame_to_bgr_image(frame)
    if bgr_image is not None:
        return bgr_image[:, :, ::-1].copy()  # BGR -> RGB
    return None
