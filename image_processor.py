"""
Модуль обработки изображений для анализа деталей вала
Использует методы компьютерного зрения для извлечения признаков
"""

import cv2
import numpy as np
from skimage import feature, filters, measure
from typing import Dict, Tuple, List
import os


class ImageProcessor:
    """Класс для обработки изображений деталей вала"""
    
    def __init__(self):
        self.feature_names = [
            # Базовые геометрические
            'area', 'perimeter', 'circularity', 'solidity',
            # Интенсивность
            'mean_intensity', 'std_intensity',
            # Текстура
            'contrast', 'homogeneity', 'energy', 'correlation',
            # Дефекты
            'edge_density', 'defect_count', 'max_defect_area',
            # Специфические дефекты вала
            'taper_ratio',           # Конусообразность
            'oval_ratio',            # Овальность
            'barrel_ratio',          # Бочкообразность
            'saddle_ratio',          # Седлообразность (противоположность бочкообразности)
            'bend_angle',            # Изгиб (прогиб)
            'size_deviation'         # Отклонение размеров
        ]
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Загрузка изображения с поддержкой кириллицы в путях"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        
        # Исправление проблемы с кириллицей в путях на Windows
        # OpenCV не может читать файлы с кириллицей напрямую, используем numpy
        try:
            # Читаем файл как бинарные данные
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Декодируем изображение из памяти
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                # Если не получилось, пробуем через PIL
                from PIL import Image
                pil_image = Image.open(image_path)
                # Конвертируем в RGB, если нужно
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        except Exception as e:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}. Ошибка: {e}")
        
        return image
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предобработка изображения:
        - Конвертация в grayscale
        - Улучшение контраста
        - Размытие для уменьшения шума
        """
        # Конвертация в grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Улучшение контраста с помощью CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Размытие для уменьшения шума
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        return enhanced, blurred
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Обнаружение краев с помощью Canny"""
        edges = cv2.Canny(image, 50, 150)
        return edges
    
    def detect_defects(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Обнаружение потенциальных дефектов:
        - Трещины, царапины, вмятины, неровности поверхности
        """
        # Несколько методов обнаружения дефектов
        
        # Метод 1: Адаптивная бинаризация
        binary1 = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Метод 2: Пороговая обработка с Otsu
        _, binary2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Метод 3: Обнаружение краев (дефекты часто создают дополнительные края)
        edges = cv2.Canny(image, 50, 150)
        binary3 = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Комбинируем методы
        combined = cv2.bitwise_or(binary1, binary2)
        combined = cv2.bitwise_or(combined, binary3)
        
        # Морфологические операции для выделения дефектов
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        
        # Открытие для удаления мелкого шума
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # Закрытие для соединения разрывов в дефектах
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        # Дополнительное расширение для лучшего выделения дефектов
        dilated = cv2.dilate(closed, kernel_small, iterations=1)
        
        # Поиск контуров
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Фильтрация контуров по размеру (более строгий порог)
        min_area = image.shape[0] * image.shape[1] * 0.0005  # 0.05% от площади изображения
        max_area = image.shape[0] * image.shape[1] * 0.5  # Максимум 50% (чтобы не захватить весь вал)
        
        defects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                # Дополнительная проверка: отношение периметра к площади (дефекты часто имеют высокое отношение)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter ** 2)
                    # Дефекты часто имеют низкую компактность (вытянутые формы)
                    if compactness < 0.8 or area > min_area * 2:  # Более строгие критерии
                        defects.append(cnt)
        
        return defects, dilated
    
    def calculate_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Вычисление текстурных признаков (GLCM)"""
        # Нормализация изображения
        normalized = (image / 255.0 * 15).astype(np.uint8)
        
        # Вычисление GLCM матрицы
        glcm = feature.graycomatrix(
            normalized,
            distances=[1, 2],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=16,
            symmetric=True,
            normed=True
        )
        
        # Извлечение признаков из GLCM
        contrast = feature.graycoprops(glcm, 'contrast').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        energy = feature.graycoprops(glcm, 'energy').mean()
        correlation = feature.graycoprops(glcm, 'correlation').mean()
        
        return {
            'contrast': float(contrast),
            'homogeneity': float(homogeneity),
            'energy': float(energy),
            'correlation': float(correlation)
        }
    
    def detect_shaft_contour(self, image: np.ndarray) -> np.ndarray:
        """
        Обнаружение контура вала на изображении
        Возвращает контур основного объекта (вала)
        """
        # Бинаризация для выделения вала
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Инверсия, если вал темнее фона
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        # Морфологические операции для очистки
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Поиск контуров
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Выбираем самый большой контур (предположительно вал)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Фильтруем слишком маленькие контуры
        min_area = image.shape[0] * image.shape[1] * 0.1  # 10% от площади изображения
        if cv2.contourArea(main_contour) < min_area:
            return None
        
        return main_contour
    
    def calculate_taper_ratio(self, contour: np.ndarray) -> float:
        """
        Вычисление конусообразности (taper ratio)
        Сравнивает диаметры в начале и конце вала
        """
        if contour is None or len(contour) < 4:
            return 0.0
        
        # Получаем ограничивающий прямоугольник
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        
        # Определяем длину и ширину вала
        length = max(width, height)
        width_min = min(width, height)
        
        # Разделяем контур на части (начало, середина, конец)
        # Берем точки из разных частей контура
        n_points = len(contour)
        if n_points < 6:
            return 0.0
        
        # Начало вала (первые 20% точек)
        start_idx = n_points // 5
        start_points = contour[:start_idx]
        
        # Конец вала (последние 20% точек)
        end_idx = n_points - n_points // 5
        end_points = contour[end_idx:]
        
        # Вычисляем средние диаметры
        if len(start_points) > 0 and len(end_points) > 0:
            # Находим точки, наиболее удаленные от центральной линии
            start_diameters = []
            end_diameters = []
            
            # Упрощенный подход: используем высоту ограничивающего прямоугольника
            # в разных частях контура
            box_points = cv2.boxPoints(rect)
            box_points = np.int32(box_points)
            
            # Вычисляем отношение минимальной и максимальной ширины
            taper_ratio = abs(width_min / length) if length > 0 else 0.0
            
            # Более точный метод: анализ изменения ширины вдоль вала
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Разделяем на сегменты и вычисляем ширину каждого
                n_segments = 10
                segment_widths = []
                
                for i in range(n_segments):
                    start_seg = int(i * n_points / n_segments)
                    end_seg = int((i + 1) * n_points / n_segments)
                    seg_points = contour[start_seg:end_seg]
                    
                    if len(seg_points) > 0:
                        # Вычисляем ширину сегмента
                        seg_rect = cv2.minAreaRect(seg_points)
                        seg_width = min(seg_rect[1])
                        segment_widths.append(seg_width)
                
                if len(segment_widths) >= 2:
                    # Конусообразность = разница между началом и концом
                    first_width = segment_widths[0]
                    last_width = segment_widths[-1]
                    
                    if first_width > 0:
                        taper_ratio = abs((last_width - first_width) / first_width)
                    else:
                        taper_ratio = 0.0
        
        return float(taper_ratio)
    
    def calculate_oval_ratio(self, contour: np.ndarray) -> float:
        """
        Вычисление овальности
        Сравнивает максимальный и минимальный диаметры поперечного сечения
        """
        if contour is None or len(contour) < 4:
            return 0.0
        
        # Находим минимальную ограничивающую окружность
        (x, y), radius = cv2.minEnclosingCircle(contour)
        max_diameter = 2 * radius
        
        # Находим минимальную площадь прямоугольника
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        min_diameter = min(width, height)
        
        # Овальность = отношение разницы диаметров к среднему
        if max_diameter > 0:
            oval_ratio = abs((max_diameter - min_diameter) / max_diameter)
        else:
            oval_ratio = 0.0
        
        # Альтернативный метод: анализ эксцентриситета эллипса
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            a, b = ellipse[1][0] / 2, ellipse[1][1] / 2  # Полуоси
            if a > 0 and b > 0:
                # Эксцентриситет эллипса
                if a > b:
                    eccentricity = np.sqrt(1 - (b/a)**2)
                else:
                    eccentricity = np.sqrt(1 - (a/b)**2)
                oval_ratio = max(oval_ratio, eccentricity)
        
        return float(oval_ratio)
    
    def calculate_barrel_ratio(self, contour: np.ndarray) -> float:
        """
        Вычисление бочкообразности
        Проверяет, расширяется ли вал в середине
        """
        if contour is None or len(contour) < 6:
            return 0.0
        
        n_points = len(contour)
        
        # Разделяем контур на сегменты
        n_segments = 10
        segment_widths = []
        segment_positions = []
        
        for i in range(n_segments):
            start_seg = int(i * n_points / n_segments)
            end_seg = int((i + 1) * n_points / n_segments)
            seg_points = contour[start_seg:end_seg]
            
            if len(seg_points) > 0:
                seg_rect = cv2.minAreaRect(seg_points)
                seg_width = min(seg_rect[1])
                segment_widths.append(seg_width)
                segment_positions.append(i / n_segments)  # Относительная позиция
        
        if len(segment_widths) < 3:
            return 0.0
        
        # Находим ширину в начале, середине и конце
        start_width = segment_widths[0]
        middle_width = segment_widths[len(segment_widths) // 2]
        end_width = segment_widths[-1]
        
        # Бочкообразность = если середина шире, чем края
        if start_width > 0 and end_width > 0:
            avg_edge_width = (start_width + end_width) / 2
            if avg_edge_width > 0:
                barrel_ratio = (middle_width - avg_edge_width) / avg_edge_width
                barrel_ratio = max(0.0, barrel_ratio)  # Только положительные значения
            else:
                barrel_ratio = 0.0
        else:
            barrel_ratio = 0.0
        
        return float(barrel_ratio)
    
    def calculate_saddle_ratio(self, contour: np.ndarray) -> float:
        """
        Вычисление седлообразности (противоположность бочкообразности)
        Проверяет, сужается ли вал в середине (седлообразная форма)
        """
        if contour is None or len(contour) < 6:
            return 0.0
        
        n_points = len(contour)
        
        # Разделяем контур на сегменты
        n_segments = 10
        segment_widths = []
        
        for i in range(n_segments):
            start_seg = int(i * n_points / n_segments)
            end_seg = int((i + 1) * n_points / n_segments)
            seg_points = contour[start_seg:end_seg]
            
            if len(seg_points) > 0:
                seg_rect = cv2.minAreaRect(seg_points)
                seg_width = min(seg_rect[1])
                segment_widths.append(seg_width)
        
        if len(segment_widths) < 3:
            return 0.0
        
        # Находим ширину в начале, середине и конце
        start_width = segment_widths[0]
        middle_width = segment_widths[len(segment_widths) // 2]
        end_width = segment_widths[-1]
        
        # Седлообразность = если середина уже, чем края
        if start_width > 0 and end_width > 0:
            avg_edge_width = (start_width + end_width) / 2
            if avg_edge_width > 0:
                saddle_ratio = (avg_edge_width - middle_width) / avg_edge_width
                saddle_ratio = max(0.0, saddle_ratio)  # Только положительные значения
            else:
                saddle_ratio = 0.0
        else:
            saddle_ratio = 0.0
        
        return float(saddle_ratio)
    
    def calculate_bend_angle(self, contour: np.ndarray) -> float:
        """
        Вычисление изгиба (прогиба) вала
        Определяет отклонение от прямой линии
        """
        if contour is None or len(contour) < 4:
            return 0.0
        
        # Находим линию, соединяющую концы вала
        # Берем крайние точки контура
        leftmost = tuple(contour[contour[:,:,0].argmin()][0])
        rightmost = tuple(contour[contour[:,:,0].argmax()][0])
        topmost = tuple(contour[contour[:,:,1].argmin()][0])
        bottommost = tuple(contour[contour[:,:,1].argmax()][0])
        
        # Определяем основное направление вала
        # Используем главную ось минимального ограничивающего прямоугольника
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        
        # Вычисляем отклонение точек контура от прямой линии
        # Строим линию через центр масс
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return 0.0
        
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        
        # Находим максимальное отклонение от прямой
        max_deviation = 0.0
        
        # Упрощенный метод: используем угол наклона
        # Идеальный вал должен быть прямым (угол близок к 0° или 90°)
        normalized_angle = abs(angle) % 90
        if normalized_angle > 45:
            normalized_angle = 90 - normalized_angle
        
        # Вычисляем отклонение точек от линии, проходящей через центр
        # Берем несколько точек контура и вычисляем расстояние до линии
        n_samples = min(20, len(contour))
        sample_indices = np.linspace(0, len(contour) - 1, n_samples, dtype=int)
        
        deviations = []
        for idx in sample_indices:
            point = contour[idx][0]
            px, py = point[0], point[1]
            
            # Расстояние от точки до центра
            dist_to_center = np.sqrt((px - cx)**2 + (py - cy)**2)
            deviations.append(dist_to_center)
        
        if len(deviations) > 0:
            # Стандартное отклонение показывает изгиб
            mean_dev = np.mean(deviations)
            if mean_dev > 0:
                std_dev = np.std(deviations)
                bend_angle = std_dev / mean_dev  # Нормализованное отклонение
            else:
                bend_angle = 0.0
        else:
            bend_angle = 0.0
        
        # Комбинируем с углом наклона
        bend_angle = max(bend_angle, normalized_angle / 90.0)
        
        return float(bend_angle)
    
    def calculate_size_deviation(self, contour: np.ndarray, reference_size: float = None) -> float:
        """
        Вычисление отклонения размеров
        Сравнивает размеры вала с эталонными значениями
        """
        if contour is None or len(contour) < 4:
            return 0.0
        
        # Вычисляем основные размеры
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        length = max(width, height)
        diameter = min(width, height)
        
        # Если эталонный размер не задан, используем средний размер из контура
        if reference_size is None:
            # Используем площадь контура для оценки эталонного размера
            area = cv2.contourArea(contour)
            # Предполагаем, что вал цилиндрический
            # Площадь проекции ≈ длина × диаметр
            if length > 0:
                estimated_diameter = area / length
                reference_size = estimated_diameter
            else:
                reference_size = diameter
        
        # Вычисляем отклонение
        if reference_size > 0:
            # Относительное отклонение диаметра
            diameter_deviation = abs((diameter - reference_size) / reference_size)
            
            # Также учитываем отклонение длины (если есть эталон)
            # Для простоты используем только диаметр
            size_deviation = diameter_deviation
        else:
            size_deviation = 0.0
        
        # Альтернативный метод: анализ вариаций диаметра вдоль вала
        n_segments = 10
        n_points = len(contour)
        segment_diameters = []
        
        for i in range(n_segments):
            start_seg = int(i * n_points / n_segments)
            end_seg = int((i + 1) * n_points / n_segments)
            seg_points = contour[start_seg:end_seg]
            
            if len(seg_points) > 0:
                seg_rect = cv2.minAreaRect(seg_points)
                seg_diameter = min(seg_rect[1])
                segment_diameters.append(seg_diameter)
        
        if len(segment_diameters) > 0:
            mean_diameter = np.mean(segment_diameters)
            if mean_diameter > 0:
                # Стандартное отклонение диаметров показывает отклонение размеров
                std_diameter = np.std(segment_diameters)
                diameter_variation = std_diameter / mean_diameter
                size_deviation = max(size_deviation, diameter_variation)
        
        return float(size_deviation)
    
    def calculate_geometric_features(self, image: np.ndarray, defects: List) -> Dict[str, float]:
        """Вычисление геометрических признаков, включая специфические дефекты вала"""
        # Основные характеристики изображения
        area = image.shape[0] * image.shape[1]
        perimeter = 2 * (image.shape[0] + image.shape[1])
        
        # Характеристики дефектов
        defect_areas = [cv2.contourArea(cnt) for cnt in defects]
        defect_count = len(defects)
        max_defect_area = max(defect_areas) if defect_areas else 0
        
        # Обнаружение контура вала
        shaft_contour = self.detect_shaft_contour(image)
        
        # Вычисление округлости и сплошности
        # Используем контур вала, если найден, иначе используем весь контур изображения
        if shaft_contour is not None and len(shaft_contour) > 0:
            main_area = cv2.contourArea(shaft_contour)
            main_perimeter = cv2.arcLength(shaft_contour, True)
            circularity = (4 * np.pi * main_area) / (main_perimeter ** 2) if main_perimeter > 0 else 0
            
            hull = cv2.convexHull(shaft_contour)
            hull_area = cv2.contourArea(hull)
            solidity = main_area / hull_area if hull_area > 0 else 0
        else:
            # Если контур вала не найден, используем дефекты или весь контур изображения
            if len(defects) > 0:
                # Используем самый большой дефект, но только если он не слишком большой
                # (чтобы не захватить весь вал)
                max_defect_area = max([cv2.contourArea(cnt) for cnt in defects])
                image_area = image.shape[0] * image.shape[1]
                
                # Если самый большой "дефект" больше 30% изображения, это скорее всего весь вал
                if max_defect_area < image_area * 0.3:
                    main_contour = max(defects, key=cv2.contourArea)
                    main_area = cv2.contourArea(main_contour)
                    main_perimeter = cv2.arcLength(main_contour, True)
                    circularity = (4 * np.pi * main_area) / (main_perimeter ** 2) if main_perimeter > 0 else 0.5
                    
                    hull = cv2.convexHull(main_contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = main_area / hull_area if hull_area > 0 else 0.8
                else:
                    # Если "дефект" слишком большой, используем значения по умолчанию
                    circularity = 0.5  # Среднее значение для проекции вала
                    solidity = 0.9  # Высокая сплошность
            else:
                # Нет дефектов - используем значения по умолчанию для исправного вала
                circularity = 0.5  # Для проекции вала это нормально
                solidity = 0.95  # Высокая сплошность
            shaft_contour = None
        
        # Вычисление специфических дефектов вала
        taper_ratio = self.calculate_taper_ratio(shaft_contour) if shaft_contour is not None else 0.0
        oval_ratio = self.calculate_oval_ratio(shaft_contour) if shaft_contour is not None else 0.0
        barrel_ratio = self.calculate_barrel_ratio(shaft_contour) if shaft_contour is not None else 0.0
        saddle_ratio = self.calculate_saddle_ratio(shaft_contour) if shaft_contour is not None else 0.0
        bend_angle = self.calculate_bend_angle(shaft_contour) if shaft_contour is not None else 0.0
        size_deviation = self.calculate_size_deviation(shaft_contour) if shaft_contour is not None else 0.0
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'solidity': float(solidity),
            'defect_count': float(defect_count),
            'max_defect_area': float(max_defect_area),
            # Специфические дефекты вала
            'taper_ratio': float(taper_ratio),
            'oval_ratio': float(oval_ratio),
            'barrel_ratio': float(barrel_ratio),
            'saddle_ratio': float(saddle_ratio),
            'bend_angle': float(bend_angle),
            'size_deviation': float(size_deviation)
        }
    
    def calculate_intensity_features(self, image: np.ndarray) -> Dict[str, float]:
        """Вычисление признаков интенсивности"""
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        return {
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity)
        }
    
    def calculate_edge_features(self, edges: np.ndarray) -> Dict[str, float]:
        """Вычисление признаков краев"""
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return {
            'edge_density': float(edge_density)
        }
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Извлечение всех признаков из изображения
        Возвращает вектор признаков для машинного обучения
        """
        # Загрузка и предобработка
        image = self.load_image(image_path)
        enhanced, blurred = self.preprocess_image(image)
        
        # Обнаружение дефектов
        defects, binary_mask = self.detect_defects(blurred)
        
        # Обнаружение краев
        edges = self.detect_edges(blurred)
        
        # Извлечение признаков
        texture_features = self.calculate_texture_features(enhanced)
        geometric_features = self.calculate_geometric_features(enhanced, defects)
        intensity_features = self.calculate_intensity_features(enhanced)
        edge_features = self.calculate_edge_features(edges)
        
        # Объединение всех признаков
        features = {
            **geometric_features,
            **intensity_features,
            **texture_features,
            **edge_features
        }
        
        # Преобразование в массив в правильном порядке
        feature_vector = np.array([features[name] for name in self.feature_names])
        
        return feature_vector
    
    def visualize_analysis(self, image_path: str, save_path: str = None) -> np.ndarray:
        """
        Визуализация анализа изображения
        Показывает исходное изображение с выделенными дефектами
        """
        image = self.load_image(image_path)
        enhanced, blurred = self.preprocess_image(image)
        defects, binary_mask = self.detect_defects(blurred)
        edges = self.detect_edges(blurred)
        
        # Создание визуализации
        vis_image = image.copy()
        
        # Рисование контуров дефектов
        cv2.drawContours(vis_image, defects, -1, (0, 0, 255), 2)
        
        # Добавление текста с информацией
        info_text = f"Найдено дефектов: {len(defects)}"
        cv2.putText(vis_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image



