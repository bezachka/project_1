"""
Основное приложение для анализа деталей вала на наличие дефектов
"""

import os
import sys
import argparse
from pathlib import Path
from image_processor import ImageProcessor
from defect_classifier import DefectClassifier
import cv2
import numpy as np


class ShaftDefectAnalyzer:
    """Основной класс приложения для анализа дефектов вала"""
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Инициализация анализатора
        
        Args:
            model_path: Путь к сохраненной модели (опционально)
            scaler_path: Путь к сохраненному масштабировщику (опционально)
        """
        self.image_processor = ImageProcessor()
        self.classifier = DefectClassifier(model_type='random_forest')
        
        # Загрузка модели, если указаны пути
        if model_path and scaler_path:
            try:
                self.classifier.load_model(model_path, scaler_path)
                print("✓ Модель успешно загружена")
            except Exception as e:
                print(f"⚠ Предупреждение: Не удалось загрузить модель: {e}")
                print("  Будет использована модель по умолчанию (требуется обучение)")
        else:
            print("⚠ Модель не загружена. Для использования необходимо обучить модель.")
    
    def analyze_image(self, image_path: str, visualize: bool = True, 
                     output_dir: str = None) -> dict:
        """
        Анализ одного изображения
        
        Args:
            image_path: Путь к изображению
            visualize: Создавать ли визуализацию
            output_dir: Директория для сохранения результатов
        
        Returns:
            Словарь с результатами анализа
        """
        print(f"\n📸 Анализ изображения: {image_path}")
        
        try:
            # Извлечение признаков
            print("  Извлечение признаков...")
            features = self.image_processor.extract_features(image_path)
            
            # Предсказание
            if self.classifier.is_trained:
                print("  Классификация...")
                prediction, probability = self.classifier.predict(features)
                
                # Дополнительная проверка на основе признаков дефектов
                # Если признаки дефектов превышают пороги, считаем дефектом
                defect_indicators = self._check_defect_indicators(features)
                
                # Комбинируем предсказание модели с правилами
                # КРИТИЧЕСКИ ВАЖНО: Доверяем модели, так как она обучена на этих данных
                # Переопределяем ТОЛЬКО в исключительных случаях, когда модель явно ошиблась
                if defect_indicators['has_defects']:
                    # Переопределяем ТОЛЬКО если:
                    # 1. Модель очень уверена в "исправна" (probability < 0.2 - очень низкая)
                    # 2. И найдено ОЧЕНЬ МНОГО критических признаков дефектов (>= 6)
                    # 3. И признаки действительно критичны (не просто проекция вала)
                    if probability < 0.2 and defect_indicators['indicators_count'] >= 6:
                        # Только в крайних случаях переопределяем
                        prediction = 1
                        probability = 0.55  # Умеренная вероятность дефекта
                        print(f"  ⚠ Обнаружены множественные критичные признаки дефектов: {defect_indicators['reasons']}")
                    else:
                        # НЕ переопределяем решение модели - доверяем обученной модели
                        # Просто информируем о найденных признаках
                        if defect_indicators['indicators_count'] > 0:
                            model_status = "ДЕФЕКТНА" if prediction == 1 else "ИСПРАВНА"
                            print(f"  ℹ Найдено признаков дефектов: {defect_indicators['indicators_count']} (решение модели: {model_status})")
                
                status = "ДЕФЕКТНА" if prediction == 1 else "ИСПРАВНА"
                confidence = probability if prediction == 1 else (1 - probability)
                
                # Для исправных деталей показываем вероятность исправности, а не дефекта
                # Для дефектных - вероятность дефекта
                if prediction == 1:
                    # Дефектная деталь - показываем вероятность дефекта
                    probability_display = probability
                else:
                    # Исправная деталь - показываем вероятность исправности (1 - вероятность дефекта)
                    probability_display = 1 - probability
                
                result = {
                    'image_path': image_path,
                    'status': status,
                    'prediction': prediction,
                    'confidence': confidence,
                    'probability_defect': probability,  # Всегда вероятность дефекта (для внутреннего использования)
                    'probability_display': probability_display,  # Вероятность для отображения
                    'features': features.tolist(),
                    'defect_indicators': defect_indicators
                }
                
                print(f"  ✓ Результат: {status} (уверенность: {confidence:.2%})")
            else:
                print("  ⚠ Модель не обучена. Только извлечение признаков.")
                result = {
                    'image_path': image_path,
                    'status': 'UNKNOWN',
                    'features': features.tolist(),
                    'note': 'Модель не обучена, требуется обучение для классификации'
                }
            
            # Визуализация
            if visualize:
                print("  Создание визуализации...")
                vis_image = self.image_processor.visualize_analysis(image_path)
                
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(
                        output_dir, 
                        f"analysis_{Path(image_path).stem}.jpg"
                    )
                    cv2.imwrite(output_path, vis_image)
                    result['visualization_path'] = output_path
                    print(f"  ✓ Визуализация сохранена: {output_path}")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Ошибка при анализе: {e}")
            return {
                'image_path': image_path,
                'status': 'ERROR',
                'error': str(e)
            }
    
    def analyze_directory(self, directory: str, visualize: bool = True,
                         output_dir: str = None) -> list:
        """
        Анализ всех изображений в директории
        
        Args:
            directory: Путь к директории с изображениями
            visualize: Создавать ли визуализацию
            output_dir: Директория для сохранения результатов
        
        Returns:
            Список результатов анализа
        """
        print(f"\n📁 Анализ директории: {directory}")
        
        # Поддерживаемые форматы изображений
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Поиск всех изображений
        image_files = [
            f for f in os.listdir(directory)
            if Path(f).suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print("  ⚠ Изображения не найдены")
            return []
        
        print(f"  Найдено изображений: {len(image_files)}")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(directory, image_file)
            print(f"\n[{i}/{len(image_files)}]")
            result = self.analyze_image(image_path, visualize, output_dir)
            results.append(result)
        
        return results
    
    def _check_defect_indicators(self, features: np.ndarray) -> dict:
        """
        Проверка признаков дефектов на основе пороговых значений
        Дополнительная проверка помимо ML модели
        """
        feature_names = self.image_processor.feature_names
        feature_dict = dict(zip(feature_names, features))
        
        has_defects = False
        reasons = []
        
        # Проверка специфических дефектов вала
        # Пороги установлены с учетом допустимых отклонений, которые не влияют на работу вала
        # Учитываем, что проекции вала могут иметь высокие значения из-за перспективы
        
        # Конусообразность - для проекций может быть высокой из-за перспективы
        # Критично только очень сильное отклонение (> 1.0 = 100%)
        taper = feature_dict.get('taper_ratio', 0)
        if taper > 1.0:  # Очень критическое отклонение (100% и более)
            has_defects = True
            reasons.append(f"Конусообразность: {taper:.3f} (критично > 1.0)")
        
        # Овальность - для проекций вала может быть очень высокой (вал виден сбоку)
        # Почти всегда высокая для проекций, поэтому не используем как критический признак
        # Только если овальность близка к 1.0 (почти прямая линия) - это может быть дефект
        # Но для проекций это нормально, поэтому не добавляем в критические
        
        # Бочкообразность - для проекций может быть высокой
        # Критично только очень сильное отклонение (> 0.5 = 50%)
        barrel = feature_dict.get('barrel_ratio', 0)
        if barrel > 0.5:  # Очень критическое отклонение
            has_defects = True
            reasons.append(f"Бочкообразность: {barrel:.3f} (критично > 0.5)")
        
        # Седлообразность - для проекций может быть высокой
        # Критично только очень сильное отклонение (> 0.5 = 50%)
        saddle = feature_dict.get('saddle_ratio', 0)
        if saddle > 0.5:  # Очень критическое отклонение
            has_defects = True
            reasons.append(f"Седлообразность: {saddle:.3f} (критично > 0.5)")
        
        # Изгиб (прогиб вала) - для проекций может быть высоким из-за угла съемки
        # Критично только очень сильное отклонение (> 0.5 = 50%)
        bend = feature_dict.get('bend_angle', 0)
        if bend > 0.5:  # Очень критическое отклонение
            has_defects = True
            reasons.append(f"Изгиб (прогиб): {bend:.3f} (критично > 0.5)")
        
        # Отклонение размеров - НЕ используется как критерий дефекта
        # Размеры могут сильно варьироваться в зависимости от проекции и перспективы съемки
        
        # Проверка количества дефектов (более строгий порог)
        if feature_dict.get('defect_count', 0) > 5:  # Увеличен порог
            has_defects = True
            reasons.append(f"Много дефектов: {feature_dict['defect_count']:.0f}")
        
        # Проверка площади дефектов - нужно учитывать размер изображения
        # Исключаем случаи, когда "дефект" - это весь вал
        image_area_estimate = feature_dict.get('area', 100000)  # Примерная площадь
        max_defect_threshold = image_area_estimate * 0.10  # 10% от площади изображения (увеличен порог)
        defect_area = feature_dict.get('max_defect_area', 0)
        # Проверяем только если дефект не слишком большой (не весь вал)
        if defect_area > max_defect_threshold and defect_area < image_area_estimate * 0.5:
            has_defects = True
            reasons.append(f"Большой дефект: площадь {defect_area:.0f} (порог: {max_defect_threshold:.0f})")
        
        # Проверка округлости - для проекций вала может быть очень низкой (вал виден сбоку)
        # Используем очень низкий порог, так как проекции редко круглые
        if feature_dict.get('circularity', 1.0) < 0.2:  # Очень низкий порог для проекций
            has_defects = True
            reasons.append(f"Низкая округлость: {feature_dict['circularity']:.3f} (порог: 0.2)")
        
        # Проверка сплошности
        if feature_dict.get('solidity', 1.0) < 0.75:  # Снижен порог (было 0.85)
            has_defects = True
            reasons.append(f"Низкая сплошность: {feature_dict['solidity']:.3f} (порог: 0.75)")
        
        # Проверка плотности краев
        if feature_dict.get('edge_density', 0) > 0.25:  # Увеличен порог
            has_defects = True
            reasons.append(f"Высокая плотность краев: {feature_dict['edge_density']:.3f} (порог: 0.25)")
        
        # Проверка контраста
        if feature_dict.get('contrast', 0) > 0.7:  # Увеличен порог
            has_defects = True
            reasons.append(f"Высокий контраст: {feature_dict['contrast']:.3f} (порог: 0.7)")
        
        return {
            'has_defects': has_defects,
            'reasons': reasons,
            'indicators_count': len(reasons)
        }
    
    def train_model(self, data_dir: str, labels_file: str = None):
        """
        Обучение модели на данных
        
        Args:
            data_dir: Директория с изображениями
            labels_file: Путь к файлу с метками (CSV: image_path, label)
                         Если None, ожидается структура: data_dir/good/ и data_dir/defect/
        """
        print("\n🎓 Обучение модели...")
        
        # Загрузка данных
        if labels_file and os.path.exists(labels_file):
            # Загрузка из CSV файла
            import pandas as pd
            df = pd.read_csv(labels_file)
            image_paths = df['image_path'].tolist()
            labels = df['label'].tolist()
        else:
            # Загрузка из структуры директорий
            good_dir = os.path.join(data_dir, 'good')
            defect_dir = os.path.join(data_dir, 'defect')
            
            image_paths = []
            labels = []
            
            if os.path.exists(good_dir):
                for f in os.listdir(good_dir):
                    if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                        # Правильное объединение путей с поддержкой кириллицы
                        full_path = os.path.join(good_dir, f)
                        image_paths.append(full_path)
                        labels.append(0)  # Исправна
            
            if os.path.exists(defect_dir):
                for f in os.listdir(defect_dir):
                    if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                        # Правильное объединение путей с поддержкой кириллицы
                        full_path = os.path.join(defect_dir, f)
                        image_paths.append(full_path)
                        labels.append(1)  # Дефектна
        
        if not image_paths:
            raise ValueError("Не найдены изображения для обучения")
        
        print(f"  Загружено изображений: {len(image_paths)}")
        print(f"  Исправных: {labels.count(0)}")
        print(f"  Дефектных: {labels.count(1)}")
        
        # Извлечение признаков
        print("  Извлечение признаков из изображений...")
        features_list = []
        valid_labels = []
        
        for i, (img_path, label) in enumerate(zip(image_paths, labels), 1):
            try:
                print(f"    [{i}/{len(image_paths)}] {Path(img_path).name}")
                features = self.image_processor.extract_features(img_path)
                features_list.append(features)
                valid_labels.append(label)
            except Exception as e:
                print(f"    ⚠ Пропущено {img_path}: {e}")
        
        if not features_list:
            raise ValueError("Не удалось извлечь признаки ни из одного изображения")
        
        # Обучение модели
        print("\n  Обучение модели...")
        X = np.array(features_list)
        y = np.array(valid_labels)
        
        metrics = self.classifier.train(X, y)
        
        print("\n  ✓ Обучение завершено!")
        print(f"  Точность на обучающей выборке: {metrics['train_accuracy']:.2%}")
        print(f"  Точность на тестовой выборке: {metrics['test_accuracy']:.2%}")
        print("\n  Отчет о классификации:")
        print(metrics['classification_report'])
        
        # Сохранение модели
        model_path = 'model.pkl'
        scaler_path = 'scaler.pkl'
        self.classifier.save_model(model_path, scaler_path)
        
        return metrics


def main():
    """Главная функция приложения"""
    parser = argparse.ArgumentParser(
        description='Анализ деталей вала на наличие дефектов с помощью компьютерного зрения и ML'
    )
    
    parser.add_argument(
        'mode',
        choices=['analyze', 'train'],
        help='Режим работы: analyze - анализ изображений, train - обучение модели'
    )
    
    parser.add_argument(
        '--input',
        '-i',
        required=True,
        help='Путь к изображению или директории с изображениями'
    )
    
    parser.add_argument(
        '--model',
        '-m',
        help='Путь к файлу модели (model.pkl)'
    )
    
    parser.add_argument(
        '--scaler',
        '-s',
        help='Путь к файлу масштабировщика (scaler.pkl)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        help='Директория для сохранения результатов анализа'
    )
    
    parser.add_argument(
        '--labels',
        '-l',
        help='Путь к CSV файлу с метками (для обучения)'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Не создавать визуализацию'
    )
    
    args = parser.parse_args()
    
    # Инициализация анализатора
    model_path = args.model or 'model.pkl'
    scaler_path = args.scaler or 'scaler.pkl'
    
    analyzer = ShaftDefectAnalyzer(
        model_path if os.path.exists(model_path) else None,
        scaler_path if os.path.exists(scaler_path) else None
    )
    
    if args.mode == 'train':
        # Обучение модели
        analyzer.train_model(args.input, args.labels)
    
    elif args.mode == 'analyze':
        # Анализ изображений
        output_dir = args.output or 'results'
        visualize = not args.no_visualize
        
        if os.path.isfile(args.input):
            # Анализ одного изображения
            result = analyzer.analyze_image(args.input, visualize, output_dir)
            print("\n" + "="*50)
            print("РЕЗУЛЬТАТ АНАЛИЗА:")
            print("="*50)
            print(f"Изображение: {result['image_path']}")
            if 'status' in result:
                print(f"Статус: {result['status']}")
                if 'confidence' in result:
                    print(f"Уверенность: {result['confidence']:.2%}")
        elif os.path.isdir(args.input):
            # Анализ директории
            results = analyzer.analyze_directory(args.input, visualize, output_dir)
            
            # Статистика
            if results and 'status' in results[0]:
                statuses = [r['status'] for r in results if 'status' in r]
                print("\n" + "="*50)
                print("СТАТИСТИКА АНАЛИЗА:")
                print("="*50)
                print(f"Всего проанализировано: {len(results)}")
                print(f"Исправных: {statuses.count('ИСПРАВНА')}")
                print(f"Дефектных: {statuses.count('ДЕФЕКТНА')}")
        else:
            print(f"Ошибка: Путь не существует: {args.input}")
            sys.exit(1)


if __name__ == '__main__':
    main()

