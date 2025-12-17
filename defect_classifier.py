"""
Модуль машинного обучения для классификации дефектов деталей вала
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, List, Optional
import pandas as pd


class DefectClassifier:
    """Классификатор дефектов деталей вала"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Инициализация классификатора
        
        Args:
            model_type: Тип модели ('random_forest' или 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,  # Увеличено количество деревьев для большей уверенности
                max_depth=15,      # Увеличена глубина для лучшего обучения
                min_samples_split=2,  # Минимум образцов для разделения
                min_samples_leaf=1,   # Минимум образцов в листе
                class_weight='balanced',  # Балансировка классов при малом количестве данных
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                learning_rate=0.1
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        """
        Обучение модели
        
        Args:
            X: Матрица признаков (n_samples, n_features)
            y: Вектор меток (0 - исправна, 1 - дефектна)
            test_size: Доля тестовой выборки
        
        Returns:
            Словарь с метриками обучения
        """
        # Разделение на обучающую и тестовую выборки
        # Используем stratify только если в каждом классе минимум 2 образца
        min_samples_per_class = min(np.bincount(y.astype(int)))
        use_stratify = min_samples_per_class >= 2
        
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # Если данных мало, не используем стратификацию
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Масштабирование признаков
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Обучение модели
        self.model.fit(X_train_scaled, y_train)
        
        # Оценка качества
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        self.is_trained = True
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test, y_test_pred),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
        }
        
        return metrics
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Предсказание класса и вероятности
        
        Args:
            features: Вектор признаков (n_features,)
        
        Returns:
            Кортеж (класс, вероятность): 0 - исправна, 1 - дефектна
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите метод train()")
        
        # Масштабирование признаков
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Предсказание
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return int(prediction), float(probability[1])  # вероятность дефекта
    
    def predict_batch(self, features_list: List[np.ndarray]) -> List[Tuple[int, float]]:
        """
        Предсказание для нескольких изображений
        
        Args:
            features_list: Список векторов признаков
        
        Returns:
            Список кортежей (класс, вероятность)
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите метод train()")
        
        # Объединение в матрицу
        X = np.array(features_list)
        
        # Масштабирование
        X_scaled = self.scaler.transform(X)
        
        # Предсказания
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return [(int(pred), float(prob[1])) for pred, prob in zip(predictions, probabilities)]
    
    def save_model(self, model_path: str, scaler_path: str):
        """Сохранение модели и масштабировщика"""
        if not self.is_trained:
            raise ValueError("Модель не обучена. Нечего сохранять.")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Модель сохранена: {model_path}")
        print(f"Масштабировщик сохранен: {scaler_path}")
    
    def load_model(self, model_path: str, scaler_path: str):
        """Загрузка модели и масштабировщика"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Файл масштабировщика не найден: {scaler_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        print(f"Модель загружена: {model_path}")
    
    def get_feature_importance(self) -> dict:
        """Получение важности признаков"""
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = [
                # Базовые геометрические
                'area', 'perimeter', 'circularity', 'solidity',
                # Интенсивность
                'mean_intensity', 'std_intensity',
                # Текстура
                'contrast', 'homogeneity', 'energy', 'correlation',
                # Дефекты
                'edge_density', 'defect_count', 'max_defect_area',
                # Специфические дефекты вала
                'taper_ratio',      # Конусообразность
                'oval_ratio',       # Овальность
                'barrel_ratio',     # Бочкообразность
                'saddle_ratio',     # Седлообразность
                'bend_angle',       # Изгиб (прогиб)
                'size_deviation'    # Отклонение размеров
            ]
            
            importance_dict = dict(zip(feature_names, importances))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}


