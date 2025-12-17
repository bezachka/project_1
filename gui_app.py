"""
Графический интерфейс для анализатора дефектов деталей вала
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
import threading
from pathlib import Path
from main import ShaftDefectAnalyzer
import cv2
import numpy as np


class ShaftDefectAnalyzerGUI:
    """Графический интерфейс приложения"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор дефектов деталей вала")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Инициализация анализатора
        self.analyzer = None
        self.current_image_path = None
        self.current_result = None
        
        # Фиксированный размер для отображения всех изображений (одинаковый масштаб)
        self.display_image_width = 800
        self.display_image_height = 600
        
        # Создание интерфейса
        self.create_widgets()
        
        # Попытка загрузить модель при запуске
        self.load_model_auto()
    
    def create_widgets(self):
        """Создание виджетов интерфейса"""
        
        # Заголовок
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🔍 Анализатор дефектов деталей вала",
            font=('Arial', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Основной контейнер
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - управление
        left_panel = tk.Frame(main_container, bg='#ecf0f1', width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Правая панель - изображение и результаты
        right_panel = tk.Frame(main_container, bg='white')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # === ЛЕВАЯ ПАНЕЛЬ ===
        
        # Секция модели
        model_frame = tk.LabelFrame(
            left_panel,
            text="Модель",
            font=('Arial', 10, 'bold'),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.model_status_label = tk.Label(
            model_frame,
            text="Статус: Не загружена",
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='#e74c3c'
        )
        self.model_status_label.pack(anchor=tk.W, pady=5)
        
        tk.Button(
            model_frame,
            text="Загрузить модель",
            command=self.load_model,
            bg='#3498db',
            fg='white',
            font=('Arial', 9),
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(fill=tk.X, pady=5)
        
        tk.Button(
            model_frame,
            text="Обучить модель",
            command=self.train_model_dialog,
            bg='#27ae60',
            fg='white',
            font=('Arial', 9),
            relief=tk.FLAT,
            padx=10,
            pady=5
        ).pack(fill=tk.X, pady=5)
        
        # Секция анализа
        analysis_frame = tk.LabelFrame(
            left_panel,
            text="Анализ",
            font=('Arial', 10, 'bold'),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        analysis_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(
            analysis_frame,
            text="📁 Выбрать изображение",
            command=self.select_image,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT,
            padx=10,
            pady=10
        ).pack(fill=tk.X, pady=5)
        
        tk.Button(
            analysis_frame,
            text="🔍 Анализировать",
            command=self.analyze_current_image,
            bg='#e67e22',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT,
            padx=10,
            pady=10,
            state=tk.DISABLED
        ).pack(fill=tk.X, pady=5)
        
        self.analyze_button = None  # Будет установлен после создания
        
        tk.Button(
            analysis_frame,
            text="📂 Анализ директории",
            command=self.analyze_directory,
            bg='#16a085',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT,
            padx=10,
            pady=10
        ).pack(fill=tk.X, pady=5)
        
        # Секция результатов
        results_frame = tk.LabelFrame(
            left_panel,
            text="Результаты",
            font=('Arial', 10, 'bold'),
            bg='#ecf0f1',
            padx=10,
            pady=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.result_text = scrolledtext.ScrolledText(
            results_frame,
            height=10,
            font=('Arial', 9),
            wrap=tk.WORD,
            bg='white',
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # === ПРАВАЯ ПАНЕЛЬ ===
        
        # Вкладки
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Вкладка изображения
        image_tab = tk.Frame(notebook, bg='white')
        notebook.add(image_tab, text="Изображение")
        
        # Контейнер для изображения
        image_container = tk.Frame(image_tab, bg='#34495e')
        image_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.image_label = tk.Label(
            image_container,
            text="Выберите изображение для анализа",
            bg='#34495e',
            fg='white',
            font=('Arial', 14)
        )
        self.image_label.pack(expand=True)
        
        # Вкладка визуализации
        vis_tab = tk.Frame(notebook, bg='white')
        notebook.add(vis_tab, text="Визуализация дефектов")
        
        vis_container = tk.Frame(vis_tab, bg='#34495e')
        vis_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.vis_label = tk.Label(
            vis_container,
            text="Визуализация будет отображена после анализа",
            bg='#34495e',
            fg='white',
            font=('Arial', 14)
        )
        self.vis_label.pack(expand=True)
        
        # Сохранение ссылки на кнопку анализа
        for widget in analysis_frame.winfo_children():
            if isinstance(widget, tk.Button) and widget.cget('text') == "🔍 Анализировать":
                self.analyze_button = widget
                break
    
    def load_model_auto(self):
        """Автоматическая загрузка модели при запуске"""
        model_path = 'model.pkl'
        scaler_path = 'scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.analyzer = ShaftDefectAnalyzer(model_path, scaler_path)
                self.model_status_label.config(
                    text="Статус: Загружена ✓",
                    fg='#27ae60'
                )
            except Exception as e:
                self.model_status_label.config(
                    text=f"Статус: Ошибка загрузки",
                    fg='#e74c3c'
                )
        else:
            self.analyzer = ShaftDefectAnalyzer()
    
    def load_model(self):
        """Загрузка модели из файлов"""
        model_path = filedialog.askopenfilename(
            title="Выберите файл модели",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not model_path:
            return
        
        scaler_path = filedialog.askopenfilename(
            title="Выберите файл масштабировщика",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not scaler_path:
            return
        
        try:
            self.analyzer = ShaftDefectAnalyzer(model_path, scaler_path)
            self.model_status_label.config(
                text="Статус: Загружена ✓",
                fg='#27ae60'
            )
            messagebox.showinfo("Успех", "Модель успешно загружена!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель:\n{e}")
    
    def train_model_dialog(self):
        """Диалог обучения модели"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Обучение модели")
        dialog.geometry("600x450")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)  # Делаем диалог модальным
        
        # Заголовок
        header_frame = tk.Frame(dialog, bg='#2c3e50', height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="🎓 Обучение модели",
            font=('Arial', 16, 'bold'),
            bg='#2c3e50',
            fg='white'
        ).pack(pady=12)
        
        # Основной контент
        content_frame = tk.Frame(dialog, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Инструкция
        tk.Label(
            content_frame,
            text="Выберите директорию с данными для обучения:",
            font=('Arial', 10),
            bg='#f0f0f0',
            anchor='w'
        ).pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            content_frame,
            text="Ожидается структура: директория/good/ и директория/defect/",
            font=('Arial', 9),
            bg='#f0f0f0',
            fg='#7f8c8d',
            anchor='w'
        ).pack(fill=tk.X, pady=(0, 15))
        
        data_dir_var = tk.StringVar()
        
        # Фрейм для выбора директории
        dir_frame = tk.Frame(content_frame, bg='#f0f0f0')
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        dir_entry = tk.Entry(
            dir_frame,
            textvariable=data_dir_var,
            font=('Arial', 9),
            state='readonly',
            readonlybackground='white'
        )
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        def select_data_dir():
            dir_path = filedialog.askdirectory(title="Выберите директорию с данными")
            if dir_path:
                data_dir_var.set(dir_path)
                # Проверка структуры директории
                check_directory_structure(dir_path)
        
        select_btn = tk.Button(
            dir_frame,
            text="📁 Выбрать",
            command=select_data_dir,
            bg='#3498db',
            fg='white',
            font=('Arial', 9),
            padx=15,
            pady=5
        )
        select_btn.pack(side=tk.RIGHT)
        
        # Информация о структуре
        info_label = tk.Label(
            content_frame,
            text="",
            font=('Arial', 9),
            bg='#f0f0f0',
            fg='#27ae60',
            anchor='w',
            justify='left'
        )
        info_label.pack(fill=tk.X, pady=(0, 10))
        
        def check_directory_structure(dir_path):
            """Проверка структуры директории"""
            good_dir = os.path.join(dir_path, 'good')
            defect_dir = os.path.join(dir_path, 'defect')
            
            good_exists = os.path.exists(good_dir)
            defect_exists = os.path.exists(defect_dir)
            
            info_text = ""
            if good_exists and defect_exists:
                # Подсчет файлов
                good_files = [f for f in os.listdir(good_dir) 
                             if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
                defect_files = [f for f in os.listdir(defect_dir) 
                               if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
                
                info_text = f"✓ Структура корректна\n"
                info_text += f"  Исправных изображений: {len(good_files)}\n"
                info_text += f"  Дефектных изображений: {len(defect_files)}"
                info_label.config(fg='#27ae60', text=info_text)
            elif good_exists or defect_exists:
                info_text = "⚠️ Найдена только одна папка (good или defect)\n"
                info_text += "  Нужны обе папки: good/ и defect/"
                info_label.config(fg='#e67e22', text=info_text)
            else:
                info_text = "❌ Не найдены папки good/ и defect/\n"
                info_text += "  Создайте структуру: директория/good/ и директория/defect/"
                info_label.config(fg='#e74c3c', text=info_text)
        
        # Прогресс
        progress_label = tk.Label(
            content_frame,
            text="",
            font=('Arial', 9),
            bg='#f0f0f0',
            fg='#3498db',
            anchor='w'
        )
        progress_label.pack(fill=tk.X, pady=(10, 0))
        
        # Кнопка обучения
        train_btn = tk.Button(
            content_frame,
            text="🚀 Начать обучение",
            command=lambda: start_training(),
            bg='#27ae60',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=30,
            pady=12,
            state=tk.NORMAL
        )
        train_btn.pack(pady=(20, 0))
        
        def start_training():
            data_dir = data_dir_var.get()
            if not data_dir:
                messagebox.showwarning("Предупреждение", "Выберите директорию с данными")
                return
            
            # Проверка структуры перед обучением
            good_dir = os.path.join(data_dir, 'good')
            defect_dir = os.path.join(data_dir, 'defect')
            
            if not os.path.exists(good_dir) or not os.path.exists(defect_dir):
                messagebox.showerror(
                    "Ошибка",
                    "Не найдены папки good/ и defect/\n\n"
                    "Создайте структуру:\n"
                    f"{data_dir}/\n"
                    "  ├── good/     (изображения исправных деталей)\n"
                    "  └── defect/   (изображения дефектных деталей)"
                )
                return
            
            # Проверка наличия изображений
            good_files = [f for f in os.listdir(good_dir) 
                         if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
            defect_files = [f for f in os.listdir(defect_dir) 
                           if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
            
            if not good_files and not defect_files:
                messagebox.showerror(
                    "Ошибка",
                    "Не найдены изображения в папках good/ и defect/\n\n"
                    "Добавьте изображения в формате: .jpg, .jpeg, .png, .bmp"
                )
                return
            
            if not good_files:
                messagebox.showwarning(
                    "Предупреждение",
                    "Не найдены изображения в папке good/\n"
                    "Рекомендуется иметь изображения обоих классов"
                )
            
            if not defect_files:
                messagebox.showwarning(
                    "Предупреждение",
                    "Не найдены изображения в папке defect/\n"
                    "Рекомендуется иметь изображения обоих классов"
                )
            
            # Блокируем кнопку во время обучения
            train_btn.config(state=tk.DISABLED)
            select_btn.config(state=tk.DISABLED)
            
            def train():
                try:
                    progress_label.config(
                        text="🔄 Обучение модели... Пожалуйста, подождите.\nЭто может занять несколько минут.",
                        fg='#3498db'
                    )
                    dialog.update()
                    
                    analyzer = ShaftDefectAnalyzer()
                    metrics = analyzer.train_model(data_dir)
                    
                    self.analyzer = analyzer
                    self.model_status_label.config(
                        text="Статус: Загружена ✓",
                        fg='#27ae60'
                    )
                    
                    progress_label.config(
                        text=f"✅ Обучение завершено!\nТочность на тесте: {metrics['test_accuracy']:.2%}",
                        fg='#27ae60'
                    )
                    
                    messagebox.showinfo(
                        "Успех",
                        f"Модель успешно обучена!\n\n"
                        f"📊 Точность на тестовой выборке: {metrics['test_accuracy']:.2%}\n"
                        f"📊 Точность на обучающей выборке: {metrics['train_accuracy']:.2%}\n\n"
                        f"Модель сохранена в:\n"
                        f"  - model.pkl\n"
                        f"  - scaler.pkl"
                    )
                    
                    # Разблокируем кнопки
                    train_btn.config(state=tk.NORMAL)
                    select_btn.config(state=tk.NORMAL)
                    
                except Exception as e:
                    progress_label.config(
                        text=f"❌ Ошибка при обучении",
                        fg='#e74c3c'
                    )
                    error_msg = str(e)
                    print(f"Ошибка обучения: {error_msg}")  # Отладка
                    import traceback
                    traceback.print_exc()  # Полный traceback
                    
                    messagebox.showerror(
                        "Ошибка при обучении",
                        f"Не удалось обучить модель:\n\n{error_msg}\n\n"
                        f"Проверьте:\n"
                        f"1. Структуру директории (good/ и defect/)\n"
                        f"2. Формат изображений (.jpg, .png, .bmp)\n"
                        f"3. Доступность файлов"
                    )
                    
                    # Разблокируем кнопки
                    train_btn.config(state=tk.NORMAL)
                    select_btn.config(state=tk.NORMAL)
            
            threading.Thread(target=train, daemon=True).start()
    
    def select_image(self):
        """Выбор изображения для анализа"""
        try:
            file_path = filedialog.askopenfilename(
                title="Выберите изображение",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                print(f"Выбран файл: {file_path}")  # Отладка
                self.current_image_path = file_path
                self.display_image(file_path)
                if self.analyze_button:
                    self.analyze_button.config(state=tk.NORMAL)
                # Обновление текста в результатах
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(1.0, f"Загружено изображение: {os.path.basename(file_path)}\n")
            else:
                print("Файл не выбран")  # Отладка
        except Exception as e:
            print(f"Ошибка при выборе файла: {e}")  # Отладка
            messagebox.showerror("Ошибка", f"Ошибка при выборе файла:\n{e}")
    
    def display_image(self, image_path):
        """Отображение изображения в фиксированном масштабе без обрезки"""
        try:
            print(f"Загрузка изображения: {image_path}")  # Отладка
            
            # Проверка существования файла
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Файл не найден: {image_path}")
            
            # Загрузка изображения
            img = Image.open(image_path)
            original_size = img.size
            print(f"Исходное изображение: {original_size}")  # Отладка
            
            # Масштабирование до максимального размера с сохранением пропорций
            # Изображение будет масштабировано так, чтобы полностью поместиться в контейнер
            img.thumbnail((self.display_image_width, self.display_image_height), Image.Resampling.LANCZOS)
            scaled_size = img.size
            
            # Вычисляем масштаб для единообразия
            scale_factor = min(self.display_image_width / original_size[0], 
                             self.display_image_height / original_size[1])
            print(f"Масштабированное изображение: {scaled_size} (масштаб: {scale_factor:.3f}x)")  # Отладка
            
            # Конвертация для tkinter
            photo = ImageTk.PhotoImage(img)
            
            # Обновление метки - без фиксированных размеров, изображение отображается полностью
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Сохранение ссылки (важно!)
            
            print("Изображение успешно отображено без обрезки")  # Отладка
            
            # Обновление интерфейса
            self.root.update()
            
        except Exception as e:
            error_msg = f"Не удалось загрузить изображение:\n{e}"
            print(f"Ошибка: {error_msg}")  # Отладка
            messagebox.showerror("Ошибка", error_msg)
            import traceback
            traceback.print_exc()  # Полный traceback для отладки
    
    def analyze_current_image(self):
        """Анализ текущего изображения"""
        if not self.current_image_path:
            messagebox.showwarning("Предупреждение", "Сначала выберите изображение")
            return
        
        if not self.analyzer or not self.analyzer.classifier.is_trained:
            messagebox.showwarning(
                "Предупреждение",
                "Модель не загружена или не обучена.\nПожалуйста, загрузите или обучите модель."
            )
            return
        
        def analyze():
            try:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Анализ изображения...\nПожалуйста, подождите.\n")
                self.root.update()
                
                result = self.analyzer.analyze_image(
                    self.current_image_path,
                    visualize=True,
                    output_dir='results'
                )
                
                self.current_result = result
                
                # Отображение результатов
                result_text = f"📊 РЕЗУЛЬТАТЫ АНАЛИЗА\n"
                result_text += "=" * 40 + "\n\n"
                result_text += f"Изображение: {os.path.basename(result['image_path'])}\n\n"
                
                if 'status' in result:
                    status = result['status']
                    if status == 'ДЕФЕКТНА':
                        result_text += f"Статус: ⚠️ {status}\n"
                    elif status == 'ИСПРАВНА':
                        result_text += f"Статус: ✅ {status}\n"
                    else:
                        result_text += f"Статус: {status}\n"
                    
                    if 'confidence' in result:
                        result_text += f"Уверенность: {result['confidence']:.2%}\n"
                        # Показываем правильную вероятность в зависимости от статуса
                        if status == 'ДЕФЕКТНА':
                            prob = result.get('probability_defect', result.get('probability_display', 0))
                            result_text += f"Вероятность дефекта: {prob:.2%}\n"
                        else:
                            # Для исправных деталей показываем вероятность исправности
                            prob = result.get('probability_display', 1 - result.get('probability_defect', 0))
                            result_text += f"Вероятность исправности: {prob:.2%}\n"
                    
                    # Показываем информацию о найденных дефектах ТОЛЬКО если модель определила деталь как дефектную
                    if 'defect_indicators' in result:
                        indicators = result['defect_indicators']
                        # Показываем признаки дефектов только если статус "ДЕФЕКТНА"
                        if status == 'ДЕФЕКТНА' and indicators.get('has_defects', False):
                            result_text += f"\n🔍 Обнаруженные признаки дефектов:\n"
                            for reason in indicators.get('reasons', []):
                                result_text += f"  • {reason}\n"
                        elif status == 'ИСПРАВНА':
                            # Если модель говорит "исправна", не показываем признаки дефектов
                            # (даже если они найдены, модель считает их незначительными)
                            result_text += f"\n✅ Деталь соответствует норме\n"
                        elif indicators.get('indicators_count', 0) == 0:
                            result_text += f"\n✅ Признаков дефектов не обнаружено\n"
                
                if 'error' in result:
                    result_text += f"\nОшибка: {result['error']}\n"
                
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(1.0, result_text)
                
                # Отображение визуализации
                if 'visualization_path' in result and os.path.exists(result['visualization_path']):
                    self.display_visualization(result['visualization_path'])
                
                messagebox.showinfo("Анализ завершен", "Анализ изображения выполнен успешно!")
                
            except Exception as e:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(1.0, f"Ошибка при анализе:\n{str(e)}")
                messagebox.showerror("Ошибка", f"Ошибка при анализе:\n{e}")
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def display_visualization(self, vis_path):
        """Отображение визуализации дефектов в фиксированном масштабе без обрезки"""
        try:
            img = Image.open(vis_path)
            original_size = img.size
            print(f"Исходная визуализация: {original_size}")  # Отладка
            
            # Масштабирование до максимального размера с сохранением пропорций
            # Изображение будет масштабировано так, чтобы полностью поместиться в контейнер
            img.thumbnail((self.display_image_width, self.display_image_height), Image.Resampling.LANCZOS)
            scaled_size = img.size
            
            # Вычисляем масштаб для единообразия
            scale_factor = min(self.display_image_width / original_size[0], 
                             self.display_image_height / original_size[1])
            print(f"Масштабированная визуализация: {scaled_size} (масштаб: {scale_factor:.3f}x)")  # Отладка
            
            # Конвертация для tkinter
            photo = ImageTk.PhotoImage(img)
            
            # Обновление метки - без фиксированных размеров, изображение отображается полностью
            self.vis_label.config(image=photo, text="")
            self.vis_label.image = photo
            
            print("Визуализация успешно отображена без обрезки")  # Отладка
            
        except Exception as e:
            print(f"Ошибка при отображении визуализации: {e}")
    
    def analyze_directory(self):
        """Анализ всех изображений в директории"""
        dir_path = filedialog.askdirectory(title="Выберите директорию с изображениями")
        
        if not dir_path:
            return
        
        if not self.analyzer or not self.analyzer.classifier.is_trained:
            messagebox.showwarning(
                "Предупреждение",
                "Модель не загружена или не обучена.\nПожалуйста, загрузите или обучите модель."
            )
            return
        
        output_dir = filedialog.askdirectory(title="Выберите директорию для сохранения результатов")
        if not output_dir:
            output_dir = 'results'
        
        def analyze():
            try:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Анализ директории: {dir_path}\n")
                self.result_text.insert(tk.END, "Обработка изображений...\n\n")
                self.root.update()
                
                results = self.analyzer.analyze_directory(dir_path, visualize=True, output_dir=output_dir)
                
                # Статистика
                if results:
                    statuses = [r.get('status', 'UNKNOWN') for r in results]
                    total = len(results)
                    good = statuses.count('ИСПРАВНА')
                    defect = statuses.count('ДЕФЕКТНА')
                    
                    result_text = f"📊 РЕЗУЛЬТАТЫ ПАКЕТНОГО АНАЛИЗА\n"
                    result_text += "=" * 40 + "\n\n"
                    result_text += f"Всего проанализировано: {total}\n"
                    result_text += f"✅ Исправных: {good}\n"
                    result_text += f"⚠️ Дефектных: {defect}\n\n"
                    result_text += "Детальные результаты:\n" + "-" * 40 + "\n"
                    
                    for i, result in enumerate(results, 1):
                        filename = os.path.basename(result.get('image_path', 'unknown'))
                        status = result.get('status', 'UNKNOWN')
                        if 'confidence' in result:
                            conf = result['confidence']
                            result_text += f"{i}. {filename}: {status} ({conf:.2%})\n"
                        else:
                            result_text += f"{i}. {filename}: {status}\n"
                    
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(1.0, result_text)
                    
                    messagebox.showinfo(
                        "Анализ завершен",
                        f"Обработано изображений: {total}\n"
                        f"Исправных: {good}\n"
                        f"Дефектных: {defect}"
                    )
                
            except Exception as e:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(1.0, f"Ошибка при анализе:\n{str(e)}")
                messagebox.showerror("Ошибка", f"Ошибка при анализе:\n{e}")
        
        threading.Thread(target=analyze, daemon=True).start()


def main():
    """Запуск графического приложения"""
    root = tk.Tk()
    app = ShaftDefectAnalyzerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

