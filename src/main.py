import cv2
from ultralytics import YOLO
import psutil

def load_model(model_path='yolo11x.pt'):
    """Загрузка модели YOLO из указанного файла."""
    model = YOLO(model_path)
    return model

def resize_image(image, target_width=640):
    """Изменение размера изображения, сохраняя соотношение сторон."""
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height
    target_height = int(target_width / aspect_ratio)
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized

def preprocess_image(image_path):
    """Предварительная обработка изображения: чтение и изменение размера."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение по пути: {image_path}")
    img = resize_image(img)
    return img

def convert_to_rgb(image):
    """Преобразование изображения из BGR в RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def detect_objects(model, img):
    """Детекция объектов на изображении и отображение результатов."""
    results = model(img)
    for result in results:
        result.show() 

def process_single_image(image_path):
    """Обработка одного изображения."""
    model = load_model()  # Загрузка модели
    # Предобработка изображения
    img = preprocess_image(image_path)

    # Преобразование цвета
    img_rgb = convert_to_rgb(img)

    # Выполнение детектирования
    detect_objects(model, img_rgb)

def measure_memory_usage(function, *args):
    """Измерение использования памяти во время выполнения функции."""
    process = psutil.Process()
    
    # Начальное использование памяти
    memory_before = process.memory_info().rss  # В байтах
    
    function(*args)  # Вызов функции
    
    # Использование памяти после выполнения функции
    memory_after = process.memory_info().rss  # В байтах

    return (memory_after - memory_before) / (1024 * 1024)  # Возвращаем разницу в МБ

if __name__ == "__main__":
    # Путь к изображению
    image_path = './input_image.jpg'  # Укажите свой путь к изображению
    
    # Оценка использования памяти при обработке изображения
    memory_used = measure_memory_usage(process_single_image, image_path)

    print(f"Затраченная память на вычисления: {memory_used:.2f} МБ")