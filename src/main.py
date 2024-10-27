import cv2
from ultralytics import YOLO
from concurrent.futures import ProcessPoolExecutor
import time

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
        result.show()  # Отобразить изображение с аннотациями

def process_single_image(image_path):
    """Обработка одного изображения с использованием процессов."""
    model = load_model()  # Загрузка модели внутри процесса
    # Предобработка изображения
    img = preprocess_image(image_path)

    # Преобразование цвета
    img_rgb = convert_to_rgb(img)

    # Выполнение детектирования
    detect_objects(model, img_rgb)

if __name__ == "__main__":
    # Путь к изображению
    image_path = './input_image2.jpg'  # Укажите свой путь к изображению
    
    # Обработка одного изображения с использованием многопроцессорности
    start_time = time.time()
    
    with ProcessPoolExecutor() as executor:
        executor.submit(process_single_image, image_path)

    elapsed_time = time.time() - start_time
    print(f"Время выполнения: {elapsed_time:.2f} секунд")