import cv2
from ultralytics import YOLO

def load_model(model_path='yolo11x.pt'):
    """Загрузка модели YOLO из указанного файла."""
    model = YOLO(model_path)
    return model

def resize_image(image, target_width=640):
    """Изменение размера изображения, сохраняя соотношение сторон."""
    # Получение оригинальных размеров
    original_height, original_width = image.shape[:2]
    
    # Вычисление нового размера с сохранением соотношения сторон
    aspect_ratio = original_width / original_height
    target_height = int(target_width / aspect_ratio)
    
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized

def preprocess_image(image_path):
    """Предварительная обработка изображения: чтение, изменение размера и преобразование в RGB."""
    # Чтение изображения
    img = cv2.imread(image_path)

    # Проверка на успешное чтение изображения
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение по пути: {image_path}")

    # Изменение размера изображения
    img = resize_image(img)

    # Преобразование из BGR в RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def detect_objects(model, image_path):
    """Детекция объектов на изображении и отображение результатов."""
    # Предобработка изображения
    img = preprocess_image(image_path)

    # Выполнение детектирования
    results = model(img)

    # Перебор результатов и отображение каждого
    for result in results:
        result.show()  # Отобразить изображение с аннотациями

if __name__ == "__main__":
    # Загрузка модели
    model = load_model()
    
    image_path = './input_image2.jpg'  
    detect_objects(model, image_path)