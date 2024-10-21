import cv2
from ultralytics import YOLO

def load_model(model_path='yolo11x.pt'):
    model = YOLO(model_path)
    return model

def detect_objects(model, image_path):
    # Чтение изображения
    img = cv2.imread(image_path)
    
    # Выполнение детектирования
    results = model(img)


    # Обучение модели 
    # results = model.train(data="/Users/planet-9/Documents/ITMO/3_sem/MCV/Lab_4/OD-WeaponDetection/Knife_detection/data.yaml", epochs=100, imgsz=640)

     # Перебор результатов и отображение каждого
    for result in results:
        result.show()  # Отобразить изображение с аннотациями


    # results.save('output_image.jpg')  

if __name__ == "__main__":
    # Загрузка модели
    model = load_model()
    
    image_path = './input_image.jpg'  
    detect_objects(model, image_path)
