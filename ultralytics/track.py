from ultralytics import YOLO
import csv
import os
import time

model_path = "/media/lhp/LHP_SSD/jiancegenzong/YOLOv11_BotSort/ultralytics/yolo11n.pt" 
model = YOLO(model_path)
video_path = "/media/lhp/LHP_SSD/jiancegenzong/YOLOv11_BotSort/ultralytics/test01.mp4"
results = model.track(source=video_path, show=True, save=True, tracker="botsort.yaml")
try:
    save_dir = results[0].save_dir  
    csv_path = os.path.join(save_dir, "tracking_results.csv") 
except AttributeError:
    csv_path = "tracking_results.csv"  

allowed_classes = [0, 1, 2, 3, 7]  # person, bicycle, car, motorcycle, truck

# 新增：类别 ID 到名称的映射
class_map = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    7: 'truck'
}

# 将结果写入 CSV
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['track_id', 'class', 'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y', 'frame_id'])
    for frame_id, result in enumerate(results):
        if result.boxes is None or len(result.boxes) == 0:
            continue  
        
        ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []  
        classes = result.boxes.cls.cpu().numpy()  
        xyxy = result.boxes.xyxy.cpu().numpy()  
        
        for i in range(len(xyxy)):
            cls_id = int(classes[i]) if i < len(classes) else -1
            if cls_id not in allowed_classes:
                continue  
            
            cls_name = class_map.get(cls_id, 'unknown')  
            track_id = int(ids[i]) if i < len(ids) else -1  
            x1, y1, x2, y2 = xyxy[i]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            writer.writerow([track_id, cls_name, x1, y1, x2, y2, center_x, center_y, frame_id])

print(f"跟踪结果已保存到 {csv_path}")

for result in results:
    print(result)  
