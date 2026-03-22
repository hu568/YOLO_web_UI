from ultralytics import YOLO

# 加载焊接检测模型
model = YOLO("Weld inspection-14_YOLOn_V11.pt")

# 对示例图片进行预测
results = model("bus.jpg")

# 显示预测结果
results[0].show()

# 打印检测到的对象信息
for result in results:
    boxes = result.boxes
    for box in boxes:
        print(f"类别: {result.names[int(box.cls)]}, 置信度: {box.conf.item():.2f}, 坐标: {box.xyxy[0].tolist()}")

print("\n焊接检测模型测试成功！")