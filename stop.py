import cv2
import os
import numpy as np
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
def visualize_perspective_points(image, src_points):
    """在原始图像上可视化透视变换的四个点"""
    result = image.copy()
    
    # 绘制四个点和连接线
    for i, point in enumerate(src_points):
        # 绘制点
        cv2.circle(result, point, 5, (0, 0, 255), -1)
        
        # 添加标签
        cv2.putText(result, f"P{i+1}", 
                   (point[0]+10, point[1]+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 绘制连接线
    for i in range(4):
        cv2.line(result, src_points[i], src_points[(i+1)%4], (0, 255, 255), 2)
    
    return result

def merge_similar_lines(lines, rho_threshold=20, theta_threshold=np.pi / 36):
    merged_lines = []
    if lines is None:
        return merged_lines

    for rho, theta in lines[:, 0]:
        add_new = True
        for i, (rho_avg, theta_avg, count) in enumerate(merged_lines):
            # 如果 rho 和 theta 足够接近，就合并
            if abs(rho - rho_avg) < rho_threshold and abs(theta - theta_avg) < theta_threshold:
                merged_lines[i] = ((rho_avg * count + rho) / (count + 1), 
                                   (theta_avg * count + theta) / (count + 1), 
                                   count + 1)
                add_new = False
                break
        
        if add_new:
            merged_lines.append((rho, theta, 1))

    return [(rho, theta) for rho, theta, _ in merged_lines]

def perspective_transform(image, src_points, dst_size):
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Define destination points for the top-down view
    width, height = dst_size
    dst_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
    
    # Color filtering for white lines
    lower = np.array([220, 180, 180])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)
    
    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32), dst_points)
    inv_matrix = cv2.getPerspectiveTransform(dst_points, np.array(src_points, dtype=np.float32))  # 计算逆变换矩阵
    
    # Apply the perspective warp
    transformed = cv2.warpPerspective(mask, matrix, (width, height))
    
    # 形態學處理
    kernel_dilate = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((10, 10), np.uint8)
    
    # 先進行膨脹操作，填補小的空隙
    dilated = cv2.dilate(transformed, kernel_dilate, iterations=1)
    
    # 再進行閉運算，連接相近的區域並平滑邊緣
    r = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
    return r, inv_matrix, transformed  # 返回原始变换后的图像

def draw_line(original_img, img, inv_matrix, iterations=1):
    gray_img = img.copy()
    result_img = original_img.copy()  # 創建結果圖像副本
    debug_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # 創建調試圖像，確保是彩色的

    # 預處理：高斯濾波
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 1)

    # Canny 邊緣檢測
    dst_img = cv2.Canny(gray_img, 30, 100)

    # 霍夫變換（檢測直線）
    lines = cv2.HoughLines(dst_img, 1, np.pi / 180, 50, None, 0, 0)
    
    # 沒有檢測到線條時直接返回原圖
    if lines is None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
        return original_img, [], img_rgb
        
    # 合併相似直線
    merged_lines = merge_similar_lines(lines)

    # 獲取圖像寬度，定義停止線的長度標準（圖像寬度的九成）
    height, width = img.shape[:2]
    min_line_length = width * 0.9  # 停止線的長度標準
    transformed_lines = []
    
    # 繪製並檢測停止線
    for rho, theta in merged_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = rho * a
        y0 = rho * b
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        
        # 計算直線的長度
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # 判斷是否為水平直線並且長度符合停止線標準
        if abs(y2 - y1) < abs(x2 - x1) * 0.1 and line_length >= min_line_length:
            transformed_lines.append(((x1, y1), (x2, y2)))
            # 在調試圖像上繪製所有霍夫變換檢測到的水平線（藍色）
            cv2.line(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 插入相鄰線段的中點線
    for _ in range(iterations):
        midpoint_lines = []
        transformed_lines.sort(key=lambda line: line[0][1])  # 以 y 坐標排序（從上到下）

        for i in range(len(transformed_lines) - 1):
            (x1, y1), (x2, y2) = transformed_lines[i]
            (nx1, ny1), (nx2, ny2) = transformed_lines[i + 1]

            # 計算中點
            mid_x1 = (x1 + nx1) // 2
            mid_y1 = (y1 + ny1) // 2
            mid_x2 = (x2 + nx2) // 2
            mid_y2 = (y2 + ny2) // 2

            midpoint_lines.append(((mid_x1, mid_y1), (mid_x2, mid_y2)))
            cv2.line(debug_img, (mid_x1, mid_y1), (mid_x2, mid_y2), (0, 255, 255), 2)  # 黃色

        transformed_lines.extend(midpoint_lines)  # 添加新計算出的中間線

    # 白色區域篩選
    valid_lines = []
    for (x1, y1), (x2, y2) in transformed_lines:
        # 創建黑色遮罩
        mask = np.zeros_like(img)
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)  # 在遮罩上畫出白線
        
        white_pixels = cv2.countNonZero(cv2.bitwise_and(img, mask))  # 計算線段上的白色像素
        total_pixels = cv2.countNonZero(mask)  # 計算整條線段的像素數
        
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
        
        # 僅保留白色比例 >= 0.8 的線段
        if white_ratio >= 0.8:
            valid_lines.append(((x1, y1), (x2, y2)))
            # 在調試圖像上繪製通過白色篩選的線段（紅色）
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 繪製所有保留的線段
    transformed_result_lines = []  # 存儲逆變換後的線段，用於返回
    for (x1, y1), (x2, y2) in valid_lines:
        # 逆透視變換回原始圖像
        pts = np.array([[[x1, y1]], [[x2, y2]]], dtype=np.float32)  
        restored_pts = cv2.perspectiveTransform(pts, inv_matrix)  
        (rx1, ry1), (rx2, ry2) = restored_pts.reshape(2, 2).astype(int)

        # 畫出恢復後的直線
        cv2.line(result_img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        transformed_result_lines.append(((rx1, ry1), (rx2, ry2)))
    
    # 在調試圖像上添加白色比例閾值說明
    cv2.putText(debug_img, "White ratio >= 0.8", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 將原始二值圖像轉換為RGB以便與調試信息合併
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    
    # 合併原始二值圖像和調試圖像
    alpha = 0.7
    beta = 0.3
    debug_combined = cv2.addWeighted(img_rgb, alpha, debug_img, beta, 0)
    
    return result_img, transformed_result_lines, debug_combined

def process_frame_objects(frame, traffic_light_model, car_model, font):
    """
    將 OpenCV 讀取的 frame (BGR) 轉成 PIL Image 處理，
    將圖片分成 9 個區域，每個區域分別進行紅綠燈偵測，
    並在整張圖進行車輛偵測，
    最後把結果標記回原圖後轉回 OpenCV 格式 (BGR)。
    """
    # 轉換色彩空間並建立 PIL 影像
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 取得圖片大小與劃分 3x3 區域
    img_width, img_height = pil_img.size
    rows, cols = 3, 3
    width_per_part = img_width // cols
    height_per_part = img_height // rows

    # 定義紅綠燈 YOLO 的類別與決策
    light_class_names = ["red", "green", "front", "left", "right"]
    light_class_decisions = {
        "red": "red",
        "green": "green",
        "front": "front",
        "left": "left",
        "right": "right"
    }

    # 車輛偵測類別
    car_class_names = ["car", "truck", "bus", "motorcycle"]

    # 首先進行車輛偵測 (在整張圖上)
    car_results = car_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=0.3)
    detected_cars = []
    
    # 創建一個遮罩圖像來遮蓋車輛區域（與原圖相同大小）
    mask_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    if car_results and car_results[0].boxes is not None:
        for box in car_results[0].boxes:
            # 取得車輛邊框座標
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            
            if 0 <= class_id < len(car_class_names):
                car_label = car_class_names[class_id]
                detected_cars.append((x1, y1, x2, y2, car_label, confidence))
                
                # 畫出車輛邊框 (藍色)
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
                
                # 在遮罩圖上用黑色填充車輛區域
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), (0, 0, 0), -1)
                
                # 設定文字與計算文字大小
                text = f"{car_label} ({confidence:.2f})"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 根據空間決定文字顯示位置
                if y1 - text_height - 5 < 0:
                    text_x = x1
                    text_y = y2 + 5
                else:
                    text_x = x1
                    text_y = y1 - text_height - 5
                
                # 畫出文字背景
                draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill="blue")
                draw.text((text_x, text_y), text, fill="white", font=font)

    # 將 PIL 影像轉為 OpenCV 格式以便對車輛區域進行遮罩處理
    frame_for_light_detection = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # 應用遮罩：只有在車輛區域是黑色的，其餘區域保持原樣
    # 創建邏輯遮罩（非零為 True）
    non_black_mask = np.any(mask_img != 0, axis=2)
    
    # 將遮罩擴展為三通道
    non_black_mask_3channel = np.stack([non_black_mask] * 3, axis=2)
    
    # 使用邏輯索引更新原圖：只在非黑色區域保留原圖
    frame_for_light_detection[~non_black_mask_3channel] = frame[~non_black_mask_3channel]
    
    # 在黑色區域（車輛區域）使用遮罩值
    frame_for_light_detection[non_black_mask_3channel] = mask_img[non_black_mask_3channel]
    
    # 將處理後的圖像轉回 PIL 格式以便繼續處理
    pil_img_masked = Image.fromarray(cv2.cvtColor(frame_for_light_detection, cv2.COLOR_BGR2RGB))

    # 依序處理每個分割區域進行紅綠燈偵測
    detected_lights = []
    for i in range(rows):
        for j in range(cols):
            # 計算目前區域的邊界
            left = j * width_per_part
            upper = i * height_per_part
            right = left + width_per_part
            lower = upper + height_per_part

            # 裁剪區域影像 (使用已遮罩處理過的圖像)
            part = pil_img_masked.crop((left, upper, right, lower))

            # 進行紅綠燈 YOLO 偵測
            results = traffic_light_model(part, conf=0.25)
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    # 取得當前區域內的邊框座標
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    
                    if class_id < 0 or class_id >= len(light_class_names):
                        label = "Unknown"
                        decision = "❓ 無法判斷"
                    else:
                        label = light_class_names[class_id]
                        decision = light_class_decisions[label]

                    # 計算全圖座標 (加上區域的左上角 offset)
                    global_x1 = left + x1
                    global_y1 = upper + y1
                    global_x2 = left + x2
                    global_y2 = upper + y2
                    
                    detected_lights.append((global_x1, global_y1, global_x2, global_y2, label, decision, confidence))

                    # 畫出邊框 (黃色) - 在原始的 pil_img 上繪製
                    draw.rectangle([global_x1, global_y1, global_x2, global_y2], outline="yellow", width=3)

                    # 設定文字與計算文字大小
                    text = f"{label} - {decision} ({confidence:.2f})"
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # 根據空間決定文字顯示位置
                    if global_y1 - text_height - 5 < 0:
                        text_x = global_x1
                        text_y = global_y2 + 5
                    else:
                        text_x = global_x1
                        text_y = global_y1 - text_height - 5

                    # 畫出文字背景
                    draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill="black")
                    draw.text((text_x, text_y), text, fill="yellow", font=font)

    # 轉回 OpenCV 格式 (RGB -> BGR)
    processed_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return processed_frame, detected_cars, detected_lights

def check_car_at_stop_line(cars, stop_lines, threshold=50):
    """检查车辆是否接近或停在停止线上"""
    cars_at_stop_line = []
    
    for car_info in cars:
        x1, y1, x2, y2, car_type, conf = car_info
        car_bottom_center = (x1 + x2) // 2, y2  # 车辆底部中心点
        
        for line in stop_lines:
            (lx1, ly1), (lx2, ly2) = line
            
            # 计算车辆底部中心点到线段的距离
            # 简化：仅检查车辆底部中心点的 y 坐标与线段 y 坐标的差距
            if min(ly1, ly2) - threshold <= car_bottom_center[1] <= max(ly1, ly2) + threshold:
                # 检查 x 坐标是否在线段范围内
                if min(lx1, lx2) <= car_bottom_center[0] <= max(lx1, lx2):
                    cars_at_stop_line.append((car_info, line))
                    break
    
    return cars_at_stop_line

def process_video(video_path, output_path, src_points=None, dst_size=None, show_debug=True, show_perspective=True):
    """
    整合处理视频，同时执行交通灯检测、车辆检测、停止线检测和车牌识别
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 处理进度显示
    start_time = time.time()
    last_update_time = start_time
    
    # 如果没有指定透视变换参数，使用预设值
    if src_points is None:
        src_points = [(777, 615), (930, 619), (1263, 969), (385, 957)]
    if dst_size is None:
        dst_size = (300, 400)
    
    # 延迟创建debug视频写入器
    debug_out = None
    perspective_out = None
    debug_path = None
    perspective_path = None
    
    # 载入YOLO模型
    traffic_light_model = YOLO("best.pt")
    car_model = YOLO("yolo11n.pt")
    
    # 载入车牌识别模型
    license_plate_model = YOLO("D:\\taiwan_licence_dataset\\runs\\detect\\train4\\weights\\best.pt")
    
    # 初始化PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # 载入中文字体
    font_path = r"SimHei.ttf"
    font_size = 20
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Warning: Font file {font_path} not found. Using default font.")
        font = ImageFont.load_default()
    
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理当前帧
        try:
            # 1. 先进行物体检测（红绿灯和车辆）
            obj_result_frame, detected_cars, detected_lights = process_frame_objects(
                frame, traffic_light_model, car_model, font
            )
            
            # 2. 进行停止线检测
            if show_perspective:
                transformed_image, inv_matrix, transformed_view = perspective_transform(frame, src_points, dst_size)
                stop_line_frame, stop_lines, debug_img = draw_line(obj_result_frame, transformed_image, inv_matrix)
                
                # 检查车辆是否在停止线附近
                cars_at_line = check_car_at_stop_line(detected_cars, stop_lines)
                
                # 在结果帧上标记车辆与停止线的关系
                result_frame = stop_line_frame.copy()
                
                for (car_info, line) in cars_at_line:
                    x1, y1, x2, y2, car_type, _ = car_info
                    (lx1, ly1), (lx2, ly2) = line
                    
                    # 在车辆和停止线之间画一条连接线
                    car_bottom = ((x1 + x2) // 2, y2)
                    line_center = ((lx1 + lx2) // 2, (ly1 + ly2) // 2)
                    cv2.line(result_frame, car_bottom, line_center, (255, 0, 255), 2)  # 紫色
                    
                    # 添加警告文字
                    cv2.putText(result_frame, "CAR AT STOP LINE!", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 截取车辆区域进行车牌识别
                    car_roi = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                    
                    # 进行车牌识别
                    if car_roi.size > 0:
                        text, confidence, plate_box = recognize_license_plate(car_roi, license_plate_model, ocr)
                        
                        if text and confidence:
                            # 计算车牌在原始图像中的位置
                            if plate_box:
                                px1, py1, px2, py2 = plate_box
                                # 调整坐标到原始图像中的位置
                                px1 += x1
                                py1 += y1
                                px2 += x1
                                py2 += y1
                                
                                # 绘制车牌边界框（绿色）
                                cv2.rectangle(result_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                            
                            # 添加车牌识别结果文字
                            cv2.putText(result_frame, f"Plate: {text} ({confidence:.2f})", 
                                       (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 在帧上添加帧号和检测到的信息
                cv2.putText(result_frame, f"Frame: {frame_index}, Cars: {len(detected_cars)}, Lights: {len(detected_lights)}, Lines: {len(stop_lines)}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # 如果不进行透视变换，直接使用物体检测结果
                result_frame = obj_result_frame
                cv2.putText(result_frame, f"Frame: {frame_index}, Cars: {len(detected_cars)}, Lights: {len(detected_lights)}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 写入处理后的帧
            out.write(result_frame)
            
            # 创建debug视频写入器（仅在第一帧时）
            if show_debug and show_perspective and debug_out is None and debug_img is not None:
                debug_height, debug_width = debug_img.shape[:2]
                debug_path = output_path.replace('.mp4', '_debug.mp4')
                debug_out = cv2.VideoWriter(debug_path, fourcc, fps, (debug_width, debug_height))
                print(f"Debug video size set to: {debug_width}x{debug_height}")
            
            # 创建透视变换视频写入器（仅在第一帧时）
            if show_perspective and perspective_out is None and transformed_view is not None:
                transformed_view_rgb = cv2.cvtColor(transformed_view, cv2.COLOR_GRAY2BGR)
                perspective_height, perspective_width = transformed_view_rgb.shape[:2]
                perspective_path = output_path.replace('.mp4', '_perspective.mp4')
                perspective_out = cv2.VideoWriter(perspective_path, fourcc, fps, (perspective_width, perspective_height))
                print(f"Perspective video size set to: {perspective_width}x{perspective_height}")
            
            # 如果启用了调试模式并且debug_out已创建，写入调试图像
            if show_debug and show_perspective and debug_out is not None and debug_img is not None:
                # 确保debug_img是彩色图像
                if len(debug_img.shape) == 2 or debug_img.shape[2] == 1:
                    debug_img_color = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
                else:
                    debug_img_color = debug_img
                debug_out.write(debug_img_color)
            
            # 写入透视变换视频
            if show_perspective and perspective_out is not None and transformed_view is not None:
                transformed_view_rgb = cv2.cvtColor(transformed_view, cv2.COLOR_GRAY2BGR)
                perspective_out.write(transformed_view_rgb)
            
            # 更新进度显示（每秒更新一次）
            current_time = time.time()
            if current_time - last_update_time >= 1.0:
                elapsed = current_time - start_time
                percent_complete = (frame_index + 1) / frame_count * 100
                estimated_total = elapsed / (percent_complete / 100) if percent_complete > 0 else 0
                remaining = estimated_total - elapsed
                
                print(f"Processing: {percent_complete:.1f}% complete. "
                      f"Frame {frame_index+1}/{frame_count}. "
                      f"Est. remaining: {remaining:.1f}s")
                last_update_time = current_time
                
        except Exception as e:
            print(f"Error processing frame {frame_index}: {str(e)}")
            # 如果处理出错，写入原始帧
            out.write(frame)
        
        frame_index += 1
    
    # 释放资源
    cap.release()
    out.release()
    if debug_out is not None:
        debug_out.release()
    if perspective_out is not None:
        perspective_out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete. Output saved to {output_path}")
    if debug_path:
        print(f"Debug video saved to {debug_path}")
    if perspective_path:
        print(f"Perspective video saved to {perspective_path}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

        
def recognize_license_plate(car_image, license_plate_model, ocr):
    try:
        # 转换为PIL图像用于OCR
        pil_image = Image.fromarray(cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB))
        
        # 使用YOLO模型检测车牌位置
        results = license_plate_model(car_image)
        
        if results and len(results) > 0:
            # 获取车牌边界框
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            
            # 处理每个检测到的车牌
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(int, box)
                
                # 检查边界是否在图像内
                if x1 >= 0 and y1 >= 0 and x2 <= car_image.shape[1] and y2 <= car_image.shape[0]:
                    # 裁剪车牌区域
                    cropped_img = pil_image.crop((x1, y1, x2, y2))
                    cropped_img_np = np.array(cropped_img)
                    
                    # 对裁剪区域执行OCR
                    result = ocr.ocr(cropped_img_np, cls=True)
                    
                    if result and result[0]:  # 检查结果是否非空
                        data = result[0][0][1]  # 提取识别的文字和置信度
                        text, confidence = data
                        return text, confidence, (x1, y1, x2, y2)
        
        return None, None, None
    except Exception as e:
        print(f"Error in license plate recognition: {str(e)}")
        return None, None, None
# 使用示例
if __name__ == "__main__":
    video_path = "input3.mp4"
    output_path = "output_combined.mp4"
    
    # 透视变换的参数设置
    src_points = [(912, 625), (1156, 630), (1503, 823), (735, 819)]
    dst_size = (300, 400)
    
    # 处理视频，设置show_debug=True显示处理过程，show_perspective=True启用透视变换和停止线检测
    process_video(video_path, output_path, src_points, dst_size, show_debug=True, show_perspective=True)