import base64
import io
import os
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 加载预训练的YOLO模型
model = YOLO('yolo11x.pt')

# COCO数据集类别ID映射（英文）
COCO_CLASSES_EN = {
    0: 'person',  # 人
    1: 'bicycle',  # 自行车
    2: 'car',  # 汽车
    3: 'motorcycle',  # 摩托车
    5: 'bus',  # 公交车
    7: 'truck',  # 卡车
    9: 'traffic light',  # 红绿灯
    10: 'fire hydrant',  # 消防栓
    11: 'stop sign',  # 停止标志
    13: 'parking meter',  # 停车计费
    14: 'bench',  # 长凳
    15: 'bird',  # 鸟
    16: 'cat',  # 猫
    17: 'dog',  # 狗
    18: 'horse',  # 马
    19: 'sheep',  # 羊
    20: 'cow',  # 牛
    21: 'elephant',  # 象
    22: 'bear',  # 熊
    23: 'zebra',  # 斑马
    24: 'giraffe',  # 长颈鹿
    25: 'backpack',  # 背包
    27: 'umbrella',  # 雨伞
    28: 'handbag',  # 手提包
    31: 'tie',  # 领带
    33: 'frisbee',  # 飞盘
    34: 'skis',  # 滑雪板
    35: 'snowboard',  # 单板滑雪
    36: 'sports ball',  # 运动球
    37: 'kite',  # 风筝
    38: 'baseball bat',  # 棒球棒
    39: 'baseball glove',  # 棒球手套
    40: 'skateboard',  # 滑板
    41: 'surfboard',  # 水上滑雪板
    42: 'tennis racket',  # 网球拍
    43: 'bottle',  # 瓶子
    44: 'wine glass',  # 酒杯
    46: 'cup',  # 杯子
    47: 'fork',  # 叉子
    48: 'knife',  # 刀子
    49: 'spoon',  # 勺子
    50: 'bowl',  # 碗
    51: 'banana',  # 香蕉
    52: 'apple',  # 苹果
    53: 'sandwich',  # 三明治
    54: 'orange',  # 橘子
    55: 'broccoli',  # 西兰花
    57: 'hot dog',  # 热狗
    58: 'pizza',  # 披萨
    59: 'donut',  # 甜甜圈
    60: 'cake',  # 蛋糕
    61: 'chair',  # 椅子
    62: 'couch',  # 沙发
    63: 'potted plant',  # 盆栽
    64: 'bed',  # 床
    65: 'dining table',  # 餐桌
    67: 'toilet',  # 厕所
    70: 'tv',  # 电视
    72: 'laptop',  # 笔记本电脑
    73: 'mouse',  # 鼠标
    74: 'remote',  # 遥控器
    75: 'keyboard',  # 键盘
    76: 'cell phone',  # 手机
    77: 'microwave',  # 微波炉
    78: 'oven',  # 烤箱
    79: 'toaster',  # 烤面包机
    80: 'sink',  # 水槽
    81: 'refrigerator',  # 冰箱
    82: 'book',  # 书
    84: 'clock',  # 钟
    85: 'vase',  # 花瓶
    86: 'scissors',  # 剪刀
    87: 'teddy bear',  # 泰迪熊
    88: 'hair drier',  # 吹风机
    89: 'toothbrush'  # 牙刷
}

# COCO数据集类别ID映射（中文）
COCO_CLASSES_CN = {
    0: '人',
    1: '自行车',
    2: '车',
    3: '摩托车',
    5: '公交车',
    7: '卡车',
    9: '红绿灯',
    10: '消防栓',
    11: '停止标志',
    13: '停车计费器',
    14: '长凳',
    15: '鸟',
    16: '猫',
    17: '狗',
    18: '马',
    19: '羊',
    20: '牛',
    21: '大象',
    22: '熊',
    23: '斑马',
    24: '长颈鹿',
    25: '背包',
    27: '雨伞',
    28: '手提包',
    31: '领带',
    33: '飞盘',
    34: '滑雪板',
    35: '单板滑雪',
    36: '运动球',
    37: '风筝',
    38: '棒球棒',
    39: '棒球手套',
    40: '滑板',
    41: '冲浪板',
    42: '网球拍',
    43: '瓶子',
    44: '酒杯',
    46: '杯子',
    47: '叉子',
    48: '刀子',
    49: '勺子',
    50: '碗',
    51: '香蕉',
    52: '苹果',
    53: '三明治',
    54: '橘子',
    55: '西兰花',
    57: '热狗',
    58: '披萨',
    59: '甜甜圈',
    60: '蛋糕',
    61: '椅子',
    62: '沙发',
    63: '盆栽',
    64: '床',
    65: '餐桌',
    67: '厕所',
    70: '电视',
    72: '笔记本电脑',
    73: '鼠标',
    74: '遥控器',
    75: '键盘',
    76: '手机',
    77: '微波炉',
    78: '烤箱',
    79: '烤面包机',
    80: '水槽',
    81: '冰箱',
    82: '书',
    84: '钟',
    85: '花瓶',
    86: '剪刀',
    87: '泰迪熊',
    88: '吹风机',
    89: '牙刷'
}

# 为不同类别定义默认颜色（BGR格式，用于OpenCV）
DEFAULT_CLASS_COLORS = {
    0: (0, 255, 0),  # 人 - 绿色
    2: (0, 0, 255),  # 车 - 红色
    1: (255, 0, 0),  # 自行车 - 蓝色
    3: (255, 0, 255),  # 摩托车 - 紫色
    5: (0, 255, 255),  # 公交车 - 黄色
    7: (255, 255, 0),  # 卡车 - 青色
    76: (128, 0, 128),  # 手机 - 紫色
    62: (0, 128, 128),  # 沙发 - 青绿色
}


def get_contrast_color(bg_color_rgb):
    """
    根据背景颜色计算最佳对比文字颜色
    使用WCAG 2.0标准的相对亮度计算
    """
    # 将RGB值归一化到0-1范围
    r, g, b = [x / 255.0 for x in bg_color_rgb]

    # 计算线性RGB值
    r_linear = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    g_linear = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    b_linear = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4

    # 计算相对亮度
    relative_luminance = 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear

    # 根据WCAG标准，如果背景亮度 > 0.179，使用黑色文字，否则使用白色文字
    # 这个阈值确保了4.5:1的对比度（AA级标准）
    if relative_luminance > 0.179:
        return (0, 0, 0)  # 黑色
    else:
        return (255, 255, 255)  # 白色


def calculate_text_color(bg_color_rgb):
    """
    根据背景颜色计算文字颜色（黑或白）
    使用亮度公式：Y = 0.299*R + 0.587*G + 0.114*B
    如果亮度 > 160，则使用黑色文字，否则使用白色文字
    阈值调整为160以更好地处理黄色等亮色
    """
    r, g, b = bg_color_rgb
    # 计算相对亮度
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    # 调试输出
    print(f"颜色 RGB: {bg_color_rgb}, 亮度: {brightness:.1f}, {'黑色文字' if brightness > 160 else '白色文字'}")
    return (0, 0, 0) if brightness > 160 else (255, 255, 255)


def draw_custom_annotations(image, detections, use_chinese=True, custom_colors=None, line_width=2, font_size=0.5):
    """
    自定义绘制检测框和标签
    """
    # 将OpenCV图像(BGR)转换为PIL图像(RGB)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    # 尝试加载支持中文的字体
    try:
        # Windows系统常用中文字体
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',    # 黑体
            'C:/Windows/Fonts/msyh.ttc',      # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',    # 宋体
            '/System/Library/Fonts/PingFang.ttc',  # macOS
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Linux
        ]

        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break

        if font_path:
            font = ImageFont.truetype(font_path, int(14 * font_size))
        else:
            font = ImageFont.load_default()
    except Exception as e:
        print(f"加载字体失败: {e}")
        font = ImageFont.load_default()

    # 遍历所有检测结果
    for box, class_id, confidence in zip(detections['boxes'], detections['classes'], detections['confidences']):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(class_id)
        confidence = float(confidence)

        # 获取类别名称
        class_name = COCO_CLASSES_CN[class_id] if use_chinese else COCO_CLASSES_EN[class_id]
        label = f"{class_name} {confidence:.2f}"

        # 获取颜色
        if custom_colors and class_id in custom_colors:
            # 自定义颜色 (BGR -> RGB转换)
            b, g, r = custom_colors[class_id]
            color_rgb = (r, g, b)
        elif class_id in DEFAULT_CLASS_COLORS:
            # 默认颜色 (BGR -> RGB转换)
            b, g, r = DEFAULT_CLASS_COLORS[class_id]
            color_rgb = (r, g, b)
        else:
            # 随机生成颜色
            np.random.seed(class_id)
            color = np.random.randint(0, 255, 3)
            color_rgb = (int(color[2]), int(color[1]), int(color[0]))

        # 调试：打印公交车颜色
        if class_id == 5:  # 公交车
            print(f"公交车颜色 (RGB): {color_rgb}")
            # 手动验证黄色背景的文字颜色
            if color_rgb == (255, 255, 0):  # 标准黄色
                print("检测到标准黄色背景，强制使用黑色文字")
                text_color = (0, 0, 0)
            else:
                text_color = get_contrast_color(color_rgb)
        else:
            text_color = get_contrast_color(color_rgb)

        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=line_width)

        # 绘制标签背景
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 确保标签背景在图像范围内
        bg_x1 = max(0, x1)
        bg_y1 = max(0, y1 - text_height - 4)
        bg_x2 = min(img_pil.width, x1 + text_width + 8)
        bg_y2 = min(img_pil.height, y1)

        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color_rgb, outline=(0, 0, 0))

        # 绘制标签文字
        draw.text((x1 + 2, y1 - text_height - 2), label, font=font, fill=text_color)

    # 将PIL图像转换回OpenCV格式(BGR)
    annotated_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return annotated_img


def parse_color_param(color_param):
    """
    解析颜色参数，支持多种格式：
    - "255,0,0" (BGR格式)
    - "red", "green", "blue" 等预定义颜色
    - "#FF0000" (十六进制)

    Returns:
        color_dict: {class_id: (b,g,r)} 或 None
    """
    if not color_param:
        return None

    try:
        # 预定义颜色映射 (BGR格式)
        color_map = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'orange': (0, 165, 255),
            'purple': (128, 0, 128),
            'pink': (203, 192, 255)
        }

        # 如果是预定义颜色名称
        if color_param.lower() in color_map:
            base_color = color_map[color_param.lower()]
            # 为所有类别使用同一种颜色
            return {class_id: base_color for class_id in COCO_CLASSES_CN.keys()}

        # 如果是十六进制格式 #RRGGBB
        if color_param.startswith('#') and len(color_param) >= 7:
            hex_color = color_param[1:7]
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            base_color = (b, g, r)  # 转换为BGR
            return {class_id: base_color for class_id in COCO_CLASSES_CN.keys()}

        # 如果是BGR格式 "b,g,r"
        color_values = [int(x.strip()) for x in color_param.split(',') if x.strip().isdigit()]
        if len(color_values) == 3:
            base_color = tuple(color_values)  # (b, g, r)
            return {class_id: base_color for class_id in COCO_CLASSES_CN.keys()}

        # 如果是类别特定颜色 "0:255,0,0;2:0,0,255"
        color_dict = {}
        class_color_pairs = color_param.split(';')
        for pair in class_color_pairs:
            if ':' in pair:
                class_str, color_str = pair.split(':', 1)
                class_id = int(class_str.strip())
                color_values = [int(x.strip()) for x in color_str.split(',') if x.strip().isdigit()]
                if len(color_values) == 3:
                    color_dict[class_id] = tuple(color_values)

        if color_dict:
            return color_dict

        return None

    except Exception as e:
        print(f"解析颜色参数时出错: {e}")
        return None


@app.route('/detect', methods=['POST'])
def detect_objects():
    """
    目标检测API接口
    支持单张或多张图片上传
    支持指定检测类别
    支持自定义颜色
    支持中英文标签
    """
    if 'images' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400

    # 获取要检测的类别参数（可选）
    classes_param = request.form.get('classes', '')
    classes_to_detect = []
    if classes_param:
        try:
            classes_to_detect = [int(cls.strip()) for cls in classes_param.split(',') if cls.strip().isdigit()]
        except ValueError:
            return jsonify({'error': '无效的类别参数格式。请使用逗号分隔的数字，例如：0,2'}), 400

    # 获取语言参数（可选）- 'zh' 中文, 'en' 英文
    lang_param = request.form.get('lang', 'zh')
    use_chinese = lang_param.lower() == 'zh'

    # 获取颜色参数（可选）
    color_param = request.form.get('color', '')
    custom_colors = parse_color_param(color_param)

    # 获取线宽参数（可选）
    line_width_param = request.form.get('line_width', '2')
    try:
        line_width = max(1, min(10, int(line_width_param)))
    except:
        line_width = 2

    # 获取字体大小参数（可选）
    font_size_param = request.form.get('font_size', '1.0')
    try:
        font_size = max(0.5, min(3.0, float(font_size_param)))
    except:
        font_size = 1.0

    files = request.files.getlist('images')
    results = []

    for file in files:
        if file.filename == '':
            continue

        # 读取图片
        img_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 使用YOLO进行目标检测，指定要检测的类别
        if classes_to_detect:
            results_model = model(img, classes=classes_to_detect)
        else:
            results_model = model(img)

        # 提取检测结果
        detections = {
            'boxes': results_model[0].boxes.xyxy.cpu().numpy(),
            'classes': results_model[0].boxes.cls.cpu().numpy(),
            'confidences': results_model[0].boxes.conf.cpu().numpy()
        }

        # 自定义绘制标注（支持中文和自定义颜色）
        annotated_img = draw_custom_annotations(
            img,
            detections,
            use_chinese=use_chinese,
            custom_colors=custom_colors,
            line_width=line_width,
            font_size=font_size
        )

        # 将标注后的图片转换为字节流
        _, img_encoded = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_bytes_annotated = img_encoded.tobytes()

        # 保存检测结果信息
        detection_info = {
            'filename': file.filename,
            'annotated_image': img_bytes_annotated,
            'detected_objects': []
        }

        # 获取检测到的对象信息
        boxes = results_model[0].boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            detection_info['detected_objects'].append({
                'class_id': class_id,
                'class_name': COCO_CLASSES_CN[class_id] if use_chinese else COCO_CLASSES_EN[class_id],
                'confidence': round(confidence, 2),
                'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            })

        results.append(detection_info)

    # 如果只有一张图片，直接返回图片
    if len(results) == 1:
        return send_file(
            io.BytesIO(results[0]['annotated_image']),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f"annotated_{secure_filename(results[0]['filename'])}"
        )
    else:
        # 多张图片打包成zip返回
        memory_file = io.BytesIO()
        with ZipFile(memory_file, 'w') as zf:
            for i, result in enumerate(results):
                zf.writestr(f"annotated_{secure_filename(result['filename'])}", result['annotated_image'])

        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='annotated_images.zip'
        )

@app.route('/detect_json', methods=['POST'])
def detect_objects_json():
    """
    返回JSON格式的检测结果，包含图片和检测信息
    """
    if 'images' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400

    # 获取参数
    classes_param = request.form.get('classes', '')
    lang_param = request.form.get('lang', 'zh')
    color_param = request.form.get('color', '')
    line_width_param = request.form.get('line_width', '2')
    font_size_param = request.form.get('font_size', '1.0')

    classes_to_detect = []
    if classes_param:
        try:
            classes_to_detect = [int(cls.strip()) for cls in classes_param.split(',') if cls.strip().isdigit()]
        except ValueError:
            return jsonify({'error': '无效的类别参数格式。请使用逗号分隔的数字，例如：0,2'}), 400

    use_chinese = lang_param.lower() == 'zh'
    custom_colors = parse_color_param(color_param)

    try:
        line_width = max(1, min(10, int(line_width_param)))
    except:
        line_width = 2

    try:
        font_size = max(0.5, min(3.0, float(font_size_param)))
    except:
        font_size = 1.0

    file = request.files['images']
    if file.filename == '':
        return jsonify({'error': '无效的文件名'}), 400

    # 读取图片
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 进行目标检测
    if classes_to_detect:
        results_model = model(img, classes=classes_to_detect)
    else:
        results_model = model(img)

    # 提取检测结果
    detections = {
        'boxes': results_model[0].boxes.xyxy.cpu().numpy(),
        'classes': results_model[0].boxes.cls.cpu().numpy(),
        'confidences': results_model[0].boxes.conf.cpu().numpy()
    }

    # 自定义绘制标注
    annotated_img = draw_custom_annotations(
        img,
        detections,
        use_chinese=use_chinese,
        custom_colors=custom_colors,
        line_width=line_width,
        font_size=font_size
    )

    # 转换为字节流
    _, img_encoded = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_bytes_annotated = img_encoded.tobytes()

    # 获取检测结果
    detection_results = []
    boxes = results_model[0].boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detection_results.append({
            'class_id': class_id,
            'class_name': COCO_CLASSES_CN[class_id] if use_chinese else COCO_CLASSES_EN[class_id],
            'confidence': round(confidence, 2),
            'bbox': {
                'x1': round(x1, 1),
                'y1': round(y1, 1),
                'x2': round(x2, 1),
                'y2': round(y2, 1),
                'width': round(x2 - x1, 1),
                'height': round(y2 - y1, 1)
            }
        })

    # 将图片转换为base64编码
    img_base64 = base64.b64encode(img_bytes_annotated).decode('utf-8')

    return jsonify({
        'filename': file.filename,
        'annotated_image_base64': img_base64,
        'detected_objects': detection_results,
        'total_objects': len(detection_results),
        'filtered_classes': classes_to_detect if classes_to_detect else 'all',
        'language': 'chinese' if use_chinese else 'english'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model': 'yolo11x',
        'supported_classes': len(COCO_CLASSES_CN),
        'available_languages': ['chinese', 'english']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
