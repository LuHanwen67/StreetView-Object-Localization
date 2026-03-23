import os
import cv2
import math
import torch
import numpy as np
from glob import glob
from tqdm import tqdm  # 用于显示进度条

# 导入 DA3METRIC-Large 模型
from depth_anything_3.api import DepthAnything3
from ultralytics import YOLO 

def get_perspective_grid(equ_h, equ_w, cx, cy, fov_deg=90, out_w=512, out_h=512):
    """基于目标中心点和视场角，生成消除全景畸变的映射网格 (严密 3D 射线逆向追踪版)"""
    lon0 = (cx / equ_w - 0.5) * 2 * np.pi
    lat0 = (0.5 - cy / equ_h) * np.pi
    f = (0.5 * out_w) / np.tan(np.radians(fov_deg) / 2)

    x = np.linspace(-out_w/2, out_w/2, out_w)
    y = np.linspace(-out_h/2, out_h/2, out_h)
    xx, yy = np.meshgrid(x, y)

    x_cam = xx
    y_cam = -yy  # OpenCV y轴向下，转为数学+y向上
    z_cam = np.full_like(xx, f)

    # 绕 X 轴旋转 (Pitch)
    x_rot = x_cam
    y_rot = y_cam * np.cos(lat0) + z_cam * np.sin(lat0)
    z_rot = -y_cam * np.sin(lat0) + z_cam * np.cos(lat0)

    # 绕 Y 轴旋转 (Yaw)
    x_world = x_rot * np.cos(lon0) + z_rot * np.sin(lon0)
    y_world = y_rot
    z_world = -x_rot * np.sin(lon0) + z_rot * np.cos(lon0)

    # 反算回球面
    rho = np.sqrt(x_world**2 + y_world**2 + z_world**2)
    lon = np.arctan2(x_world, z_world)
    lat = np.arcsin(np.clip(y_world / rho, -1, 1))

    # 映射回全景图
    map_x = (lon / (2 * np.pi) + 0.5) * equ_w
    map_y = (0.5 - lat / np.pi) * equ_h

    return map_x.astype(np.float32), map_y.astype(np.float32)


class StreetViewBatchProcessor:
    def __init__(self, yolo_weights_path):
        self.earth_radius = 6378137.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在使用的计算设备: {self.device}")
        
        print("正在加载 YOLOv26 模型...")
        self.yolo_model = YOLO(yolo_weights_path) 
        
        print("正在加载 DA3METRIC-Large 深度模型...")
        self.depth_model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
        self.depth_model = self.depth_model.to(self.device)
        print("所有模型加载完毕！\n" + "="*40)

    def parse_filename(self, filename):
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        try:
            coords = parts[0].split(',')
            return float(coords[0]), float(coords[1]) # 返回 lat, lon
        except:
            return 0.0, 0.0

    def calculate_new_coordinates(self, lat, lon, distance, bearing_degrees):
        lat_rad, lon_rad, bearing_rad = map(math.radians, [lat, lon, bearing_degrees])
        lat_new_rad = math.asin(math.sin(lat_rad) * math.cos(distance / self.earth_radius) +
                                math.cos(lat_rad) * math.sin(distance / self.earth_radius) * math.cos(bearing_rad))
        lon_new_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance / self.earth_radius) * math.cos(lat_rad),
                                           math.cos(distance / self.earth_radius) - math.sin(lat_rad) * math.sin(lat_new_rad))
        return math.degrees(lat_new_rad), math.degrees(lon_new_rad)

    def process_single_image(self, input_image_path, output_base_dir):
        filename = os.path.basename(input_image_path)
        img_name_without_ext = os.path.splitext(filename)[0]
        
        # 无论是否有长椅，先创建同名文件夹
        output_dir = os.path.join(output_base_dir, img_name_without_ext)
        os.makedirs(output_dir, exist_ok=True)
        
        img_bgr = cv2.imdecode(np.fromfile(input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return
            
        h, w = img_bgr.shape[:2]
        cam_lat, cam_lon = self.parse_filename(filename)
        
        # YOLO 快速推断 (关闭 verbose 避免刷屏)
        results = self.yolo_model.predict(img_bgr, verbose=False) 
        result = results[0]

        # 提早拦截：检查是否包含长椅
        has_bench = False
        if result.masks is not None:
            classes = result.boxes.cls.cpu().numpy()
            names = result.names
            for cls_id in classes:
                if 'bench' in names[int(cls_id)].lower():
                    has_bench = True
                    break
                    
        # 如果没有长椅，直接退出当前图片的循环，文件夹保持为空
        if not has_bench:
            return

        # ================== 以下是有长椅时的重度处理逻辑 ==================
        
        # 保存全景 YOLO 结果（可选）
        result.save(filename=os.path.join(output_dir, "yolo_pano_result.jpg"))

        bench_info_list = []
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names
        
        for i, mask in enumerate(masks):
            if 'bench' not in names[int(classes[i])].lower():
                continue
                
            x1, y1, x2, y2 = boxes[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bearing = (cx / w) * 360.0
            mask_pano = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            CROP_W, CROP_H = 512, 512
            map_x, map_y = get_perspective_grid(h, w, cx, cy, fov_deg=90, out_w=CROP_W, out_h=CROP_H)
            
            crop_bgr = cv2.remap(img_bgr, map_x, map_y, cv2.INTER_LINEAR)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_mask = cv2.remap(mask_pano, map_x, map_y, cv2.INTER_NEAREST)
            
            # DA3 深度计算
            pred = self.depth_model.inference([crop_rgb])
            crop_depth = pred.depth[0]
            
            if crop_depth.shape[0] != CROP_H or crop_depth.shape[1] != CROP_W:
                crop_depth = cv2.resize(crop_depth, (CROP_W, CROP_H), interpolation=cv2.INTER_LINEAR)
            
            # 保存中间结果图
            cv2.imencode('.jpg', crop_bgr)[1].tofile(os.path.join(output_dir, f"undistorted_bench_{i}.jpg"))
            
            depth_viz = (crop_depth - np.min(crop_depth)) / (np.max(crop_depth) - np.min(crop_depth) + 1e-5) * 255
            depth_viz = depth_viz.astype(np.uint8)
            depth_viz_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
            cv2.imencode('.jpg', depth_viz_colored)[1].tofile(os.path.join(output_dir, f"undistorted_depth_viz_{i}.jpg"))

            bench_depth_colored = np.zeros_like(depth_viz_colored)
            masked_depth_viz = depth_viz[crop_mask > 0.5]
            if masked_depth_viz.size > 0:
                masked_depth_colored = cv2.applyColorMap(masked_depth_viz, cv2.COLORMAP_JET)
                bench_depth_colored[crop_mask > 0.5] = masked_depth_colored.squeeze()
            cv2.imencode('.jpg', bench_depth_colored)[1].tofile(os.path.join(output_dir, f"crop_bench_depth_viz_{i}.jpg"))
            cv2.imencode('.png', (crop_mask * 255).astype(np.uint8))[1].tofile(os.path.join(output_dir, f"crop_bench_mask_{i}.png"))

            bench_pixels_depth = crop_depth[crop_mask > 0.5]
            if len(bench_pixels_depth) == 0: continue
            
            avg_depth = np.median(bench_pixels_depth)
            bench_lat, bench_lon = self.calculate_new_coordinates(cam_lat, cam_lon, avg_depth, bearing)
            
            bench_info_list.append({
                "bench_id": i, "depth": avg_depth, "bearing": bearing, 
                "lat": bench_lat, "lon": bench_lon
            })
    
        # 写入 txt 结果
        with open(os.path.join(output_dir, "benches_coordinates.txt"), 'w', encoding='utf-8') as f:
            f.write(f"相机原坐标: Lat={cam_lat}, Lon={cam_lon}\n")
            f.write("-" * 50 + "\n")
            for info in bench_info_list:
                f.write(f"长椅 ID: {info['bench_id']}\n")
                f.write(f"  绝对距离 (米): {info['depth']:.2f}\n")
                f.write(f"  推算航向角 (度): {info['bearing']:.2f}\n")
                f.write(f"  推算坐标: Lat={info['lat']:.6f}, Lon={info['lon']:.6f}\n")
                f.write("-" * 50 + "\n")

    def process_directory(self, input_dir, output_dir):
        """批量处理文件夹内的所有图片"""
        valid_exts = {'.jpg', '.png', '.jpeg', '.bmp'}
        # 获取所有图片路径
        image_paths = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        
        if not image_paths:
            print(f"在 {input_dir} 中没有找到支持的图片格式！")
            return
            
        print(f"发现 {len(image_paths)} 张街景图片，开始批量处理...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用 tqdm 包装循环，显示进度条
        for img_path in tqdm(image_paths, desc="处理进度", unit="张"):
            self.process_single_image(img_path, output_dir)
            
        print("\n处理全部完成！")


if __name__ == '__main__':
    # ================== 批量处理路径配置区 ==================
    INPUT_DIR = r"E:\DaChuang\input"   # 存放所有原始街景图的文件夹
    OUTPUT_DIR = r"E:\DaChuang\output" # 输出的根目录
    YOLO_WEIGHTS = r"D:\VSCodeProjects\yolo26l-seg.pt"
    # ========================================================

    processor = StreetViewBatchProcessor(YOLO_WEIGHTS)
    
    # 调用批量处理方法
    processor.process_directory(INPUT_DIR, OUTPUT_DIR)