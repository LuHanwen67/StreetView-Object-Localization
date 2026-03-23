# StreetView-Object-Localization
A pipeline for bench detection and geolocating in panoramic street view images using YOLO and Depth Anything V3.

This project provides an automated pipeline to detect specific objects (e.g., benches) in 360° panoramic street view images and estimate their real-world geographic coordinates.

## 🚀 Key Features
- **Instance Segmentation**: Uses YOLO26-seg to identify objects in equirectangular panoramas.
- **Distortion Correction**: Implements a rigorous 3D ray-tracing algorithm to generate undistorted perspective crops from panoramas.
- **Monocular Depth Estimation**: Integrates **Depth Anything V3 (DA3)** to predict high-resolution metric depth maps.
- **Geolocation Calculation**: Automatically derives absolute GPS coordinates (Lat/Lon) for detected objects based on camera metadata and estimated depth.

## 🛠 Tech Stack
- **Languages**: Python
- **Libraries**: PyTorch, OpenCV, Ultralytics, NumPy, tqdm
- **Models**: YOLO26, Depth Anything V3

## 📖 How it Works
The script parses camera coordinates from filenames, detects target objects, transforms the view to eliminate spherical distortion, and finally calculates the median depth to triangulate the object's location.

## 📊 Sample Results
You can find the full processing pipeline results in the `/case` folder.
