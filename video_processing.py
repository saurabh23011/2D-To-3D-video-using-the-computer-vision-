# import cv2
# import torch
# import numpy as np
# import os
# from torchvision.transforms import Compose

# def process_video(input_path, output_path):
#     '''
#     Process the input video to generate a depth-mapped 3D video.

#     Parameters:
#     - input_path: Path to the input video file.
#     - output_path: Path to save the processed video.
#     '''
#     # Load MiDaS model for depth estimation
#     model_type = "DPT_Large"
#     midas = torch.hub.load("intel-isl/MiDaS", model_type)

#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     midas.to(device)
#     midas.eval()

#     # Define transforms
#     if model_type in ("DPT_Large", "DPT_Hybrid"):
#         transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
#     else:
#         transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

#     # Open video file
#     cap = cv2.VideoCapture(input_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Define video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     processed_frames = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to RGB and perform transforms
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_batch = transform(img).to(device)

#         # Prediction
#         with torch.no_grad():
#             prediction = midas(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=img.shape[:2],
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()

#         depth_map = prediction.cpu().numpy()

#         # Normalize and convert to 8-bit image
#         depth_min = depth_map.min()
#         depth_max = depth_map.max()
#         depth_map = (depth_map - depth_min) / (depth_max - depth_min)
#         depth_map = (255 * depth_map).astype(np.uint8)

#         # Merge depth map with original frame (for visualization)
#         depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
#         blended_frame = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)

#         # Write frame to output video
#         out.write(blended_frame)

#         # Optional: Update progress
#         processed_frames += 1
#         print(f"Processing frame {processed_frames}/{frame_count}", end='\r')

#     # Release resources
#     cap.release()
#     out.release()
#     print("\nProcessing complete.")



import cv2
import torch
import numpy as np
import os
from torchvision import transforms
import PIL.Image as Image
import open3d as o3d

def process_video(input_path, output_video_path, depth_output_folder, pointcloud_output_folder):
    '''
    Process the input video to generate depth maps and point clouds.

    Parameters:
    - input_path: Path to the input video file.
    - output_video_path: Path to save the processed video.
    - depth_output_folder: Folder to save depth maps.
    - pointcloud_output_folder: Folder to save point clouds.
    '''
    # Load MiDaS model for depth estimation
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()

    # Define transforms
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    else:
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    # Open video file
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer for blended video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Create output folders if they don't exist
    os.makedirs(depth_output_folder, exist_ok=True)
    os.makedirs(pointcloud_output_folder, exist_ok=True)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and perform transforms
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        # Prediction
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize depth map
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)

        # Save depth map as image
        depth_map_image = (255 * depth_map_normalized).astype(np.uint8)
        depth_filename = os.path.join(depth_output_folder, f'depth_{processed_frames:05d}.png')
        cv2.imwrite(depth_filename, depth_map_image)

        # Merge depth map with original frame (for visualization)
        depth_colored = cv2.applyColorMap(depth_map_image, cv2.COLORMAP_JET)
        blended_frame = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)

        # Write frame to output video
        out.write(blended_frame)

        # Save point cloud
        pointcloud_filename = os.path.join(pointcloud_output_folder, f'pointcloud_{processed_frames:05d}.ply')
        save_point_cloud(depth_map_normalized, img, pointcloud_filename)

        # Update progress
        processed_frames += 1
        if processed_frames % 10 == 0:
            print(f"Processing frame {processed_frames}/{frame_count}")

    # Release resources
    cap.release()
    out.release()
    print("\nProcessing complete.")

def save_point_cloud(depth_map, color_image, output_filename):
    '''
    Save the point cloud from the depth map and color image.

    Parameters:
    - depth_map: 2D NumPy array of depth values.
    - color_image: Original color image corresponding to the depth map.
    - output_filename: Filename to save the point cloud (PLY format).
    '''
    height, width = depth_map.shape

    # Camera intrinsics (assuming focal length = 1)
    fx = fy = 1
    cx = width / 2
    cy = height / 2

    # Generate grid of pixel coordinates
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    x_coords = x_coords.reshape(-1)
    y_coords = y_coords.reshape(-1)
    depth_flat = depth_map.reshape(-1)

    # Convert pixel coordinates to 3D coordinates
    x = (x_coords - cx) * depth_flat / fx
    y = (y_coords - cy) * depth_flat / fy
    z = depth_flat

    points = np.vstack((x, y, z)).transpose()

    # Get color values
    colors = color_image.reshape(-1, 3) / 255.0  # Normalize to [0,1]

    # Remove points with invalid depth
    valid_indices = np.where(depth_flat > 0)
    points = points[valid_indices]
    colors = colors[valid_indices]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save the point cloud to a file
    o3d.io.write_point_cloud(output_filename, pcd) 