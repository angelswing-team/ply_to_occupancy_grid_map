import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import json
from tqdm import tqdm
import sys
from multiprocessing import Pool, cpu_count
import gc

def chunk_reader(ply_path, chunk_size=1000000):
    """Read PLY file in chunks to handle large files"""
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    total_points = len(points)

    for i in range(0, total_points, chunk_size):
        chunk = points[i:i + chunk_size]
        yield chunk
        del chunk
        gc.collect()

def process_chunk(chunk, nb_neighbors=20, std_ratio=2.0, voxel_size=0.15):
    """Process a single chunk of points"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(chunk)

    # Downsample
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Filter outliers
    if len(pcd.points) > nb_neighbors:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return np.asarray(cl.points)
    return np.asarray(pcd.points)

class OccupancyGridMap:
    def __init__(self, resolution, width, height, x_offset=0, z_offset=0):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.grid = np.zeros((self.grid_height, self.grid_width))
        self.trajectory_points = []
        self.x_offset = x_offset
        self.z_offset = z_offset

    def world_to_grid(self, points):
        # Improved coordinate transformation
        grid_x = ((points[:, 0] - self.x_offset + self.width/2) / self.resolution).astype(int)
        grid_y = ((points[:, 1] - self.z_offset + self.height/2) / self.resolution).astype(int)
        return grid_x, grid_y

    def update_from_point_cloud(self, points_2d, sensor_pos=(0, 0)):
        # Vectorized point cloud update
        grid_x, grid_y = self.world_to_grid(points_2d)
        valid_points = (grid_x >= 0) & (grid_x < self.grid_width) & \
                      (grid_y >= 0) & (grid_y < self.grid_height)

        grid_x = grid_x[valid_points]
        grid_y = grid_y[valid_points]
        self.grid[grid_y, grid_x] = 1.0

    def update_from_point_cloud_chunk(self, points_2d):
        """Update grid map with a chunk of points"""
        grid_x, grid_y = self.world_to_grid(points_2d)
        valid_points = (grid_x >= 0) & (grid_x < self.grid_width) & \
                      (grid_y >= 0) & (grid_y < self.grid_height)

        grid_x = grid_x[valid_points]
        grid_y = grid_y[valid_points]
        self.grid[grid_y, grid_x] = 1.0

        del grid_x, grid_y
        gc.collect()

    def add_trajectory_point(self, x, y):
        grid_x, grid_y = self.world_to_grid(np.array([[x, y]]))
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            self.trajectory_points.append((grid_x[0], grid_y[0]))

    def visualize(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, origin='lower', cmap='binary',
                  extent=[-self.width/2, self.width/2, -self.height/2, self.height/2])

        if self.trajectory_points:
            traj_x = [(x - self.grid_width/2) * self.resolution for x, _ in self.trajectory_points]
            traj_y = [(y - self.grid_height/2) * self.resolution for _, y in self.trajectory_points]
            plt.plot(traj_x, traj_y, 'r.-', linewidth=2, markersize=6)

        plt.title('Occupancy Grid Map')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.show()

    def export_for_threejs(self, output_path):
        map_data = {
            'resolution': self.resolution,
            'width': self.width,
            'height': self.height
        }

        # Increase DPI for higher resolution output
        plt.figure(figsize=(20, 20), dpi=300)
        plt.imshow(self.grid, cmap='binary', interpolation='nearest')
        plt.axis('off')
        plt.savefig(f'{output_path}_grid.png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        with open(f'{output_path}_data.json', 'w') as f:
            json.dump(map_data, f)

def read_trajectory(file_path):
    trajectory = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            x = float(parts[1])
            z = float(parts[3])  # Use z instead of y
            trajectory.append((x, z))
    return trajectory

def filter_point_cloud(pcd, nb_neighbors=15, std_ratio=2.0, voxel_size=0.15):
    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)

    # Downsample using voxel grid filter
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Convert to numpy for height filtering
    points = np.asarray(pcd.points)

    # Get height range
    # z_min = np.min(points[:, 1])
    # z_max = np.max(points[:, 1])
    # z_mid = (z_max + z_min)

    # # Keep points in lower half of height range
    # mask = points[:, 1] <= z_mid

    # Get height range (y coordinate is height)
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    height_range = y_max - y_min

    # Calculate height thresholds (25% to 75% range)
    y_lower = y_min + (height_range * 0.05)
    y_upper = y_min + (height_range * 0.95)

    # Keep points in middle height range
    mask = (points[:, 1] >= y_lower) & (points[:, 1] <= y_upper)
    filtered_points = points[mask]

    # Create new point cloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(pcd.colors)[mask])
    return filtered_pcd

def process_ply_file(ply_path, resolution=0.015, width=20, height=20, sensor_pos=(0, 0)):
    pcd = o3d.io.read_point_cloud(ply_path)
    filtered_pcd = filter_point_cloud(pcd)

    points = np.asarray(filtered_pcd.points)
    points_2d = points[:, [0, 2]]

    grid_map = OccupancyGridMap(resolution=resolution, width=width, height=height)
    grid_map.update_from_point_cloud(points_2d, sensor_pos)
    return grid_map

def main():
    ply_file_path = "/home/aasman/angelswing/output2/angelswing_data.ply"
    trajectory_file_path = "/home/aasman/angelswing/output2/trajectory/keyframe_trajectory.txt"
    # ply_file_path = "/home/aasman/angelswing/output1/circular_data.ply"
    # trajectory_file_path = "/home/aasman/angelswing/output1/circular_trajectory/keyframe_trajectory.txt"
    print("Calculating point cloud bounds...")
    # First pass: calculate bounds
    x_min, x_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    for chunk in chunk_reader(ply_file_path):
        x_min = min(x_min, np.min(chunk[:, 0]))
        x_max = max(x_max, np.max(chunk[:, 0]))
        z_min = min(z_min, np.min(chunk[:, 2]))
        z_max = max(z_max, np.max(chunk[:, 2]))

    # Calculate map dimensions with 3x size and better centering
    width = (x_max - x_min)*2   # Triple the width
    height = (z_max - z_min)*2 # Triple the height
    padding = max(width, height)  # Increased padding to 15%

    width += 2 * padding
    height += 2 * padding

    # Calculate center offsets
    x_center = (x_max + x_min) / 2
    z_center = (z_max + z_min) / 2

    # Adjust resolution for larger map
    min_grid_dimension = 3000  # Increased for larger map
    resolution = min(
        width / min_grid_dimension,
        height / min_grid_dimension,
        0.05  # Even finer resolution
    )

    print(f"Center point: X({x_center:.2f}), Z({z_center:.2f})")
    print(f"Original point cloud bounds: X({x_min:.2f} to {x_max:.2f}), Z({z_min:.2f} to {z_max:.2f})")
    print(f"Expanded map dimensions: {width:.2f}m x {height:.2f}m")
    print(f"Resolution: {resolution:.4f}m per cell")
    print(f"Grid size: {int(width/resolution)} x {int(height/resolution)} cells")

    # Create grid map with expanded dimensions and offsets
    grid_map = OccupancyGridMap(resolution=resolution, width=width, height=height,
                               x_offset=0, z_offset=0)

    # Process chunks in parallel with adjusted parameters
    print("Processing point cloud in chunks...")
    with Pool(cpu_count()) as pool:
        for chunk in tqdm(chunk_reader(ply_file_path)):
            # Process chunk with finer voxel size
            filtered_points = process_chunk(chunk,
                                         voxel_size=resolution,
                                         nb_neighbors=40,
                                         std_ratio=2.5)
            points_2d = filtered_points[:, [0, 2]]

            # Update grid map without centering (handled by world_to_grid)
            grid_map.update_from_point_cloud_chunk(points_2d)

            # Clean up
            del filtered_points, points_2d
            gc.collect()

    # Process trajectory with proper offsets
    print("Processing trajectory...")
    trajectory = np.array(read_trajectory(trajectory_file_path))
    for x, z in trajectory:
        grid_map.add_trajectory_point(x, z)  # Original coordinates will be transformed in world_to_grid

    # Add debug visualization before export
    print("Generating debug visualization...")
    grid_map.visualize()

    print("Exporting map...")
    grid_map.export_for_threejs('/home/aasman/angelswing/360/output/mapoutput/video2/minimap')
    print("Done!")

if __name__ == "__main__":
    main()