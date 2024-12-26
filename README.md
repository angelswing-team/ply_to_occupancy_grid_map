# PLY to Occupancy Grid Map Converter

This script converts PLY (Polygon File Format) point cloud files into 2D occupancy grid maps.

## Features

- Converts PLY point cloud files to 2D occupancy grid maps
- Supports different resolutions for the output grid
- Handles both binary and ASCII PLY files
- Projects 3D points onto a 2D plane
- Generates occupancy probabilities based on point density

## Requirements

- Python 3.9
- NumPy
- Open3D
- matplotlib

## Installation

```bash
pip install numpy open3d matplotlib tqdm
```

## Usage
Update the path in the scripts before running the code
```bash
python main.py
```


## Example

```bash
python main.py
```

## How It Works

1. The program reads the input PLY file using Open3D
2. Points are projected onto the XY plane
3. The space is discretized into a grid based on the specified resolution
4. Point density in each cell is calculated
5. An occupancy probability is assigned to each cell
6. The resulting grid map is saved as an image file and json




