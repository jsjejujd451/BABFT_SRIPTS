import glob
import json
import logging
import os
import random
import threading
import webbrowser
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import time
import numpy as np
from PIL import Image
import numba as nb
from rich.box import DOUBLE, ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.theme import Theme

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),     
        logging.StreamHandler()             
    ]
)
logger = logging.getLogger(__name__) 
# Global variables
textures = []
texture_sizes = {}
texture_color_cache = OrderedDict()
MAX_CACHE_SIZE = 10000
command_history = []
current_profile = "default"
FIRST_RUN_FLAG = "first_run.flag"

# Default parameters 
DEFAULT_PARAMS = {
    "block_type": "PlasticBlock",
    "position_offset": -0.025,
    "face_inward_offset": -0.025,
    "edge_inward_offset": -0.025,
    "min_thickness": 1e-28,
    "min_size": 1e-28,
    "edge_min_size": 0.05,
    "edge_min_thickness": 0.05,
    "fallback_color": [233, 218, 218],
    "color_palette": [[255, 99, 71], [135, 206, 235], [240, 230, 140], [152, 251, 152]],
    "use_random_color": False,
    "use_textures": True,
    "tolerance": 1e-6,
    "min_block_size": 0.001,
    "fill_thickness": 0.05,
    "max_boundary_exceed": 0.1,
    "min_crop_size": 0.1,
    "texture_scale": 1.0,
    "texture_downsample": 16,
    "max_threads": max(1, os.cpu_count() // 2),
    "buffer_size": 10000,
    "auto_uv": True,
    "uv_projection": "planar",
    "edpn_scale_factor": 0.9,
    "edpn_min_area": 0.01,
    "edpn_square_size": 0.05,
    "blocks_per_file": 0  
}

# Default terminal customization
DEFAULT_TERMINAL = {
    "text_color": "#FFFFFF",
    "background_color": "#000000",
    "progress_bar_style": "bar.complete",
    "header_color": "bold #00FFFF",
    "border_style": "#00FF00",
    "prompt_color": "#FFFF00",
    "panel_box": "ROUNDED"
}

default_theme = Theme({
    "cyan": "#00FFFF",
    "green": "#00FF00",
    "red": "#FF0000",
    "yellow": "#FFFF00",
    "bold cyan": "bold #00FFFF",
    "bold green": "bold #00FF00",
    "bold red": "bold #FF0000",
    "magenta": "#FF00FF"
})

# Profile management
PROFILES_FILE = "profiles.json"
profiles = {"default": {"params": DEFAULT_PARAMS.copy(), "mode": "faces", "terminal": DEFAULT_TERMINAL.copy()}}
if os.path.exists(PROFILES_FILE):
    with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
        profiles.update(json.load(f))


class OutputWriter:
    def __init__(self, output_path, blocks_per_file):
        self.output_path = output_path
        self.blocks_per_file = blocks_per_file if blocks_per_file > 0 else float('inf')
        self.file_index = 1
        self.blocks_in_current_file = 0
        self.first_block = True
        self.current_out_f = open(self.get_current_filename(), 'w', encoding='utf-8')
        self.current_out_f.write('[')

    def get_current_filename(self):
        if self.blocks_per_file == float('inf'):
            return self.output_path
        else:
            base, ext = os.path.splitext(self.output_path)
            return f"{base}_part{self.file_index}{ext}"

    def write_block(self, block):
        if self.blocks_in_current_file >= self.blocks_per_file:
            self.current_out_f.write(']')
            self.current_out_f.close()
            self.file_index += 1
            self.current_out_f = open(self.get_current_filename(), 'w', encoding='utf-8')
            self.current_out_f.write('[')
            self.blocks_in_current_file = 0
            self.first_block = True
        if not self.first_block:
            self.current_out_f.write(',')
        self.current_out_f.write(json.dumps(block, separators=(',', ':')))
        self.blocks_in_current_file += 1
        self.first_block = False

    def close(self):
        self.current_out_f.write(']')
        self.current_out_f.close()

# Terminal and profile functions
def get_terminal_settings():
    return profiles[current_profile].get("terminal", DEFAULT_TERMINAL.copy())

def get_params():
    return profiles[current_profile]["params"]

def save_profiles():
    with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=4)
    console.print(f"[green]Profiles saved to {PROFILES_FILE}[/green]")

def update_console():
    global console
    terminal_settings = get_terminal_settings()
    bg_color = terminal_settings["background_color"]
    text_color = terminal_settings["text_color"]
    if bg_color.startswith("#"):
        bg_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        bg_style = f"on rgb({bg_rgb[0]},{bg_rgb[1]},{bg_rgb[2]})"
    else:
        bg_style = f"on {bg_color}"
    console = Console(
        theme=default_theme,
        style=f"{text_color} {bg_style}",
        force_terminal=True,
        color_system="truecolor"
    )
    return console

console = update_console()

# Helper Functions
progress_lock = threading.Lock()
progress_current = 0

def print_progress_bar(iteration, total, prefix='', suffix='', length=40):
    global progress_current
    style = get_terminal_settings()["progress_bar_style"]
    with progress_lock:
        progress_current = min(iteration, total)
        percent = f"{100 * (progress_current / total):.1f}"
        filled = int(length * progress_current // total)
        bar = '#' * filled + '-' * (length - filled)
        console.print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r' if progress_current < total else '\n', style=style)
        if progress_current >= total:
            console.print()

def get_block_color(params):
    if params.get("use_random_color", False):
        return random.choice(params.get("color_palette", [params["fallback_color"]]))
    return params["fallback_color"]

# EDPN Helper Functions
def polygon_centroid(polygon):
    n = len(polygon)
    area = 0
    cx = 0
    cy = 0
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        temp = p1[0] * p2[1] - p2[0] * p1[1]
        area += temp
        cx += (p1[0] + p2[0]) * temp
        cy += (p1[1] + p2[1]) * temp
    area /= 2
    if abs(area) < 1e-8:
        return (0, 0)
    cx /= (6 * area)
    cy /= (6 * area)
    return (cx, cy)

def polygon_area(polygon):
    n = len(polygon)
    a = 0
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        a += p1[0] * p2[1] - p2[0] * p1[1]
    return abs(a) / 2

def scale_polygon(polygon, center, scale):
    return [(center[0] + scale * (p[0] - center[0]), center[1] + scale * (p[1] - center[1])) for p in polygon]

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def edpn_fill_polygon(polygon_2d, params, center_3d, normal, local_y, texture_idx=None, material=None):
    blocks = []
    scale = params["edpn_scale_factor"]
    min_area = params["edpn_min_area"]
    square_size = params["edpn_square_size"]
    thickness = params["edge_min_thickness"]
    texture_scale = params.get("texture_scale", 1.0)
    use_textures = params.get("use_textures", True)

    P = polygon_2d
    k = 0
    while polygon_area(P) > min_area and k < 100:
        C = polygon_centroid(P)
        P_next = scale_polygon(P, C, scale)
        for i in range(len(P)):
            v0 = P[i]
            v1 = P[(i + 1) % len(P)]
            v0_prime = P_next[i]
            v1_prime = P_next[(i + 1) % len(P)]
            quad = [v0, v1, v1_prime, v0_prime]
            min_x = min(p[0] for p in quad)
            min_y = min(p[1] for p in quad)
            max_x = max(p[0] for p in quad)
            max_y = max(p[1] for p in quad)
            size_x = (max_x - min_x) * texture_scale
            size_y = (max_y - min_y) * texture_scale
            size_x = max(size_x, params["edge_min_size"])
            size_y = max(size_y, params["edge_min_size"])
            quad_center_2d = ((min_x + max_x) / 2, (min_y + max_y) / 2)
            offset_x = (quad_center_2d[0] - C[0]) * texture_scale
            offset_y = (quad_center_2d[1] - C[1]) * texture_scale
            block_center = [
                center_3d[0] + offset_x * local_y[0] + offset_y * normal[0],
                center_3d[1] + offset_x * local_y[1] + offset_y * normal[1],
                center_3d[2] + offset_x * local_y[2] + offset_y * normal[2]
            ]
            color = get_block_color(params)
            if use_textures and textures and texture_idx is not None:
                texture_path = textures[texture_idx]
                if os.path.exists(texture_path):
                    tex_size = texture_sizes.get(texture_path, (1024, 1024))
                    region = (0, 0, tex_size[0], tex_size[1])
                    color = extract_texture_color(texture_path, region, downsample=params["texture_downsample"])
                    logger.info(f"EDPN applied texture color {color} for material {material}")
            block = [
                params["block_type"],
                block_center,
                normal,
                local_y,
                [size_x, size_y, thickness],
                True,
                color
            ]
            blocks.append(block)
        P = P_next
        k += 1

    min_x = min(p[0] for p in P)
    min_y = min(p[1] for p in P)
    max_x = max(p[0] for p in P)
    max_y = max(p[1] for p in P)
    i_min = int(min_x / square_size)
    i_max = int(max_x / square_size) + 1
    j_min = int(min_y / square_size)
    j_max = int(max_y / square_size) + 1
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            center_x = (i + 0.5) * square_size
            center_y = (j + 0.5) * square_size
            if point_in_polygon((center_x, center_y), P):
                offset_x = (center_x - C[0]) * texture_scale
                offset_y = (center_y - C[1]) * texture_scale
                square_center = [
                    center_3d[0] + offset_x * local_y[0] + offset_y * normal[0],
                    center_3d[1] + offset_x * local_y[1] + offset_y * normal[1],
                    center_3d[2] + offset_x * local_y[2] + offset_y * normal[2]
                ]
                color = get_block_color(params)
                if use_textures and textures and texture_idx is not None:
                    texture_path = textures[texture_idx]
                    if os.path.exists(texture_path):
                        tex_size = texture_sizes.get(texture_path, (1024, 1024))
                        region = (0, 0, tex_size[0], tex_size[1])
                        color = extract_texture_color(texture_path, region, downsample=params["texture_downsample"])
                block = [
                    params["block_type"],
                    square_center,
                    normal,
                    local_y,
                    [square_size * texture_scale, square_size * texture_scale, thickness],
                    True,
                    color
                ]
                blocks.append(block)
    return blocks

# Automatic UV Generation
@nb.jit(nopython=True)
def planar_uv_projection(vertices, normal):
    uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    if abs(normal[2]) > 0.707:
        for i in range(len(vertices)):
            uvs[i, 0] = vertices[i, 0]
            uvs[i, 1] = vertices[i, 1]
    elif abs(normal[1]) > 0.707:
        for i in range(len(vertices)):
            uvs[i, 0] = vertices[i, 0]
            uvs[i, 1] = vertices[i, 2]
    else:
        for i in range(len(vertices)):
            uvs[i, 0] = vertices[i, 1]
            uvs[i, 1] = vertices[i, 2]
    min_uv = np.min(uvs, axis=0)
    max_uv = np.max(uvs, axis=0)
    range_uv = max_uv - min_uv
    for i in range(len(uvs)):
        if range_uv[0] > 1e-8:
            uvs[i, 0] = (uvs[i, 0] - min_uv[0]) / range_uv[0]
        if range_uv[1] > 1e-8:
            uvs[i, 1] = (uvs[i, 1] - min_uv[1]) / range_uv[1]
    return uvs

def generate_uvs(face_indices, vertices, projection_type="planar"):
    vertices_np = np.array([vertices[i] for i in face_indices], dtype=np.float32)
    normal = compute_polygon_normal(vertices_np)
    if projection_type == "planar":
        uvs = planar_uv_projection(vertices_np, normal)
    else:
        uvs = planar_uv_projection(vertices_np, normal)
    logger.info(f"Generated UVs for face with {len(face_indices)} vertices")
    return uvs.tolist()

# Texture Processing Functions
def load_textures(texture_folder, model_name=None):
    global textures, texture_sizes
    textures = []
    start_time = time.time()
    possible_texture_dirs = ['texture', 'Textures']
    texture_folder_path = None
    for dir_name in possible_texture_dirs:
        candidate = os.path.join(texture_folder, dir_name)
        if os.path.exists(candidate):
            texture_folder_path = candidate
            break
    if not texture_folder_path:
        logger.error(f"No texture folder found at {texture_folder}")
        console.print(f"[red]No texture folder found at {texture_folder}[/red]")
        return False

    texture_files = []
    if model_name:
        texture_files.extend(glob.glob(os.path.join(texture_folder_path, f"{model_name}*")))
    texture_files.extend(glob.glob(os.path.join(texture_folder_path, "*.png")))
    texture_files.extend(glob.glob(os.path.join(texture_folder_path, "*.jpg")))
    texture_files.extend(glob.glob(os.path.join(texture_folder_path, "*.jpeg")))
    texture_files.extend(glob.glob(os.path.join(texture_folder_path, "*.dds")))

    textures = list(set(texture_files))
    texture_sizes.clear()
    for texture_path in textures:
        try:
            with Image.open(texture_path) as img:
                texture_sizes[texture_path] = img.size
                logger.info(f"Loaded texture: {texture_path}, size: {img.size}")
        except Exception as e:
            logger.error(f"Failed to load texture {texture_path}: {e}")
            texture_sizes[texture_path] = (1024, 1024)
    logger.info(f"Loaded {len(textures)} textures in {time.time() - start_time:.2f} seconds")
    console.print(f"[green]Loaded {len(textures)} textures from {texture_folder_path}[/green]")
    return len(textures) > 0

@nb.jit(nopython=True)
def fast_uv_polygon_pixels(face_uv_indices, uvs, tex_width, tex_height):
    points = np.zeros((len(face_uv_indices), 2), dtype=np.int32)
    for i in range(len(face_uv_indices)):
        idx = face_uv_indices[i]
        if idx != -1 and idx < len(uvs):
            u, v = uvs[idx]
            u = max(0.0, min(1.0, u % 1.0))
            v = max(0.0, min(1.0, v % 1.0))
            points[i, 0] = int(u * (tex_width - 1))
            points[i, 1] = int((1 - v) * (tex_height - 1))
    return points

def get_uv_polygon_pixels(face_uv_indices, uvs, texture_size):
    tex_width, tex_height = texture_size
    uv_indices = np.array([-1 if idx is None or idx >= len(uvs) else idx for idx in face_uv_indices], dtype=np.int32)
    return fast_uv_polygon_pixels(uv_indices, np.array(uvs, dtype=np.float32), tex_width, tex_height)

@nb.jit(nopython=True)
def fast_adjusted_bbox(points, tex_width, tex_height, min_crop):
    if len(points) == 0:
        return (0, 0, tex_width, tex_height)
    xs = points[:, 0]
    ys = points[:, 1]
    left = max(0, np.min(xs))
    right = min(tex_width, np.max(xs))
    upper = max(0, np.min(ys))
    lower = min(tex_height, np.max(ys))
    width = right - left
    height = lower - upper
    if width < min_crop:
        center_x = (left + right) // 2
        left = max(0, center_x - min_crop // 2)
        right = min(tex_width, left + min_crop)
    if height < min_crop:
        center_y = (upper + lower) // 2
        upper = max(0, center_y - min_crop // 2)
        lower = min(tex_height, upper + min_crop)
    return (left, upper, right, lower)

def get_adjusted_bbox(points, texture_size):
    tex_width, tex_height = texture_size
    params = get_params()
    min_crop = int(params["min_crop_size"])
    return fast_adjusted_bbox(points, tex_width, tex_height, min_crop)

def get_uv_crop_region(face_uv_indices, uvs, texture_size):
    points = get_uv_polygon_pixels(face_uv_indices, uvs, texture_size)
    return get_adjusted_bbox(points, texture_size)

def extract_texture_color(texture_path, region, downsample=16):
    cache_key = (texture_path, region, downsample)
    if cache_key in texture_color_cache:
        texture_color_cache.move_to_end(cache_key)
        logger.info(f"Retrieved color from cache for {texture_path}")
        return texture_color_cache[cache_key]
    
    try:
        start_time = time.time()
        with Image.open(texture_path) as img:
            img = img.convert("RGB")
            left, upper, right, lower = region
            if left >= right or upper >= lower:
                logger.warning(f"Invalid crop region for {texture_path}: {region}")
                return [233, 218, 218]
            cropped = img.crop((left, upper, right, lower))
            if downsample > 1:
                new_size = (max(1, cropped.width // downsample), max(1, cropped.height // downsample))
                cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)
            pixels = np.array(cropped)
            if pixels.size == 0:
                logger.warning(f"Empty pixel array for {texture_path}")
                return [233, 218, 218]
            avg_color = np.mean(pixels, axis=(0, 1))
            color = [int(c) for c in avg_color]
            
            texture_color_cache[cache_key] = color
            if len(texture_color_cache) > MAX_CACHE_SIZE:
                texture_color_cache.popitem(last=False)
            logger.info(f"Extracted color {color} from {texture_path} in {time.time() - start_time:.2f} seconds")
            return color
    except Exception as e:
        logger.error(f"Texture extract error {texture_path}: {e}")
        return [233, 218, 218]

# Model Processing Functions
@nb.jit(nopython=True)
def compute_polygon_normal(polygon):
    if len(polygon) < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v1 = polygon[1] - polygon[0]
    v2 = polygon[2] - polygon[0]
    normal = np.cross(v1, v2)
    norm = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    return normal / np.float32(norm) if norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

@nb.jit(nopython=True)
def create_block_from_face_jit(face_indices, vertices, face_inward_offset, min_size, min_thickness, texture_scale, 
                              fallback_color_r, fallback_color_g, fallback_color_b):
    polygon = np.zeros((len(face_indices), 3), dtype=np.float32)
    for i in range(len(face_indices)):
        idx = face_indices[i]
        polygon[i, 0] = vertices[idx, 0]
        polygon[i, 1] = vertices[idx, 1]
        polygon[i, 2] = vertices[idx, 2]

    centroid = np.sum(polygon, axis=0) / np.float32(len(polygon))
    normal = compute_polygon_normal(polygon)
    centroid_adjusted = centroid + normal * face_inward_offset
    
    arbitrary = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(normal[2]) < 0.99 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    block_x = np.cross(normal, arbitrary)
    norm_x = np.sqrt(block_x[0]**2 + block_x[1]**2 + block_x[2]**2)
    block_x = block_x / np.float32(norm_x) if norm_x > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    block_y = np.cross(normal, block_x)
    norm_y = np.sqrt(block_y[0]**2 + block_y[1]**2 + block_y[2]**2)
    block_y = block_y / np.float32(norm_y) if norm_y > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    proj_x = np.zeros(len(polygon), dtype=np.float32)
    proj_y = np.zeros(len(polygon), dtype=np.float32)
    for i in range(len(polygon)):
        diff = polygon[i] - centroid
        proj_x[i] = np.dot(diff, block_x)
        proj_y[i] = np.dot(diff, block_y)
    size_x = (np.max(proj_x) - np.min(proj_x)) * texture_scale
    size_y = (np.max(proj_y) - np.min(proj_y)) * texture_scale
    
    size_x = max(size_x, min_size)
    size_y = max(size_y, min_size)
    
    block = (
        "PlasticBlock",
        [float(centroid_adjusted[0]), float(centroid_adjusted[1]), float(centroid_adjusted[2])],
        [float(normal[0]), float(normal[1]), float(normal[2])],
        [float(block_y[0]), float(block_y[1]), float(block_y[2])],
        [float(size_x), float(size_y), float(min_thickness)],
        True,
        [fallback_color_r, fallback_color_g, fallback_color_b]
    )
    return [block]

def create_block_from_face(face_tuple, vertices, uvs, texture_idx, params, auto_uvs=None, material=None):
    face_indices, face_uv = face_tuple
    vertices_np = np.array(vertices, dtype=np.float32)
    blocks = create_block_from_face_jit(
        np.array(face_indices, dtype=np.int32), vertices_np,
        face_inward_offset=params.get("face_inward_offset", params["position_offset"]),
        min_size=params["min_size"],
        min_thickness=params["min_thickness"],
        texture_scale=params.get("texture_scale", 1.0),
        fallback_color_r=params["fallback_color"][0],
        fallback_color_g=params["fallback_color"][1],
        fallback_color_b=params["fallback_color"][2]
    )
    
    if params.get("use_textures", True) and textures and texture_idx is not None:
        if 0 <= texture_idx < len(textures):
            texture_path = textures[texture_idx]
            logger.info(f"Applying texture {texture_path} for material {material} to face")
            if os.path.exists(texture_path):
                tex_size = texture_sizes.get(texture_path, (1024, 1024))
                effective_uvs = uvs if face_uv and any(idx is not None and idx < len(uvs) for idx in face_uv) else auto_uvs
                if effective_uvs:
                    uv_indices = face_uv if face_uv else list(range(len(auto_uvs)))
                    region = get_uv_crop_region(uv_indices, effective_uvs, tex_size)
                    color = extract_texture_color(texture_path, region, downsample=params["texture_downsample"])
                    logger.info(f"Applied texture color {color} for face with texture {texture_path}, material {material}")
                    blocks[0] = list(blocks[0])
                    blocks[0][-1] = color
                    blocks[0] = tuple(blocks[0])
                else:
                    logger.warning(f"No valid UVs for face (material {material}), using fallback color")
            else:
                logger.error(f"Texture file {texture_path} does not exist")
        else:
            logger.warning(f"Invalid texture_idx {texture_idx} for material {material}, using fallback color")
    
    converted_blocks = []
    for block in blocks:
        converted_block = [
            block[0],
            [float(x) for x in block[1]],
            [float(x) for x in block[2]],
            [float(x) for x in block[3]],
            [float(x) for x in block[4]],
            block[5],
            block[6]
        ]
        converted_blocks.append(converted_block)
    return converted_blocks

def process_face(args):
    face_tuple, vertices, uvs, texture_idx, params, idx, total, auto_uvs, material = args
    start_time = time.time()
    blocks = create_block_from_face(face_tuple, vertices, uvs, texture_idx, params, auto_uvs, material)
    elapsed = time.time() - start_time
    if elapsed > 0.1:
        logger.info(f"Processed face {idx + 1}/{total} in {elapsed:.2f} seconds")
    print_progress_bar(idx + 1, total, prefix='Processing faces', suffix='Done')
    return blocks

def parse_mtl_file(mtl_path, texture_folder):
    material_textures = {}
    current_material = None
    try:
        with open(mtl_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'newmtl':
                    current_material = parts[1]
                elif parts[0].lower() == 'map_kd' and current_material:
                    texture_name = ' '.join(parts[1:])
                    texture_path = os.path.join(texture_folder, texture_name)
                    if not os.path.exists(texture_path):
                        mtl_dir = os.path.dirname(mtl_path)
                        texture_path = os.path.join(mtl_dir, texture_name)
                    if not os.path.exists(texture_path):
                        logger.warning(f"Texture {texture_name} for material {current_material} not found")
                        continue
                    material_textures[current_material] = texture_path
                    logger.info(f"MTL mapped material {current_material} to texture {texture_path}")
    except Exception as e:
        logger.error(f"Error parsing MTL file {mtl_path}: {e}")
    return material_textures

def process_data(model_path, base_folder, output_folder=None):
    start_total = time.time()
    params = get_params()
    mode = profiles[current_profile]["mode"]
    buffer_size = params.get("buffer_size", 10000)
    max_threads = params.get("max_threads", max(1, os.cpu_count() // 2))
    use_textures = params.get("use_textures", True)
    auto_uv = params.get("auto_uv", True)
    
    if output_folder is None:
        output_folder = os.path.join(base_folder, "output")
    os.makedirs(output_folder, exist_ok=True)
    model_filename = os.path.basename(model_path)
    model_name, _ = os.path.splitext(model_filename)
    output_path = os.path.join(output_folder, f"{model_name}.build")

    vertices = []
    uvs = []
    face_count = 0
    material_textures = {}
    texture_map = {}
    face_materials = []

    mtl_path = os.path.splitext(model_path)[0] + '.mtl'
    texture_folder = os.path.join(base_folder, "Textures")
    if os.path.exists(mtl_path):
        material_textures = parse_mtl_file(mtl_path, texture_folder)

    logger.info(f"Starting to count faces for {model_path}")
    start_count = time.time()
    current_material = None
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('f '):
                    face_count += 1
                    face_materials.append(current_material)
                elif line.startswith('usemtl'):
                    current_material = line.strip().split()[1]
    except Exception as e:
        logger.error(f"Error reading model {model_path}: {e}")
        raise
    logger.info(f"Counted {face_count} faces in {time.time() - start_count:.2f} seconds")

    if face_count == 0:
        logger.error(f"No faces found in {model_path}")
        raise ValueError("Model contains no faces to process.")

    if use_textures:
        texture_found = load_textures(base_folder, model_name)
        if material_textures:
            for mat, tex_path in material_textures.items():
                if os.path.exists(tex_path) and tex_path not in textures:
                    textures.append(tex_path)
                    try:
                        with Image.open(tex_path) as img:
                            texture_sizes[tex_path] = img.size
                            logger.info(f"Added MTL texture: {tex_path}")
                    except Exception as e:
                        logger.error(f"Texture load error {tex_path}: {e}")
                        texture_sizes[tex_path] = (1024, 1024)
                texture_map[mat] = textures.index(tex_path) if tex_path in textures else None
        if not texture_found or not textures:
            logger.warning("No textures loaded, using fallback colors")
            console.print("[yellow]No textures loaded, using fallback colors[/yellow]")

    logger.info(f"Starting processing for {model_path} with {max_threads} threads")
    start_processing = time.time()
    output_writer = OutputWriter(output_path, params["blocks_per_file"])

    with open(model_path, 'r', encoding='utf-8') as f:
        if mode == 'faces':
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = []
                idx = 0
                has_uvs = False
                
                for line in f:
                    parts = line.strip().split()
                    if line.startswith('v '):
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith('vt '):
                        uvs.append([float(parts[1]), float(parts[2])])
                        has_uvs = True
                    elif line.startswith('usemtl'):
                        current_material = parts[1]
                
                logger.info(f"Model has UVs: {has_uvs}")
                if not has_uvs and auto_uv:
                    logger.info("No UVs found, will use auto UV generation")
                
                f.seek(0)
                idx = 0
                for line in f:
                    parts = line.strip().split()
                    if line.startswith('f '):
                        vertex_indices = [int(part.split('/')[0]) - 1 for part in parts[1:]]
                        uv_indices = [int(part.split('/')[1]) - 1 if len(part.split('/')) > 1 and part.split('/')[1] else None
                                      for part in parts[1:]]
                        auto_uvs = None
                        if not has_uvs and auto_uv:
                            auto_uvs = generate_uvs(vertex_indices, vertices, params.get("uv_projection", "planar"))
                            uv_indices = list(range(len(auto_uvs)))
                            logger.info(f"Generated auto UVs for face {idx + 1}")
                        if len(vertex_indices) >= 3:
                            face_tuple = (vertex_indices, uv_indices)
                            material = face_materials[idx]
                            texture_idx = texture_map.get(material, 0 if textures else None)
                            if texture_idx is None and textures:
                                logger.warning(f"No texture for material {material}, using first texture")
                                texture_idx = 0
                            futures.append(executor.submit(
                                process_face,
                                (face_tuple, vertices, uvs, texture_idx, params, idx, face_count, auto_uvs, material)
                            ))
                            idx += 1
                        
                        if len(futures) >= buffer_size or idx == face_count:
                            start_write = time.time()
                            for future in futures:
                                try:
                                    blocks = future.result(timeout=60)
                                    for block in blocks:
                                        output_writer.write_block(block)
                                except TimeoutError:
                                    logger.error(f"Timeout processing face {idx}")
                                    continue
                            logger.info(f"Wrote buffer of {len(futures)} faces in {time.time() - start_write:.2f} seconds")
                            futures = []
                
                start_write = time.time()
                for future in futures:
                    try:
                        blocks = future.result(timeout=60)
                        for block in blocks:
                            output_writer.write_block(block)
                    except TimeoutError:
                        logger.error(f"Timeout processing remaining face")
                        continue
                logger.info(f"Wrote remaining {len(futures)} faces in {time.time() - start_write:.2f} seconds")
        
        elif mode == 'edges':
            edges_to_normals = {}
            edges = []
            current_material = None
            for line in f:
                parts = line.strip().split()
                if line.startswith('v '):
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('usemtl'):
                    current_material = parts[1]
                elif line.startswith('f '):
                    vertex_indices = [int(part.split('/')[0]) - 1 for part in parts[1:]]
                    if len(vertex_indices) >= 3:
                        face_normal = compute_polygon_normal(np.array([vertices[i] for i in vertex_indices], dtype=np.float32))
                        for i in range(len(vertex_indices)):
                            edge = frozenset([vertex_indices[i], vertex_indices[(i + 1) % len(vertex_indices)]])
                            if edge not in edges_to_normals:
                                edges.append((edge, current_material))
                            edges_to_normals.setdefault(edge, []).append(face_normal)
            
            total = len(edges)
            start_write = time.time()
            for idx, (edge, edge_material) in enumerate(edges, start=1):
                idx_list = list(edge)
                p1 = np.array(vertices[idx_list[0]], dtype=np.float32)
                p2 = np.array(vertices[idx_list[1]], dtype=np.float32)
                center = (p1 + p2) / 2
                direction = p2 - p1
                length = np.linalg.norm(direction)
                if length < 1e-8:
                    continue
                direction = direction / length
                normals = np.array(edges_to_normals[edge], dtype=np.float32)
                avg_normal = np.sum(normals, axis=0) / np.float32(len(normals))
                norm_val = np.linalg.norm(avg_normal)
                avg_normal = avg_normal / norm_val if norm_val > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
                local_y = np.cross(avg_normal, direction)
                norm_y = np.linalg.norm(local_y)
                if norm_y < 1e-8:
                    local_y = np.array([0, 1, 0], dtype=np.float32)
                else:
                    local_y = local_y / norm_y
                edge_offset = params.get("edge_inward_offset", params["position_offset"])
                center_adjusted = center + avg_normal * edge_offset
                v0_2d = (0, 0)
                v1_2d = (length, 0)
                edge_thickness = params["edge_min_size"]
                polygon_2d = [
                    v0_2d,
                    v1_2d,
                    (v1_2d[0], edge_thickness),
                    (v0_2d[0], edge_thickness)
                ]
                texture_idx = texture_map.get(edge_material, 0 if textures else None)
                blocks = edpn_fill_polygon(
                    polygon_2d,
                    params,
                    center_adjusted.tolist(),
                    avg_normal.tolist(),
                    local_y.tolist(),
                    texture_idx,
                    edge_material
                )
                for block in blocks:
                    output_writer.write_block(block)
                print_progress_bar(idx, total, prefix='Processing edges with EDPN', suffix='Done', length=40)
            logger.info(f"Wrote {total} edges with EDPN in {time.time() - start_write:.2f} seconds")
        
        else:
            logger.error(f"Unknown processing mode: {mode}")
            console.print(f"[red]Unknown processing mode: {mode}. Available: faces, edges[/red]")
            return None
    
    output_writer.close()
    total_time = time.time() - start_total
    logger.info(f"Total processing time for {model_path}: {total_time:.2f} seconds")
    if params["blocks_per_file"] > 0:
        console.print(f"[green]Processing complete. Results saved to multiple files: {output_folder}/{model_name}_part*.build[/green]")
    else:
        console.print(f"[green]Processing complete. Result saved: {output_path}[/green]")
    console.print(f"[green]Total time: {total_time:.2f} seconds[/green]")
    return output_path if params["blocks_per_file"] == 0 else [os.path.join(output_folder, f"{model_name}_part{i}.build") for i in range(1, output_writer.file_index + 1)]

# Console Interface Functions
def console_menu():
    console.clear()
    terminal_settings = get_terminal_settings()
    console.print(Panel.fit(
        f"[bold #00FFFF]Welcome to 3D Model Processor!]\ntelegram: https://t.me/BABFT_sript_original [/bold #00FFFF]\nCurrent profile: {current_profile}",
        border_style=terminal_settings["border_style"],
        box=DOUBLE if terminal_settings["panel_box"] == "DOUBLE" else ROUNDED
    ))
    table = Table(title="Available Commands", show_header=True, header_style=terminal_settings["header_color"])
    table.add_column("Command", style="#00FFFF", justify="left")
    table.add_column("Description", style="#00FF00", justify="left")
    table.add_row("1. setparams", "Configure current profile parameters")
    table.add_row("2. setmode", "Select processing mode ('faces', 'edges')")
    table.add_row("3. process", "Process one or more models")
    table.add_row("4. showparams", "Show current parameters")
    table.add_row("5. resetparams", "Reset current profile parameters")
    table.add_row("6. profile", "Manage profiles (create, switch, delete)")
    table.add_row("7. history", "Show command history")
    table.add_row("8. customize_terminal", "Customize terminal appearance")
    table.add_row("9. help", "Open program guide in browser")
    table.add_row("10. exit", "Exit the program")
    console.print(table)
    console.print(f"[{terminal_settings['prompt_color']}]Enter command number or name below[/{terminal_settings['prompt_color']}]")

def set_params_console():
    terminal_settings = get_terminal_settings()
    console.print("[bold #00FFFF]Parameter Configuration[/bold #00FFFF]")
    console.print("Enter parameters as: key=value (space-separated). E.g., min_size=0.01 blocks_per_file=5000")
    console.print(f"Available parameters: {', '.join(DEFAULT_PARAMS.keys())}")

    inp = Prompt.ask(f"[{terminal_settings['prompt_color']}]Parameters[/{terminal_settings['prompt_color']}]").strip()
    command_history.append(f"setparams {inp}")
    new_params = {}

    for part in inp.split():
        if '=' not in part:
            console.print(f"[red]Error: '{part}' missing '='[/red]")
            console_menu()
            return
        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()

        try:
            if key in ["min_thickness", "min_size", "position_offset", "face_inward_offset",
                       "edge_inward_offset", "edge_min_size", "edge_min_thickness", "texture_scale",
                       "min_crop_size", "edpn_scale_factor", "edpn_min_area", "edpn_square_size"]:
                new_params[key] = float(value)
            elif key in ["texture_downsample", "max_threads", "buffer_size", "blocks_per_file"]:
                new_params[key] = int(value)
                if key == "max_threads" and new_params[key] < 1:
                    raise ValueError("max_threads must be at least 1")
                if key == "blocks_per_file" and new_params[key] < 0:
                    raise ValueError("blocks_per_file must be non-negative")
            elif key == "fallback_color":
                color_values = [int(x) for x in value.split(',')]
                if len(color_values) != 3:
                    raise ValueError("Invalid color format. Use: r,g,b")
                new_params[key] = color_values
            elif key == "color_palette":
                groups = value.split(';')
                palette = []
                for group in groups:
                    if group.strip():
                        color = [int(x) for x in group.split(',')]
                        if len(color) != 3:
                            raise ValueError("Invalid palette format. Use: r1,g1,b1;r2,g2,b2;...")
                        palette.append(color)
                new_params[key] = palette
            elif key in ["use_random_color", "use_textures", "auto_uv"]:
                new_params[key] = value.lower() in ["true", "1", "yes"]
            elif key in ["block_type", "uv_projection"]:
                new_params[key] = value
            else:
                console.print(f"[red]Unknown parameter: {key}[/red]")
                console_menu()
                return
        except ValueError as e:
            console.print(f"[red]Error in parameter '{key}': {str(e)}[/red]")
            console_menu()
            return

    profiles[current_profile]["params"].update(new_params)
    save_profiles()
    console.print(f"[green]Parameters for profile '{current_profile}' updated[/green]")
    console_menu()

def set_mode_console():
    terminal_settings = get_terminal_settings()
    console.print("[bold #00FFFF]Processing Mode Selection[/bold #00FFFF]")
    available_modes = ['faces', 'edges']
    console.print(f"Available modes: {available_modes}")
    mode = Prompt.ask(f"[{terminal_settings['prompt_color']}]Enter mode[/{terminal_settings['prompt_color']}]",
                      choices=available_modes, default=profiles[current_profile]["mode"]).lower()
    command_history.append(f"setmode {mode}")
    profiles[current_profile]["mode"] = mode
    save_profiles()
    console.print(f"[green]Mode set to: {mode} for profile '{current_profile}'[/green]")
    console_menu()

def show_params_console():
    terminal_settings = get_terminal_settings()
    console.print(f"[bold #00FFFF]Current Parameters for Profile '{current_profile}'[/bold #00FFFF]")
    table = Table(show_header=True, header_style=terminal_settings["header_color"])
    table.add_column("Parameter", style="#00FFFF")
    table.add_column("Value", style="#00FF00")
    params = get_params()
    for key, value in params.items():
        if key == "fallback_color":
            r, g, b = value
            table.add_row(key, f"[rgb({r},{g},{b})]███[/rgb({r},{g},{b})] {value}")
        else:
            table.add_row(key, str(value))
    table.add_row("mode", profiles[current_profile]["mode"])
    console.print(table)
    console_menu()

def reset_params_console():
    terminal_settings = get_terminal_settings()
    if Confirm.ask(f"[{terminal_settings['prompt_color']}]Reset parameters for '{current_profile}' to defaults?[/{terminal_settings['prompt_color']}]"):
        profiles[current_profile] = {"params": DEFAULT_PARAMS.copy(), "mode": "faces", "terminal": DEFAULT_TERMINAL.copy()}
        save_profiles()
        update_console()
        console.print(f"[green]Parameters for profile '{current_profile}' reset[/green]")
    console_menu()

def process_console():
    terminal_settings = get_terminal_settings()
    console.print("[bold #00FFFF]Model Processing[/bold #00FFFF]")
    folder_path = Prompt.ask(f"[{terminal_settings['prompt_color']}]Enter path to folder with 'source' and 'texture'[/{terminal_settings['prompt_color']}]").strip()
    command_history.append(f"process {folder_path}")

    if not os.path.isdir(folder_path):
        console.print("[red]Folder not found. Check the path.[/red]")
        console_menu()
        return

    source_folder = os.path.join(folder_path, "source")
    if not os.path.isdir(source_folder):
        console.print(f"[red]'source' folder not found in {folder_path}.[/red]")
        console_menu()
        return

    obj_files = glob.glob(os.path.join(source_folder, "*.obj"))
    if not obj_files:
        console.print("[red]No .obj files found in 'source' folder.[/red]")
        console_menu()
        return

    batch_mode = Confirm.ask(f"[{terminal_settings['prompt_color']}]Process all models in 'source'? (No - first only)[/{terminal_settings['prompt_color']}]")
    process_files = obj_files if batch_mode else [obj_files[0]]

    for model_path in process_files:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        console.print(f"[green]Processing file: {model_path}[/green]")

        output_folder = os.path.join(folder_path, "output")
        try:
            with console.status(f"[bold green]Processing {model_name}...[/bold green]", spinner="dots"):
                result = process_data(model_path, base_folder=folder_path, output_folder=output_folder)
                if result is None:
                    console.print(f"[red]Processing failed for {model_name}[/red]")
                    logger.error(f"Processing failed for {model_name}")
        except Exception as e:
            console.print(f"[red]Error processing {model_name}: {e}[/red]")
            logger.error(f"Error processing {model_name}: {e}")

    console.print(f"[bold green]Processing complete for all models.[/bold green]")
    Prompt.ask(f"[{terminal_settings['prompt_color']}]Press Enter to continue...[/{terminal_settings['prompt_color']}]")
    console_menu()

def profile_console():
    global current_profile
    terminal_settings = get_terminal_settings()
    console.print("[bold #00FFFF]Profile Management[/bold #00FFFF]")
    action = Prompt.ask(f"[{terminal_settings['prompt_color']}]Select action[/{terminal_settings['prompt_color']}]",
                        choices=["list", "switch", "create", "delete"], default="list")
    command_history.append(f"profile {action}")

    if action == "list":
        table = Table(title="Profile List", show_header=True, header_style=terminal_settings["header_color"])
        table.add_column("Name", style="#00FFFF")
        table.add_column("Active", style="#00FF00")
        for profile_name in profiles.keys():
            active = "Yes" if profile_name == current_profile else "No"
            table.add_row(profile_name, active)
        console.print(table)

    elif action == "switch":
        profile_name = Prompt.ask(f"[{terminal_settings['prompt_color']}]Enter profile name to switch[/{terminal_settings['prompt_color']}]",
                                  choices=list(profiles.keys()), default=current_profile)
        current_profile = profile_name
        update_console()
        console.print(f"[green]Switched to profile: {current_profile}[/green]")

    elif action == "create":
        profile_name = Prompt.ask(f"[{terminal_settings['prompt_color']}]Enter new profile name[/{terminal_settings['prompt_color']}]").strip()
        if profile_name in profiles:
            console.print(f"[red]Profile '{profile_name}' already exists.[/red]")
        else:
            profiles[profile_name] = {"params": DEFAULT_PARAMS.copy(), "mode": "faces", "terminal": DEFAULT_TERMINAL.copy()}
            save_profiles()
            console.print(f"[green]Created profile: {profile_name}[/green]")

    elif action == "delete":
        profile_name = Prompt.ask(f"[{terminal_settings['prompt_color']}]Enter profile name to delete[/{terminal_settings['prompt_color']}]",
                                  choices=list(profiles.keys()), default=current_profile)
        if profile_name == "default":
            console.print("[red]Cannot delete 'default' profile.[/red]")
        elif Confirm.ask(f"[{terminal_settings['prompt_color']}]Delete profile '{profile_name}'?[/{terminal_settings['prompt_color']}]"):
            del profiles[profile_name]
            if profile_name == current_profile:
                current_profile = "default"
                update_console()
            save_profiles()
            console.print(f"[green]Profile '{profile_name}' deleted[/green]")
    
    console_menu()

def history_console():
    terminal_settings = get_terminal_settings()
    console.print("[bold #00FFFF]Command History[/bold #00FFFF]")
    if not command_history:
        console.print("[yellow]History is empty.[/yellow]")
    else:
        table = Table(show_header=True, header_style=terminal_settings["header_color"])
        table.add_column("#", style="#00FFFF")
        table.add_column("Command", style="#00FF00")
        for i, cmd in enumerate(command_history, 1):
            table.add_row(str(i), cmd)
        console.print(table)
    console_menu()

def customize_terminal_console():
    terminal_settings = get_terminal_settings()
    console.print("[bold #00FFFF]Terminal Customization[/bold #00FFFF]")
    console.print("Enter settings as: key=value (space-separated). E.g., text_color=red background_color=#FFFFFF")
    console.print("Available: text_color, background_color (#RRGGBB or name), progress_bar_style (bar.back/bar.complete),")
    console.print("header_color, border_style, prompt_color, panel_box (DOUBLE/ROUNDED)")

    inp = Prompt.ask(f"[{terminal_settings['prompt_color']}]Terminal settings[/{terminal_settings['prompt_color']}]").strip()
    command_history.append(f"customize_terminal {inp}")
    new_settings = {}

    for part in inp.split():
        if '=' not in part:
            console.print(f"[red]Error: '{part}' missing '='[/red]")
            console_menu()
            return
        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()

        if key in ["text_color", "background_color", "header_color", "border_style", "prompt_color"]:
            if value.startswith("#") and len(value) == 7:
                try:
                    tuple(int(value.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    new_settings[key] = value
                except ValueError:
                    console.print(f"[red]Invalid hex color for {key}: {value}. Use #RRGGBB[/red]")
                    console_menu()
                    return
            else:
                new_settings[key] = value
        elif key == "progress_bar_style":
            if value in ["bar.back", "bar.complete"]:
                new_settings[key] = value
            else:
                console.print(f"[red]Invalid progress_bar_style: {value}. Use 'bar.back' or 'bar.complete'[/red]")
                console_menu()
                return
        elif key == "panel_box":
            if value in ["DOUBLE", "ROUNDED"]:
                new_settings[key] = value
            else:
                console.print(f"[red]Invalid panel_box: {value}. Use 'DOUBLE' or 'ROUNDED'[/red]")
                console_menu()
                return
        else:
            console.print(f"[red]Unknown parameter: {key}[/red]")
            console_menu()
            return

    profiles[current_profile]["terminal"].update(new_settings)
    save_profiles()
    update_console()
    console.print(f"[green]Terminal settings for profile '{current_profile}' updated[/green]")
    console_menu()

def help_console():
    terminal_settings = get_terminal_settings()
    console.print("[bold #00FFFF]Opening Program Guide[/bold #00FFFF]")
    command_history.append("help")
    
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    guide_path = os.path.join(parent_dir, "GUIDE OF PROGRAM.html")
    
    if os.path.exists(guide_path):
        console.print(f"[green]Found guide at: {guide_path}. Opening in browser...[/green]")
        webbrowser.open(f"file://{os.path.abspath(guide_path)}")
    else:
        console.print(f"[red]Guide file 'GUIDE OF PROGRAM.html' not found in {parent_dir}.[/red]")
    console_menu()

def check_first_run():
    if not os.path.exists(FIRST_RUN_FLAG):
        console.print("[yellow]This is your first time running the program. Please read the guide before proceeding.[/yellow]")
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        guide_path = os.path.join(parent_dir, "GUIDE OF PROGRAM.html")
        
        if os.path.exists(guide_path):
            console.print(f"[green]Opening guide at: {guide_path}[/green]")
            webbrowser.open(f"file://{os.path.abspath(guide_path)}")
        else:
            console.print(f"[red]Guide file 'GUIDE OF PROGRAM.html' not found in {parent_dir}. Please ensure it exists.[/red]")
        
        with open(FIRST_RUN_FLAG, 'w') as f:
            f.write("First run completed.")
        console.print("[green]First run flag created. You won't see this message again.[/green]")

def main_console():
    check_first_run()
    console_menu()
    while True:
        terminal_settings = get_terminal_settings()
        cmd = Prompt.ask(f"[{terminal_settings['prompt_color']}]Enter command[/{terminal_settings['prompt_color']}]").strip().lower()

        if cmd in ["setparams", "1"]:
            set_params_console()
        elif cmd in ["setmode", "2"]:
            set_mode_console()
        elif cmd in ["process", "3"]:
            process_console()
        elif cmd in ["showparams", "4"]:
            show_params_console()
        elif cmd in ["resetparams", "5"]:
            reset_params_console()
        elif cmd in ["profile", "6"]:
            profile_console()
        elif cmd in ["history", "7"]:
            history_console()
        elif cmd in ["customize_terminal", "8"]:
            customize_terminal_console()
        elif cmd in ["help", "9"]:
            help_console()
        elif cmd in ["exit", "10"]:
            console.print("[bold red]Exiting program...[/bold red]")
            break
        else:
            console.print("[red]Unknown command. Try again.[/red]")
            console_menu()

# Entry Point
if __name__ == '__main__':
    main_console()
