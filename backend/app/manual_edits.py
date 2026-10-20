"""Reusable full-resolution, non-destructive survey-map editing helpers."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def display_to_source_point(x: float, y: float, left: float, top: float,
                            display_width: float, display_height: float,
                            source_width: int, source_height: int,
                            rotation_degrees: float = 0) -> tuple[int, int]:
    """Map a screen point through a centered CSS rotation and downscale to source pixels."""
    if display_width <= 0 or display_height <= 0 or source_width <= 0 or source_height <= 0:
        raise ValueError('Image and display dimensions must be positive.')
    theta = math.radians(rotation_degrees)
    bbox_width = abs(display_width * math.cos(theta)) + abs(display_height * math.sin(theta))
    bbox_height = abs(display_width * math.sin(theta)) + abs(display_height * math.cos(theta))
    dx = x - (left + bbox_width / 2)
    dy = y - (top + bbox_height / 2)
    # Undo the centered display rotation before scaling into source coordinates.
    ux = dx * math.cos(theta) + dy * math.sin(theta) + display_width / 2
    uy = -dx * math.sin(theta) + dy * math.cos(theta) + display_height / 2
    sx = round(ux * source_width / display_width)
    sy = round(uy * source_height / display_height)
    return min(source_width - 1, max(0, sx)), min(source_height - 1, max(0, sy))


def apply_manual_operations(source_path: str | Path, output_path: str | Path,
                            operations: list[dict[str, Any]]) -> None:
    """Apply vector edits to a candidate, never to the uploaded source."""
    with Image.open(source_path) as opened:
        image = opened.convert('RGB')
    draw = ImageDraw.Draw(image)
    for op in operations:
        kind = str(op.get('tool', ''))
        points = [(int(p[0]), int(p[1])) for p in op.get('points', [])]
        width = max(1, min(4096, int(op.get('width', 1))))
        color = '#ffffff' if kind == 'erase' else str(op.get('color', '#ffffff'))
        if kind in {'brush', 'background', 'erase', 'line'} and points:
            if len(points) == 1:
                x, y = points[0]; r = max(1, width // 2)
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
            else:
                draw.line(points, fill=color, width=width, joint='curve')
                r = max(1, width // 2)
                for x, y in (points[0], points[-1]): draw.ellipse((x-r,y-r,x+r,y+r),fill=color)
        elif kind == 'selection-clear' and len(points) >= 3:
            draw.polygon(points, fill=color)
        elif kind == 'repair' and points and op.get('sample'):
            sx, sy = map(int, op['sample']); radius = max(1, width // 2)
            patch_box=(max(0,sx-radius),max(0,sy-radius),min(image.width,sx+radius+1),min(image.height,sy+radius+1))
            sampled=image.crop(patch_box)
            for x, y in points:
                px=x-(sx-patch_box[0]); py=y-(sy-patch_box[1])
                image.paste(sampled,(max(0,min(image.width-sampled.width,px)),max(0,min(image.height-sampled.height,py))))
    image.save(output_path, format='PNG', optimize=False)
