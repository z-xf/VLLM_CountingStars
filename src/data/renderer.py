from PIL import Image, ImageDraw
import math
from typing import Dict, Any
from src.core.models import SceneMetadata, Shape

class ImageRenderer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def render(self, scene: SceneMetadata) -> Image.Image:
        img = Image.new("RGB", (scene.width, scene.height), "white")
        draw = ImageDraw.Draw(img)

        shapes = scene.stars + scene.distractors
        
        for shape in shapes:
            if shape.type == "star":
                self._draw_star(draw, shape)
            elif shape.type == "circle":
                self._draw_circle(draw, shape)
            elif shape.type == "square":
                self._draw_square(draw, shape)
            elif shape.type == "triangle":
                self._draw_triangle(draw, shape)
            elif shape.type == "diamond":
                self._draw_diamond(draw, shape)

        return img

    def _draw_star(self, draw: ImageDraw.ImageDraw, shape: Shape):
        # A simple 5-point star
        points = []
        outer_radius = shape.pixel_size
        inner_radius = shape.pixel_size * 0.4
        for i in range(10):
            angle = math.pi / 2 - i * math.pi / 5
            radius = outer_radius if i % 2 == 0 else inner_radius
            points.append((shape.x + radius * math.cos(angle), shape.y - radius * math.sin(angle)))
        draw.polygon(points, fill=shape.color)

    def _draw_circle(self, draw: ImageDraw.ImageDraw, shape: Shape):
        r = shape.pixel_size
        draw.ellipse([shape.x - r, shape.y - r, shape.x + r, shape.y + r], fill=shape.color)

    def _draw_square(self, draw: ImageDraw.ImageDraw, shape: Shape):
        r = shape.pixel_size
        draw.rectangle([shape.x - r, shape.y - r, shape.x + r, shape.y + r], fill=shape.color)

    def _draw_triangle(self, draw: ImageDraw.ImageDraw, shape: Shape):
        r = shape.pixel_size
        points = [
            (shape.x, shape.y - r),
            (shape.x - r * math.cos(math.pi/6), shape.y + r * math.sin(math.pi/6)),
            (shape.x + r * math.cos(math.pi/6), shape.y + r * math.sin(math.pi/6))
        ]
        draw.polygon(points, fill=shape.color)

    def _draw_diamond(self, draw: ImageDraw.ImageDraw, shape: Shape):
        r = shape.pixel_size
        points = [
            (shape.x, shape.y - r),
            (shape.x + r, shape.y),
            (shape.x, shape.y + r),
            (shape.x - r, shape.y)
        ]
        draw.polygon(points, fill=shape.color)

    def _draw_diamond(self, draw: ImageDraw.ImageDraw, shape: Shape):
        r = shape.pixel_size
        points = [
            (shape.x, shape.y - r),
            (shape.x + r, shape.y),
            (shape.x, shape.y + r),
            (shape.x - r, shape.y)
        ]
        draw.polygon(points, fill=shape.color)
