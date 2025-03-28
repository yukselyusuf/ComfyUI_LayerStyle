import torch
import numpy as np
from PIL import Image, ImageDraw
from .imagefunc import log, tensor2pil, pil2tensor, gaussian_blur, mask2image
from .imagefunc import min_bounding_rect, max_inscribed_rect, mask_area, draw_rect


class MaskBoxDetect:

    def __init__(self):
        self.NODE_NAME = 'MaskBoxDetect'
    
    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['min_bounding_rect', 'max_inscribed_rect', 'mask_area']
        return {
            "required": {
                "mask": ("MASK", ),
                "detect": (detect_mode,),  # 探测类型：最小外接矩形/最大内接矩形
                "x_adjust": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),  # x轴修正
                "y_adjust": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),  # y轴修正
                "scale_adjust": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100, "step": 0.01}), # 比例修正
                "circle_count": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),  # Yuvarlak sayısı
                "circle_size": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),  # Yuvarlak boyutu (alan oranı)
                "overflow": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),  # Taşma miktarı
                "straight_edge_threshold": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.01}),  # Düz kenar algılama eşiği
                "remove_original": ("BOOLEAN", {"default": True}),  # Orijinal beyaz alanı kaldır
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("box_preview", "x_percent", "y_percent", "width", "height", "x", "y",)
    FUNCTION = 'mask_box_detect'
    CATEGORY = '😺dzNodes/LayerMask'

    def is_rectangular_shape(self, mask_image, x, y, width, height, threshold):
        """Maskenin belirtilen bölgesinin dikdörtgensel olup olmadığını kontrol eder"""
        # Maske görüntüsünü numpy dizisine dönüştür
        mask_np = np.array(mask_image)
        
        # Maskenin belirtilen bölgesini al
        region = mask_np[y:y+height, x:x+width]
        
        if region.size == 0:  # Bölge boşsa
            return False
            
        # Bölgedeki beyaz piksellerin sayısını hesapla
        white_pixels = np.sum(region > 128)  # 128'den büyük değerler beyaz kabul edilir
        
        # Dikdörtgen alanı
        rect_area = width * height
        
        # Beyaz piksellerin dikdörtgen alanına oranı
        ratio = white_pixels / rect_area
        
        # Oran eşik değerinden büyükse, şekil dikdörtgenseldir
        return ratio > threshold

    def has_straight_edge(self, mask_image, threshold=0.85):
        """Maskenin düz kenarı olup olmadığını kontrol eder"""
        # Maske görüntüsünü numpy dizisine dönüştür
        mask_np = np.array(mask_image)
        
        # Görüntünün kenarlarını kontrol et
        # Üst kenar
        top_edge = mask_np[0, :]
        top_white = np.sum(top_edge > 128) / len(top_edge)
        
        # Alt kenar
        bottom_edge = mask_np[-1, :]
        bottom_white = np.sum(bottom_edge > 128) / len(bottom_edge)
        
        # Sol kenar
        left_edge = mask_np[:, 0]
        left_white = np.sum(left_edge > 128) / len(left_edge)
        
        # Sağ kenar
        right_edge = mask_np[:, -1]
        right_white = np.sum(right_edge > 128) / len(right_edge)
        
        # Herhangi bir kenar düz mü?
        return (top_white > threshold or bottom_white > threshold or 
                left_white > threshold or right_white > threshold)

    def draw_circles(self, image, x, y, width, height, circle_count, circle_size, overflow):
        draw = ImageDraw.Draw(image)
        max_radius = min(width, height) * circle_size / 2
        
        overflow_amount = max_radius * overflow
        
        import random
        for _ in range(circle_count):
            # Yuvarlak merkezi için rastgele konum belirle (dikdörtgen içinde)
            # Taşma miktarını hesaba katarak sınırları genişlet
            cx = random.randint(x + int(max_radius - overflow_amount), 
                               x + width - int(max_radius - overflow_amount))
            cy = random.randint(y + int(max_radius - overflow_amount), 
                               y + height - int(max_radius - overflow_amount))
            
            # Yuvarlak çiz
            draw.ellipse((cx - max_radius, cy - max_radius, cx + max_radius, cy + max_radius), 
                         fill="white", outline=None)
        
        return image

    def mask_box_detect(self, mask, detect, x_adjust, y_adjust, scale_adjust, circle_count, circle_size, overflow, straight_edge_threshold, remove_original=True):

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)

        if mask.shape[0] > 0:
            mask = torch.unsqueeze(mask[0], 0)

        _mask = mask2image(mask).convert('RGB')
        mask_gray = mask2image(mask).convert('L')

        _mask = gaussian_blur(_mask, 20).convert('L')
        x = 0
        y = 0
        width = 0
        height = 0

        if detect == "min_bounding_rect":
            (x, y, width, height) = min_bounding_rect(_mask)
        elif detect == "max_inscribed_rect":
            (x, y, width, height) = max_inscribed_rect(_mask)
        else:
            (x, y, width, height) = mask_area(_mask)
        log(f"{self.NODE_NAME}: Box detected. x={x},y={y},width={width},height={height}")
        _width = width
        _height = height
        if scale_adjust != 1.0:
            _width = int(width * scale_adjust)
            _height = int(height * scale_adjust)
            x = x - int((_width - width) / 2)
            y = y - int((_height - height) / 2)
        x += x_adjust
        y += y_adjust
        x_percent = (x + _width / 2) / _mask.width * 100
        y_percent = (y + _height / 2) / _mask.height * 100
        
        # Önizleme görüntüsünü oluştur
        preview_image = tensor2pil(mask).convert('RGB')
        
        # Şekil dikdörtgensel mi veya düz kenarı var mı kontrol et
        is_rectangular = self.is_rectangular_shape(mask_gray, x, y, _width, _height, straight_edge_threshold)
        has_straight = self.has_straight_edge(mask_gray, straight_edge_threshold)
        
        # Eğer şekil dikdörtgensel veya düz kenarlı ise işlem yap
        if is_rectangular or has_straight:
            log(f"{self.NODE_NAME}: Shape is rectangular or has straight edges. Processing.")
            
            # Orijinal beyaz alanı kaldır
            if remove_original:
                # Siyah bir görüntü oluştur
                black_image = Image.new('RGB', preview_image.size, (0, 0, 0))
                preview_image = black_image
            
            # Yuvarlakları çiz
            preview_image = self.draw_circles(preview_image, x, y, _width, _height, circle_count, circle_size, overflow)
        else:
            log(f"{self.NODE_NAME}: Shape is not rectangular and has no straight edges. Skipping processing.")
        
        log(f"{self.NODE_NAME} Processed.", message_type='finish')
        return (pil2tensor(preview_image), round(x_percent, 2), round(y_percent, 2), _width, _height, x, y,)

class MaskCreationFromBBox:
    def __init__(self):
        self.NODE_NAME = 'MaskCreationFromBBox'
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "x": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "canvas_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "canvas_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "brush_size": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1}),
                "brush_size_variation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_hardness": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "brush_density": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "😺dzNodes/LayerMask"

    def create_brush_stroke(self, draw, x, y, size, hardness):
        # Create gradient for soft brush
        for r in range(size):
            # Calculate alpha based on distance from center and hardness
            alpha = int(255 * (1 - (r / size)) ** (1 / hardness))
            if alpha <= 0:
                continue
            # Draw circle with white color (255 for grayscale)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def create_mask(self, width, height, x, y, canvas_width, canvas_height, brush_size, brush_size_variation, edge_hardness, brush_density=None):
        import random
        import math
        
        # Create a black image
        image = Image.new('L', (canvas_width, canvas_height), 0)
        draw = ImageDraw.Draw(image)
        
        # Calculate optimal brush density if not provided
        if brush_density is None:
            # Calculate based on area and brush size to ensure good coverage
            avg_brush_size = brush_size * (1 - brush_size_variation / 2)
            brush_area = math.pi * (avg_brush_size ** 2)
            target_area = width * height
            brush_density = int(max(50, (target_area / brush_area) * 1.5))  # 1.5x overlap for good coverage
        
        # Calculate number of brush strokes
        num_strokes = int((width * height / 10000) * brush_density)
        
        # Ensure minimum number of strokes for small areas
        num_strokes = max(num_strokes, int((width * height) / (brush_size * brush_size) * 2))
        
        # Create brush strokes
        for _ in range(num_strokes):
            # Random position within the bounding box
            stroke_x = random.randint(x, x + width)
            stroke_y = random.randint(y, y + height)
            
            # Random brush size with variation
            current_size = int(brush_size * (1 - brush_size_variation + random.random() * brush_size_variation * 2))
            
            # Create brush stroke
            self.create_brush_stroke(draw, stroke_x, stroke_y, current_size, edge_hardness)
        
        # Convert to tensor
        mask_tensor = pil2tensor(image)
        
        log(f"{self.NODE_NAME}: Created brush mask with {num_strokes} strokes", message_type='finish')
        return (mask_tensor,)


NODE_CLASS_MAPPINGS = {
    "LayerMask: MaskBoxDetect": MaskBoxDetect,
    "LayerMask: MaskCreationFromBBox": MaskCreationFromBBox
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: MaskBoxDetect": "LayerMask: MaskBoxDetect",
    "LayerMask: MaskCreationFromBBox": "LayerMask: MaskCreationFromBBox"
}
