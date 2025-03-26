'''
原始代码来自 https://github.com/StartHua/Comfyui_segformer_b2_clothes
'''
import torch
import os
import numpy as np
from PIL import Image, ImageEnhance
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn
import folder_paths
from .imagefunc import log, tensor2pil, pil2tensor, mask2image, image2mask, RGB2RGBA
from .imagefunc import guided_filter_alpha, mask_edge_detail, histogram_remap, generate_VITMatte, generate_VITMatte_trimap


class SegformerPipeline:
    def __init__(self):
        self.model_name = ''
        self.segment_label = []
        self.processor = None
        self.model = None

SegPipeline = SegformerPipeline()

# 切割服装
def get_segmentation(tensor_image, model_name='segformer_b2_clothes', device='cuda'):
    cloth = tensor2pil(tensor_image)
    model_folder_path = os.path.join(folder_paths.models_dir, model_name)
    try:
        model_folder_path = os.path.normpath(folder_paths.folder_names_and_paths[model_name][0][0])
    except:
        pass

    # Use pipeline's processor and model if available
    if SegPipeline.processor is not None and SegPipeline.model is not None:
        processor = SegPipeline.processor
        model = SegPipeline.model
    else:
        processor = SegformerImageProcessor.from_pretrained(model_folder_path)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_folder_path)
        model = model.to(device)
    
    # 预处理和预测
    inputs = processor(images=cloth, return_tensors="pt")
    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Move logits to CPU for further processing
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=cloth.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # Clear CUDA cache if using GPU
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    return pred_seg, cloth


class Segformer_B2_Clothes:

    def __init__(self):
        self.NODE_NAME = 'SegformerB2ClothesUltra'


    # Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt",
    # 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
    # 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"

    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda', 'cpu']
        return {"required":
            {
                "image": ("IMAGE",),
                "face": ("BOOLEAN", {"default": False}),
                "hair": ("BOOLEAN", {"default": False}),
                "hat": ("BOOLEAN", {"default": False}),
                "sunglass": ("BOOLEAN", {"default": False}),
                "left_arm": ("BOOLEAN", {"default": False}),
                "right_arm": ("BOOLEAN", {"default": False}),
                "left_leg": ("BOOLEAN", {"default": False}),
                "right_leg": ("BOOLEAN", {"default": False}),
                "upper_clothes": ("BOOLEAN", {"default": False}),
                "skirt": ("BOOLEAN", {"default": False}),
                "pants": ("BOOLEAN", {"default": False}),
                "dress": ("BOOLEAN", {"default": False}),
                "belt": ("BOOLEAN", {"default": False}),
                "shoe": ("BOOLEAN", {"default": False}),
                "bag": ("BOOLEAN", {"default": False}),
                "scarf": ("BOOLEAN", {"default": False}),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 12, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": (
                "FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": (
                "FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "segformer_ultra"
    CATEGORY = '😺dzNodes/LayerMask'

    def segformer_ultra(self, image,
                        face, hat, hair, sunglass, upper_clothes, skirt, pants, dress, belt, shoe,
                        left_leg, right_leg, left_arm, right_arm, bag, scarf, detail_method,
                        detail_erode, detail_dilate, black_point, white_point, process_detail, device, max_megapixels,
                        ):

        ret_images = []
        ret_masks = []

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        for i in image:
            pred_seg, cloth = get_segmentation(i)
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            orig_image = tensor2pil(i).convert('RGB')

            labels_to_keep = [0]
            if not hat:
                labels_to_keep.append(1)
            if not hair:
                labels_to_keep.append(2)
            if not sunglass:
                labels_to_keep.append(3)
            if not upper_clothes:
                labels_to_keep.append(4)
            if not skirt:
                labels_to_keep.append(5)
            if not pants:
                labels_to_keep.append(6)
            if not dress:
                labels_to_keep.append(7)
            if not belt:
                labels_to_keep.append(8)
            if not shoe:
                labels_to_keep.append(9)
                labels_to_keep.append(10)
            if not face:
                labels_to_keep.append(11)
            if not left_leg:
                labels_to_keep.append(12)
            if not right_leg:
                labels_to_keep.append(13)
            if not left_arm:
                labels_to_keep.append(14)
            if not right_arm:
                labels_to_keep.append(15)
            if not bag:
                labels_to_keep.append(16)
            if not scarf:
                labels_to_keep.append(17)

            mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)

            # 创建agnostic-mask图像
            mask_image = Image.fromarray((1 - mask) * 255)
            mask_image = mask_image.convert("L")
            _mask = pil2tensor(mask_image)

            detail_range = detail_erode + detail_dilate
            if process_detail:
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device,
                                              max_megapixels=max_megapixels)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = mask2image(_mask)

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

class SegformerClothesPipelineLoader:

    def __init__(self):
        self.NODE_NAME = 'SegformerClothesPipelineLoader'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_list = ['segformer_b3_clothes', 'segformer_b2_clothes']
        device_list = ['cuda', 'cpu']
        return {"required":
            {   "model": (model_list,),
                "device": (device_list,),
            }
        }

    RETURN_TYPES = ("SegPipeline",)
    RETURN_NAMES = ("segformer_pipeline",)
    FUNCTION = "segformer_clothes_pipeline_loader"
    CATEGORY = '😺dzNodes/LayerMask'

    def segformer_clothes_pipeline_loader(self, model, device):
        pipeline = SegformerPipeline()
        pipeline.model_name = model
        pipeline.segment_label = [0]  # Default to only background
        
        # Get model path
        model_folder_path = os.path.join(folder_paths.models_dir, model)
        try:
            model_folder_path = os.path.normpath(folder_paths.folder_names_and_paths[model][0][0])
        except:
            pass
            
        # Create processor and model
        pipeline.processor = SegformerImageProcessor.from_pretrained(model_folder_path)
        pipeline.model = AutoModelForSemanticSegmentation.from_pretrained(model_folder_path)
        pipeline.model = pipeline.model.to(device)
        
        return (pipeline,)

class SegformerFashionPipelineLoader:

    def __init__(self):
        self.NODE_NAME = 'SegformerFashionPipelineLoader'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        model_list = ['segformer_b3_fashion']
        return {"required":
            {   "model": (model_list,),
                "shirt": ("BOOLEAN", {"default": False, "label_on": "enabled(衬衫、罩衫)", "label_off": "disabled(衬衫、罩衫)"}),
                "top": ("BOOLEAN", {"default": False, "label_on": "enabled(上衣、t恤)", "label_off": "disabled(上衣、t恤)"}),
                "sweater": ("BOOLEAN", {"default": False, "label_on": "enabled(毛衣)", "label_off": "disabled(毛衣)"}),
                "cardigan": ("BOOLEAN", {"default": False, "label_on": "enabled(开襟毛衫)", "label_off": "disabled(开襟毛衫)"}),
                "jacket": ("BOOLEAN", {"default": False, "label_on": "enabled(夹克)", "label_off": "disabled(夹克)"}),
                "vest": ("BOOLEAN", {"default": False, "label_on": "enabled(背心)", "label_off": "disabled(背心)"}),
                "pants": ("BOOLEAN", {"default": False, "label_on": "enabled(裤子)", "label_off": "disabled(裤子)"}),
                "shorts": ("BOOLEAN", {"default": False, "label_on": "enabled(短裤)", "label_off": "disabled(短裤)"}),
                "skirt": ("BOOLEAN", {"default": False, "label_on": "enabled(裙子)", "label_off": "disabled(裙子)"}),
                "coat": ("BOOLEAN", {"default": False, "label_on": "enabled(外套)", "label_off": "disabled(外套)"}),
                "dress": ("BOOLEAN", {"default": False, "label_on": "enabled(连衣裙)", "label_off": "disabled(连衣裙)"}),
                "jumpsuit": ("BOOLEAN", {"default": False, "label_on": "enabled(连身裤)", "label_off": "disabled(连身裤)"}),
                "cape": ("BOOLEAN", {"default": False, "label_on": "enabled(斗篷)", "label_off": "disabled(斗篷)"}),
                "glasses": ("BOOLEAN", {"default": False, "label_on": "enabled(眼镜)", "label_off": "disabled(眼镜)"}),
                "hat": ("BOOLEAN", {"default": False, "label_on": "enabled(帽子)", "label_off": "disabled(帽子)"}),
                "hairaccessory": ("BOOLEAN", {"default": False, "label_on": "enabled(头带)", "label_off": "disabled(头带)"}),
                "tie": ("BOOLEAN", {"default": False, "label_on": "enabled(领带)", "label_off": "disabled(领带)"}),
                "glove": ("BOOLEAN", {"default": False, "label_on": "enabled(手套)", "label_off": "disabled(手套)"}),
                "watch": ("BOOLEAN", {"default": False, "label_on": "enabled(手表)", "label_off": "disabled(手表)"}),
                "belt": ("BOOLEAN", {"default": False, "label_on": "enabled(皮带)", "label_off": "disabled(皮带)"}),
                "legwarmer": ("BOOLEAN", {"default": False, "label_on": "enabled(腿套)", "label_off": "disabled(腿套)"}),
                "tights": ("BOOLEAN", {"default": False, "label_on": "enabled(裤袜)","label_off": "disabled(裤袜)"}),
                "sock": ("BOOLEAN", {"default": False, "label_on": "enabled(袜子)", "label_off": "disabled(袜子)"}),
                "shoe": ("BOOLEAN", {"default": False, "label_on": "enabled(鞋子)", "label_off": "disabled(鞋子)"}),
                "bagwallet": ("BOOLEAN", {"default": False, "label_on": "enabled(手包)", "label_off": "disabled(手包)"}),
                "scarf": ("BOOLEAN", {"default": False, "label_on": "enabled(围巾)", "label_off": "disabled(围巾)"}),
                "umbrella": ("BOOLEAN", {"default": False, "label_on": "enabled(雨伞)", "label_off": "disabled(雨伞)"}),
                "hood": ("BOOLEAN", {"default": False, "label_on": "enabled(兜帽)", "label_off": "disabled(兜帽)"}),
                "collar": ("BOOLEAN", {"default": False, "label_on": "enabled(衣领)", "label_off": "disabled(衣领)"}),
                "lapel": ("BOOLEAN", {"default": False, "label_on": "enabled(翻领)", "label_off": "disabled(翻领)"}),
                "epaulette": ("BOOLEAN", {"default": False, "label_on": "enabled(肩章)", "label_off": "disabled(肩章)"}),
                "sleeve": ("BOOLEAN", {"default": False, "label_on": "enabled(袖子)", "label_off": "disabled(袖子)"}),
                "pocket": ("BOOLEAN", {"default": False, "label_on": "enabled(口袋)", "label_off": "disabled(口袋)"}),
                "neckline": ("BOOLEAN", {"default": False, "label_on": "enabled(领口)", "label_off": "disabled(领口)"}),
                "buckle": ("BOOLEAN", {"default": False, "label_on": "enabled(带扣)", "label_off": "disabled(带扣)"}),
                "zipper": ("BOOLEAN", {"default": False, "label_on": "enabled(拉链)", "label_off": "disabled(拉链)"}),
                "applique": ("BOOLEAN", {"default": False, "label_on": "enabled(贴花)", "label_off": "disabled(贴花)"}),
                "bead": ("BOOLEAN", {"default": False, "label_on": "enabled(珠子)", "label_off": "disabled(珠子)"}),
                "bow": ("BOOLEAN", {"default": False, "label_on": "enabled(蝴蝶结)", "label_off": "disabled(蝴蝶结)"}),
                "flower": ("BOOLEAN", {"default": False, "label_on": "enabled(花)", "label_off": "disabled(花)"}),
                "fringe": ("BOOLEAN", {"default": False, "label_on": "enabled(刘海)", "label_off": "disabled(刘海)"}),
                "ribbon": ("BOOLEAN", {"default": False, "label_on": "enabled(丝带)", "label_off": "disabled(丝带)"}),
                "rivet": ("BOOLEAN", {"default": False, "label_on": "enabled(铆钉)", "label_off": "disabled(铆钉)"}),
                "ruffle": ("BOOLEAN", {"default": False, "label_on": "enabled(褶饰)", "label_off": "disabled(褶饰)"}),
                "sequin": ("BOOLEAN", {"default": False, "label_on": "enabled(亮片)", "label_off": "disabled(亮片)"}),
                "tassel": ("BOOLEAN", {"default": False, "label_on": "enabled(流苏)", "label_off": "disabled(流苏)"}),
            }
        }

    RETURN_TYPES = ("SegPipeline",)
    RETURN_NAMES = ("segformer_pipeline",)
    FUNCTION = "segformer_fashion_pipeline_loader"
    CATEGORY = '😺dzNodes/LayerMask'

    def segformer_fashion_pipeline_loader(self, model,
                                          shirt, top, sweater, cardigan, jacket, vest, pants,
                                          shorts, skirt, coat, dress, jumpsuit, cape, glasses,
                                          hat, hairaccessory, tie, glove, watch, belt, legwarmer,
                                          tights, sock, shoe, bagwallet, scarf, umbrella, hood,
                                          collar, lapel, epaulette, sleeve, pocket, neckline,
                                          buckle, zipper, applique, bead, bow, flower, fringe,
                                          ribbon, rivet, ruffle, sequin, tassel
                                        ):

        pipeline = SegformerPipeline()
        labels_to_keep = [0]
        if not shirt:
            labels_to_keep.append(1)
        if not top:
            labels_to_keep.append(2)
        if not sweater:
            labels_to_keep.append(3)
        if not cardigan:
            labels_to_keep.append(4)
        if not jacket:
            labels_to_keep.append(5)
        if not vest:
            labels_to_keep.append(6)
        if not pants:
            labels_to_keep.append(7)
        if not shorts:
            labels_to_keep.append(8)
        if not skirt:
            labels_to_keep.append(9)
        if not coat:
            labels_to_keep.append(10)
        if not dress:
            labels_to_keep.append(11)
        if not jumpsuit:
            labels_to_keep.append(12)
        if not cape:
            labels_to_keep.append(13)
        if not glasses:
            labels_to_keep.append(14)
        if not hat:
            labels_to_keep.append(15)
        if not hairaccessory:
            labels_to_keep.append(16)
        if not tie:
            labels_to_keep.append(17)
        if not glove:
            labels_to_keep.append(18)
        if not watch:
            labels_to_keep.append(19)
        if not belt:
            labels_to_keep.append(20)
        if not legwarmer:
            labels_to_keep.append(21)
        if not tights:
            labels_to_keep.append(22)
        if not sock:
            labels_to_keep.append(23)
        if not shoe:
            labels_to_keep.append(24)
        if not bagwallet:
            labels_to_keep.append(25)
        if not scarf:
            labels_to_keep.append(26)
        if not umbrella:
            labels_to_keep.append(27)
        if not hood:
            labels_to_keep.append(28)
        if not collar:
            labels_to_keep.append(29)
        if not lapel:
            labels_to_keep.append(30)
        if not epaulette:
            labels_to_keep.append(31)
        if not sleeve:
            labels_to_keep.append(32)
        if not pocket:
            labels_to_keep.append(33)
        if not neckline:
            labels_to_keep.append(34)
        if not buckle:
            labels_to_keep.append(35)
        if not zipper:
            labels_to_keep.append(36)
        if not applique:
            labels_to_keep.append(37)
        if not bead:
            labels_to_keep.append(38)
        if not bow:
            labels_to_keep.append(39)
        if not flower:
            labels_to_keep.append(40)
        if not fringe:
            labels_to_keep.append(41)
        if not ribbon:
            labels_to_keep.append(42)
        if not rivet:
            labels_to_keep.append(43)
        if not ruffle:
            labels_to_keep.append(44)
        if not sequin:
            labels_to_keep.append(45)
        if not tassel:
            labels_to_keep.append(46)

        pipeline.segment_label = labels_to_keep
        pipeline.model_name = model
        return (pipeline,)

class SegformerUltraV2:

    def __init__(self):
        self.NODE_NAME = 'SegformerUltraV2'
        pass

    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda', 'cpu']
        return {
            "required": {
                "image": ("IMAGE",),
                "segformer_pipeline": ("SegPipeline",),
                "face": ("BOOLEAN", {"default": False, "label_on": "enabled(脸)", "label_off": "disabled(脸)"}),
                "hair": ("BOOLEAN", {"default": False, "label_on": "enabled(头发)", "label_off": "disabled(头发)"}),
                "hat": ("BOOLEAN", {"default": False, "label_on": "enabled(帽子)", "label_off": "disabled(帽子)"}),
                "sunglass": ("BOOLEAN", {"default": False, "label_on": "enabled(墨镜)", "label_off": "disabled(墨镜)"}),
                "left_arm": ("BOOLEAN", {"default": False, "label_on": "enabled(左臂)", "label_off": "disabled(左臂)"}),
                "right_arm": ("BOOLEAN", {"default": False, "label_on": "enabled(右臂)", "label_off": "disabled(右臂)"}),
                "left_leg": ("BOOLEAN", {"default": False, "label_on": "enabled(左腿)", "label_off": "disabled(左腿)"}),
                "right_leg": ("BOOLEAN", {"default": False, "label_on": "enabled(右腿)", "label_off": "disabled(右腿)"}),
                "left_shoe": ("BOOLEAN", {"default": False, "label_on": "enabled(左鞋)", "label_off": "disabled(左鞋)"}),
                "right_shoe": ("BOOLEAN", {"default": False, "label_on": "enabled(右鞋)", "label_off": "disabled(右鞋)"}),
                "upper_clothes": ("BOOLEAN", {"default": False, "label_on": "enabled(上衣)", "label_off": "disabled(上衣)"}),
                "skirt": ("BOOLEAN", {"default": False, "label_on": "enabled(短裙)", "label_off": "disabled(短裙)"}),
                "pants": ("BOOLEAN", {"default": False, "label_on": "enabled(裤子)", "label_off": "disabled(裤子)"}),
                "dress": ("BOOLEAN", {"default": False, "label_on": "enabled(连衣裙)", "label_off": "disabled(连衣裙)"}),
                "belt": ("BOOLEAN", {"default": False, "label_on": "enabled(腰带)", "label_off": "disabled(腰带)"}),
                "bag": ("BOOLEAN", {"default": False, "label_on": "enabled(背包)", "label_off": "disabled(背包)"}),
                "scarf": ("BOOLEAN", {"default": False, "label_on": "enabled(围巾)", "label_off": "disabled(围巾)"}),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 8, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {
                "bodySegmentationPart": ("STRING", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "segformer_ultra_v2"
    CATEGORY = '😺dzNodes/LayerMask'

    def segformer_ultra_v2(self, image, segformer_pipeline,
                        face, hat, hair, sunglass,
                        left_leg, right_leg, left_arm, right_arm, left_shoe, right_shoe,
                        upper_clothes, skirt, pants, dress, belt, bag, scarf,
                        detail_method, detail_erode, detail_dilate, black_point, white_point,
                        process_detail, device, max_megapixels,
                        bodySegmentationPart=None):
        model = segformer_pipeline.model_name
        
        # Handle bodySegmentationPart if provided
        if bodySegmentationPart is not None:
            if bodySegmentationPart == "One-Pieces" or bodySegmentationPart == "Both":
                labels_to_keep = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17]  # Include background
            elif bodySegmentationPart == "Tops":
                labels_to_keep = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 16, 17]  # Include background
            elif bodySegmentationPart == "Bottoms":
                labels_to_keep = [0, 1, 2, 3, 4, 8, 9, 10, 11, 14, 15, 16, 17]  # Include background
            else:
                # If invalid value, fall back to default behavior
                labels_to_keep = [0]  # Start with background
                if not hat:
                    labels_to_keep.append(1)
                if not hair:
                    labels_to_keep.append(2)
                if not sunglass:
                    labels_to_keep.append(3)
                if not upper_clothes:
                    labels_to_keep.append(4)
                if not skirt:
                    labels_to_keep.append(5)
                if not pants:
                    labels_to_keep.append(6)
                if not dress:
                    labels_to_keep.append(7)
                if not belt:
                    labels_to_keep.append(8)
                if not left_shoe:
                    labels_to_keep.append(9)
                if not right_shoe:
                    labels_to_keep.append(10)
                if not face:
                    labels_to_keep.append(11)
                if not left_leg:
                    labels_to_keep.append(12)
                if not right_leg:
                    labels_to_keep.append(13)
                if not left_arm:
                    labels_to_keep.append(14)
                if not right_arm:
                    labels_to_keep.append(15)
                if not bag:
                    labels_to_keep.append(16)
                if not scarf:
                    labels_to_keep.append(17)
        else:
            # Default behavior when bodySegmentationPart is not provided
            labels_to_keep = [0]  # Start with background
            if not hat:
                labels_to_keep.append(1)
            if not hair:
                labels_to_keep.append(2)
            if not sunglass:
                labels_to_keep.append(3)
            if not upper_clothes:
                labels_to_keep.append(4)
            if not skirt:
                labels_to_keep.append(5)
            if not pants:
                labels_to_keep.append(6)
            if not dress:
                labels_to_keep.append(7)
            if not belt:
                labels_to_keep.append(8)
            if not left_shoe:
                labels_to_keep.append(9)
            if not right_shoe:
                labels_to_keep.append(10)
            if not face:
                labels_to_keep.append(11)
            if not left_leg:
                labels_to_keep.append(12)
            if not right_leg:
                labels_to_keep.append(13)
            if not left_arm:
                labels_to_keep.append(14)
            if not right_arm:
                labels_to_keep.append(15)
            if not bag:
                labels_to_keep.append(16)
            if not scarf:
                labels_to_keep.append(17)

        ret_images = []
        ret_masks = []
        print("LABELS_TO_KEEP")
        print(labels_to_keep)
        print(bodySegmentationPart)

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        for i in image:
            pred_seg, cloth = get_segmentation(i, model_name=model)
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            orig_image = tensor2pil(i).convert('RGB')

            mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)

            # 创建agnostic-mask图像
            mask_image = Image.fromarray((1 - mask) * 255)
            mask_image = mask_image.convert("L")
            brightness_image = ImageEnhance.Brightness(mask_image)
            mask_image = brightness_image.enhance(factor=1.08)
            _mask = pil2tensor(mask_image)

            detail_range = detail_erode + detail_dilate
            if process_detail:
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device,
                                              max_megapixels=max_megapixels)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = mask2image(_mask)

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{self.NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

class LisaReduxImageCreate:
    def __init__(self):
        self.NODE_NAME = 'LisaReduxImageCreate'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "new_left_width", "new_left_height")
    FUNCTION = "create_side_by_side"
    CATEGORY = '😺dzNodes/LayerMask'

    def create_side_by_side(self, left_image, right_image):
        # Convert tensors to PIL images
        left_pil = tensor2pil(left_image)
        right_pil = tensor2pil(right_image)

        # Get original dimensions
        left_width, left_height = left_pil.size
        right_width, right_height = right_pil.size

        # Check if right image is landscape
        is_right_landscape = right_width > right_height

        if is_right_landscape:
            # For landscape right image, stack vertically
            new_width = right_width
            new_height = right_height * 2
            new_image = Image.new('RGB', (new_width, new_height))

            # Calculate new dimensions for left image to fit width while maintaining aspect ratio
            left_aspect_ratio = left_width / left_height
            new_left_width = right_width
            new_left_height = int(new_left_width / left_aspect_ratio)

            # Resize left image to fit width
            left_pil_resized = left_pil.resize((new_left_width, new_left_height), Image.Resampling.LANCZOS)

            # Calculate y offset to center vertically if needed
            if new_left_height < right_height:
                y_offset = (right_height - new_left_height) // 2
            else:
                # If left image is too tall, resize to fit height
                new_left_height = right_height
                new_left_width = int(new_left_height * left_aspect_ratio)
                left_pil_resized = left_pil.resize((new_left_width, new_left_height), Image.Resampling.LANCZOS)
                # Center horizontally
                y_offset = 0
                x_offset = (right_width - new_left_width) // 2
                
            # Paste images - right image at bottom, left image at top
            new_image.paste(right_pil, (0, right_height))
            new_image.paste(left_pil_resized, (x_offset if new_left_height == right_height else 0, y_offset))
        else:
            # For non-landscape right image, stack horizontally
            new_width = right_width * 2
            new_height = right_height
            new_image = Image.new('RGB', (new_width, new_height))

            # Calculate new dimensions for left image to fit height while maintaining aspect ratio
            left_aspect_ratio = left_width / left_height
            new_left_height = right_height
            new_left_width = int(new_left_height * left_aspect_ratio)

            # If left image would be too wide, scale it down to fit half width
            if new_left_width > right_width:
                new_left_width = right_width
                new_left_height = int(new_left_width / left_aspect_ratio)

            # Resize left image
            left_pil_resized = left_pil.resize((new_left_width, new_left_height), Image.Resampling.LANCZOS)

            # Calculate y offset to center vertically if needed
            y_offset = (right_height - new_left_height) // 2

            # Paste images side by side
            new_image.paste(right_pil, (right_width, 0))
            new_image.paste(left_pil_resized, (0, y_offset))

        # Convert back to tensor and return with dimensions
        return (pil2tensor(new_image), new_left_width, new_left_height,)

class LisaCalculateFluxAspectRatio:
    def __init__(self):
        self.NODE_NAME = 'LisaCalculateFluxAspectRatio'
        # Portrait ratios (width:height, where height > width)
        self.PORTRAIT_RATIOS = {
            "2:3 (Classic Portrait)": (2, 3),
            "3:4 (Golden Ratio)": (3, 4),
            "3:5 (Elegant Vertical)": (3, 5),
            "4:5 (Artistic Frame)": (4, 5),
            "5:7 (Balanced Portrait)": (5, 7),
            "5:8 (Tall Portrait)": (5, 8),
            "7:9 (Modern Portrait)": (7, 9),
            # "9:16 (Slim Vertical)": (9, 16),
            # "9:19 (Tall Slim)": (9, 19),
            # "9:21 (Ultra Tall)": (9, 21),
        }
        # Landscape ratios (width:height, where width > height)
        self.LANDSCAPE_RATIOS = {
            "3:2 (Golden Landscape)": (3, 2),
            "4:3 (Classic Landscape)": (4, 3),
            "5:3 (Wide Horizon)": (5, 3),
            "5:4 (Balanced Frame)": (5, 4),
            "7:5 (Elegant Landscape)": (7, 5),
            "8:5 (Cinematic View)": (8, 5),
            "9:7 (Artful Horizon)": (9, 7),
            # "16:9 (Panorama)": (16, 9),
            # "19:9 (Cinematic Ultrawide)": (19, 9),
            # "21:9 (Epic Ultrawide)": (21, 9),
        }
        # Square ratio is available for both orientations
        self.SQUARE_RATIO = {
            "1:1 (Perfect Square)": (1, 1)
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("aspect_ratio_name", "aspect_ratio", "width", "height")
    FUNCTION = "calculate_aspect_ratio"
    CATEGORY = '😺dzNodes/LayerMask'

    def round_to_64(self, value):
        # Round to nearest multiple of 64
        return ((value + 32) // 64) * 64

    def calculate_aspect_ratio(self, image):
        # Convert tensor to PIL image to get dimensions
        pil_image = tensor2pil(image)
        width, height = pil_image.size
        
        # Check if image is portrait or landscape
        is_portrait = height > width
        long_edge = max(width, height)
        short_edge = min(width, height)
        
        # Calculate actual aspect ratio (always as long:short)
        actual_ratio = long_edge / short_edge
        
        # Initialize variables for finding closest match
        closest_ratio_name = None
        closest_difference = float('inf')
        
        # Select the appropriate aspect ratios to check based on orientation
        if is_portrait:
            ratios_to_check = self.PORTRAIT_RATIOS
            # Only add square if the ratio is very close to 1:1
            if 0.95 <= (height / width) <= 1.05:
                ratios_to_check = {**ratios_to_check, **self.SQUARE_RATIO}
        else:
            ratios_to_check = self.LANDSCAPE_RATIOS
            # Only add square if the ratio is very close to 1:1
            if 0.95 <= (width / height) <= 1.05:
                ratios_to_check = {**ratios_to_check, **self.SQUARE_RATIO}
        
        # Compare with appropriate ratios
        for ratio_name, (w, h) in ratios_to_check.items():
            if is_portrait:
                compare_ratio = h / w  # For portrait, use height/width
            else:
                compare_ratio = w / h  # For landscape, use width/height
            
            # Calculate difference
            difference = abs((height/width if is_portrait else width/height) - compare_ratio)
            
            # Update if this is the closest match so far
            if difference < closest_difference:
                closest_difference = difference
                closest_ratio_name = ratio_name
        
        # Get the winning ratio values from the appropriate dictionary
        if closest_ratio_name in self.PORTRAIT_RATIOS:
            winning_w, winning_h = self.PORTRAIT_RATIOS[closest_ratio_name]
        elif closest_ratio_name in self.LANDSCAPE_RATIOS:
            winning_w, winning_h = self.LANDSCAPE_RATIOS[closest_ratio_name]
        else:
            winning_w, winning_h = self.SQUARE_RATIO[closest_ratio_name]
        
        # Calculate aspect ratio float
        if is_portrait:
            aspect_ratio_float = winning_h / winning_w
        else:
            aspect_ratio_float = winning_w / winning_h

        # Keep the long edge and calculate the other dimension
        if is_portrait:
            target_height = long_edge
            target_width = int(target_height / aspect_ratio_float)
        else:
            target_width = long_edge
            target_height = int(target_width / aspect_ratio_float)

        # Scale down if exceeding 1536
        if target_width > 1536 or target_height > 1536:
            scale = 1536 / max(target_width, target_height)
            target_width = int(target_width * scale)
            target_height = int(target_height * scale)

        # Round both dimensions to nearest multiple of 64
        target_width = self.round_to_64(target_width)
        target_height = self.round_to_64(target_height)

        # Final check to ensure neither dimension exceeds 1536 after rounding
        if target_width > 1536 or target_height > 1536:
            scale = 1536 / max(target_width, target_height)
            target_width = self.round_to_64(int(target_width * scale))
            target_height = self.round_to_64(int(target_height * scale))

        return (closest_ratio_name, float(aspect_ratio_float), target_width, target_height)

class LisaPngToJpegNode:
    def __init__(self):
        self.NODE_NAME = 'LisaPngToJpegNode'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert_to_jpeg"
    CATEGORY = '😺dzNodes/LayerMask'

    def convert_to_jpeg(self, image):
        # Convert tensor to PIL image
        pil_image = tensor2pil(image)
        
        # Create a new image with black background
        background = Image.new('RGB', pil_image.size, (0, 0, 0))
        
        # Paste the original image onto the black background using alpha channel
        if pil_image.mode == 'RGBA':
            background.paste(pil_image, mask=pil_image.split()[3])
        else:
            background.paste(pil_image)
        
        # Convert back to tensor and return
        return (pil2tensor(background),)

NODE_CLASS_MAPPINGS = {
    "LayerMask: SegformerB2ClothesUltra": Segformer_B2_Clothes,
    "LayerMask: SegformerUltraV2": SegformerUltraV2,
    "LayerMask: SegformerClothesPipelineLoader": SegformerClothesPipelineLoader,
    "LayerMask: SegformerFashionPipelineLoader": SegformerFashionPipelineLoader,
    "LayerMask: LisaReduxImageCreate": LisaReduxImageCreate,
    "LayerMask: LisaCalculateFluxAspectRatio": LisaCalculateFluxAspectRatio,
    "LayerMask: LisaPngToJpegNode": LisaPngToJpegNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerMask: SegformerB2ClothesUltra": "LayerMask: Segformer B2 Clothes Ultra",
    "LayerMask: SegformerUltraV2": "LayerMask: Segformer Ultra V2",
    "LayerMask: SegformerClothesPipelineLoader": "LayerMask: Segformer Clothes Pipeline",
    "LayerMask: SegformerFashionPipelineLoader": "LayerMask: Segformer Fashion Pipeline",
    "LayerMask: LisaReduxImageCreate": "LayerMask: Lisa Redux Image Create",
    "LayerMask: LisaCalculateFluxAspectRatio": "LayerMask: Lisa Calculate Flux Aspect Ratio",
    "LayerMask: LisaPngToJpegNode": "LayerMask: Lisa Png to Jpeg Node",
}

