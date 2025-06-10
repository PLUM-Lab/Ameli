import argparse
import os
import sys

import sys
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm 
import bestbuy.src.object_detect.groundingdino.datasets.transforms as T
from bestbuy.src.object_detect.groundingdino.models import build_model
from bestbuy.src.object_detect.groundingdino.util import box_ops
from bestbuy.src.object_detect.groundingdino.util.slconfig import SLConfig
from bestbuy.src.object_detect.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from random import sample


def load_model(model_config_path, model_checkpoint_path,device, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = device#"cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.to(device)
    print(load_res)
    _ = model.eval()
    return model


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image




def get_grounding_output(model, image, caption, box_threshold, text_threshold,device, with_logits=True, cpu_only=False):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    # device = "cuda" if not cpu_only else "cpu"
    # model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (batch_size,nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    logit_filt_score=logits_filt.max(dim=1)[0]
    if logit_filt_score.max()>  box_threshold:
        position=logit_filt_score.argmax()
        logits_filt = logits_filt[position:position+1]  # num_filt, 256
        boxes_filt = boxes_filt[position:position+1]  # num_filt, 4
    else:
        logits_filt=[]
        boxes_filt=[]
    # max_idx=torch.argmax
    
    # logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def save_boxes_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"
    image_pil_array=np.asarray(image_pil)
    

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # x, y, w, h =box
        # x, y, w, h =int(x), int(y), int(w), int(h )
        
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
      
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        imgRectArray = image_pil_array[y0:y1,x0:x1]
        try:
            imgRect=Image.fromarray(imgRectArray)
        except Exception as e:
            imgRect=None
   
       
    return image_pil, imgRect

class ObjectDetector:
    def __init__(self,config_file, checkpoint_path,cpu_only,box_threshold, text_threshold ,device) -> None:
        self.model = load_model(config_file, checkpoint_path, device,cpu_only= cpu_only)
        self.cpu_only=cpu_only
        self.box_threshold, self.text_threshold=box_threshold, text_threshold
        self.device=device
        
    def detecte_image(self,image_path, text_prompt):
        # load image
        image_pil, image = load_image(image_path)

        # visualize raw image
        # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

        # run model
        boxes_filt, pred_phrases = get_grounding_output(
            self.model, image, text_prompt, self.box_threshold, self.text_threshold,self.device, self.cpu_only 
        )
        is_detected=True  if len(boxes_filt)>0 else False
        # visualize pred
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        if is_detected:
             
            _,detected_image = save_boxes_image(image_pil, pred_dict)#plot_boxes_to_image
            if detected_image is None:
                is_detected=False
        else:
            detected_image=None
        return is_detected,detected_image