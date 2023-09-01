import numpy as np
from PIL import Image
import cv2
import torch
import os
from random import randint
import matplotlib.pyplot as plt

def convert_box_xywh_to_xyxy(box):
    if len(box) == 4:
        return [box[0], box[1], box[0] + box[2], box[1] + box[3]]
    else:
        result = []
        for b in box:
            b = convert_box_xywh_to_xyxy(b)
            result.append(b)               
    return result

def convert_box_cxcywh_to_xyxy(box):
    if len(box) == 4:
        return [box[0] - int(box[2]/2), box[1] - int(box[3]/2), box[0] + int(box[2]/2), box[1] + int(box[3]/2)]
    else:
        result = []
        for b in box:
            b = convert_box_xywh_to_xyxy(b)
            result.append(b)               
    return result

def segment_image(image, bbox):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    x1, y1, x2, y2 = bbox
    segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (255, 255, 255))
    # transparency_mask = np.zeros_like((), dtype=np.uint8)
    transparency_mask = np.zeros(
        (image_array.shape[0], image_array.shape[1]), dtype=np.uint8
    )
    transparency_mask[y1:y2, x1:x2] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


def format_results(result, filter=0):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        if torch.sum(mask) < filter:
            continue
        annotation["id"] = i
        annotation["segmentation"] = mask.cpu().numpy()
        annotation["bbox"] = result.boxes.data[i]
        annotation["score"] = result.boxes.conf[i]
        annotation["area"] = annotation["segmentation"].sum()
        annotations.append(annotation)
    return annotations


def filter_masks(annotations):  # filter the overlap mask
    annotations.sort(key=lambda x: x["area"], reverse=True)
    to_remove = set()
    for i in range(0, len(annotations)):
        a = annotations[i]
        for j in range(i + 1, len(annotations)):
            b = annotations[j]
            if i != j and j not in to_remove:
                # check if
                if b["area"] < a["area"]:
                    if (a["segmentation"] & b["segmentation"]).sum() / b[
                        "segmentation"
                    ].sum() > 0.8:
                        to_remove.add(j)

    return [a for i, a in enumerate(annotations) if i not in to_remove], to_remove


def get_bbox_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x1, y1, w, h = cv2.boundingRect(contours[0])
    x2, y2 = x1 + w, y1 + h
    if len(contours) > 1:
        for b in contours:
            x_t, y_t, w_t, h_t = cv2.boundingRect(b)
            # 将多个bbox合并成一个
            x1 = min(x1, x_t)
            y1 = min(y1, y_t)
            x2 = max(x2, x_t + w_t)
            y2 = max(y2, y_t + h_t)
        h = y2 - y1
        w = x2 - x1
    return [x1, y1, x2, y2]

def get_rotated_bbox_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    minRect = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)
    if len(contours) > 0:
        epsilon = 0.008*cv2.arcLength(contours[0],True)
        c = cv2.approxPolyDP(contours[0],epsilon,True)
        return minRect[0], c
    else:
        print("NO CONTOURS")
        return None, None

def crop_image(annotations, image_like):
    if isinstance(image_like, str):
        image = Image.open(image_like)
    else:
        image = image_like
    ori_w, ori_h = image.size
    mask_h, mask_w = annotations[0]["segmentation"].shape
    if ori_w != mask_w or ori_h != mask_h:
        image = image.resize((mask_w, mask_h))
    cropped_boxes = []
    cropped_images = []
    not_crop = []
    origin_id = []
    for _, mask in enumerate(annotations):
        if np.sum(mask["segmentation"]) <= 100:
            continue
        origin_id.append(_)
        bbox = get_bbox_from_mask(mask["segmentation"])  # mask 的 bbox
        cropped_boxes.append(segment_image(image, bbox))  # 保存裁剪的图片
        # cropped_boxes.append(segment_image(image,mask["segmentation"]))
        cropped_images.append(bbox)  # 保存裁剪的图片的bbox
    return cropped_boxes, cropped_images, not_crop, origin_id, annotations

def bbox_mask2ori(bbox, mask_h, mask_w, ori_h, ori_w):
    #bbox format x1y1x2y2

    max_size = max(ori_h, ori_w)
    diff = ori_w - ori_h
    if (diff > 0):
        bbox[1] = bbox[1] + int(diff/2)
        bbox[3] = bbox[3] + int(diff/2)
    else:
        bbox[0] = bbox[0] + int(-diff/2)
        bbox[2] = bbox[2] + int(-diff/2)

    if mask_h != ori_h or mask_w != ori_w:
        bbox = [
            int(bbox[0] * mask_w / max_size),
            int(bbox[1] * mask_h / max_size),
            int(bbox[2] * mask_w / max_size),
            int(bbox[3] * mask_h / max_size),
        ]
    nbbox = []*4
    nbbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
    nbbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
    nbbox[2] = round(bbox[2]) if round(bbox[2]) < mask_w else mask_w
    nbbox[3] = round(bbox[3]) if round(bbox[3]) < mask_h else mask_h

    return nbbox

def bbox_ori2mask(bbox, mask_h, mask_w, ori_h, ori_w):
    #bbox format x1y1x2y2

    max_size = max(ori_h, ori_w)
    diff = ori_w - ori_h
    if (diff > 0):
        bbox[1] = bbox[1] - int(diff/2)
        bbox[3] = bbox[3] - int(diff/2)
    else:
        bbox[0] = bbox[0] - int(-diff/2)
        bbox[2] = bbox[2] - int(-diff/2)

    if mask_h != ori_h or mask_w != ori_w:
        bbox = [
            int(bbox[0] * max_size / mask_w),
            int(bbox[1] * max_size / mask_h),
            int(bbox[2] * max_size / mask_w),
            int(bbox[3] * max_size / mask_h),
        ]
    nbbox = []*4
    nbbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
    nbbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
    nbbox[2] = round(bbox[2]) if round(bbox[2]) < ori_w else ori_w
    nbbox[3] = round(bbox[3]) if round(bbox[3]) < ori_h else ori_h

    return nbbox

def box_prompt(masks, bbox, target_height, target_width):
    h = masks.shape[1]
    w = masks.shape[2]

    max_size = max(target_height, target_width)
    half_max = int(max_size/2)
    half_w = int(target_width/2)
    half_h = int(target_height/2)


    diff = target_width - target_height
    if (diff > 0):
        bbox[1] = bbox[1] + int(diff/2)
        bbox[3] = bbox[3] + int(diff/2)
    else:
        bbox[0] = bbox[0] + int(-diff/2)
        bbox[2] = bbox[2] + int(-diff/2)

    if h != target_height or w != target_width:
        bbox = [
            int(bbox[0] * w / max_size),
            int(bbox[1] * h / max_size),
            int(bbox[2] * w / max_size),
            int(bbox[3] * h / max_size),
        ]
    bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
    bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
    bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
    bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

    # IoUs = torch.zeros(len(masks), dtype=torch.float32)
    bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

    masks_area = torch.sum(masks[:, bbox[1] : bbox[3], bbox[0] : bbox[2]], dim=(1, 2))
    orig_masks_area = torch.sum(masks, dim=(1, 2))

    union = bbox_area + orig_masks_area - masks_area
    IoUs = masks_area / union
    max_iou_index = torch.argmax(IoUs)
    max_iou = IoUs[max_iou_index]

    mask_i = masks[max_iou_index].cpu().numpy()
    mask_i = cv2.resize(mask_i, (max_size, max_size))
    mask_i = mask_i[half_max-half_h:half_max+half_h, half_max-half_w:half_max+half_w]
    minRect, contour = get_rotated_bbox_from_mask(mask_i)
    # print(minRect)
    # print(minRect.shape)

    return mask_i, minRect, contour, max_iou.cpu()


def point_prompt(masks, points, point_label, target_height, target_width):  # numpy 处理
    h = masks[0]["segmentation"].shape[0]
    w = masks[0]["segmentation"].shape[1]
    if h != target_height or w != target_width:
        points = [
            [int(point[0] * w / target_width), int(point[1] * h / target_height)]
            for point in points
        ]
    onemask = np.zeros((h, w))
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    for i, annotation in enumerate(masks):
        if type(annotation) == dict:
            mask = annotation['segmentation']
        else:
            mask = annotation
        for i, point in enumerate(points):
            if mask[point[1], point[0]] == 1 and point_label[i] == 1:
                onemask[mask] = 1
            if mask[point[1], point[0]] == 1 and point_label[i] == 0:
                onemask[mask] = 0
    onemask = onemask >= 1
    return onemask, 0

def prompt(results, args, box=None, point=None):
    ori_img = cv2.imread(args.img_path)
    ori_h = ori_img.shape[0]
    ori_w = ori_img.shape[1]
    if box:
        mask, idx = box_prompt(
            results[0].masks.data,
            convert_box_xywh_to_xyxy(args.box_prompt),
            ori_h,
            ori_w,
        )
    elif point:
        mask, idx = point_prompt(
            results, args.point_prompt, args.point_label, ori_h, ori_w
        )
    else:
        return None
    return mask

def draw_masks(input_img, results, input_size):
    w, h, c = input_img.shape
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input_img = cv2.resize(input_img, (new_h, new_w))

    masks = results[0].masks.data

    image_with_masks = np.copy(input_img)
    for i, mask_i in enumerate(masks):
        s_h = int((input_size/2-new_h/2))
        s_w = int((input_size/2-new_w/2))
        mask_i = mask_i[s_w:input_size-s_w, s_h:input_size-s_h]
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        image_with_masks = overlay(image_with_masks, mask_i, color=rand_color, alpha=0.7)

    return image_with_masks

def draw_bbox_masks(input_img, masks, input_size):
    w, h, c = input_img.shape
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input_img = cv2.resize(input_img, (new_h, new_w))

    # masks = results[0].masks.data

    image_with_masks = np.copy(input_img)
    for i, mask_i in enumerate(masks):
        s_h = int((input_size/2-new_h/2))
        s_w = int((input_size/2-new_w/2))
        mask_i = mask_i[s_w:input_size-s_w, s_h:input_size-s_h]
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        image_with_masks = overlay(image_with_masks, mask_i, color=rand_color, alpha=0.7)

    return image_with_masks


def rainbow_fill(size=50):  # simpler way to generate rainbow color
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n/size)
        color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

    return np.array(color_list)

_COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)


def vis(img, annotations, class_names=None, input_size=480):
    # print(img.shape)
    w, h, c = img.shape

    for i in range(len(annotations)):
        box = annotations[i]["bbox"]
        cls_id = int(annotations[i]["id"])
        score = annotations[i]["score"]
        
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[randint(0, 79)] * 255).astype(np.uint8).tolist()
        if class_names is None:
            text = '{}:{:.1f}%'.format(cls_id, score * 100)
        else:
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        # cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0 - 1, y0 - int(1.8 * txt_size[1])),
            (x0 + txt_size[0], y0- int(0.3 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 - int(0.7*txt_size[1])), font, 0.4, txt_color, thickness=1)

        # mask_i = cv2.resize(annotations[i]["mask"], (max_size, max_size))
        # mask_i = mask_i[half_max-half_w:half_max+half_w, half_max-half_h:half_max+half_h]

        # minRect, minEllipse = get_rotated_bbox_from_mask(mask_i)
        
        # for k, rect in enumerate(minRect):
        #     print(rect)
        box = cv2.boxPoints(annotations[i]["rbbox"])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv2.drawContours(img, [box], 0, color)
        # cv2.drawContours(img, [annotations[i]["contour"]], 0, color, 2)
        img = overlay(img, annotations[i]["mask"], color, 0.5)
    return img

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined