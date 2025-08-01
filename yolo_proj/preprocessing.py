import os
import glob
import cv2
import numpy as np

def adjust_contrast(img, alpha=1.5, beta=0):
    # alpha: 대비, beta: 밝기
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def rotate_image_and_boxes(img, boxes, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    new_boxes = []
    for box in boxes:
        cls, x, y, bw, bh = box
        # YOLO 좌표를 픽셀로 변환
        px = x * w
        py = y * h
        # 회전 적용
        coords = np.array([px, py, 1])
        px_new, py_new = np.dot(M, coords)
        # 다시 YOLO 좌표로 변환
        x_new = px_new / w
        y_new = py_new / h
        new_boxes.append([cls, x_new, y_new, bw, bh])
    return rotated_img, new_boxes

def center_crop(img, boxes, crop_ratio=0.8):
    h, w = img.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    start_x = (w - cw) // 2
    start_y = (h - ch) // 2
    cropped_img = img[start_y:start_y+ch, start_x:start_x+cw]
    new_boxes = []
    for box in boxes:
        cls, x, y, bw, bh = box
        # YOLO 좌표를 crop 기준으로 변환
        x_new = (x * w - start_x) / cw
        y_new = (y * h - start_y) / ch
        bw_new = bw * w / cw
        bh_new = bh * h / ch
        new_boxes.append([cls, x_new, y_new, bw_new, bh_new])
    return cropped_img, new_boxes

def load_labels(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            boxes.append(parts)
    return boxes

def save_labels(label_path, boxes):
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(' '.join(f'{v:.6f}' if i else str(int(v)) for i, v in enumerate(box)) + '\n')

def augment_images(img_dir, label_dir, out_img_dir, out_label_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    img_paths = glob.glob(f'{img_dir}/*.jpg')
    for img_path in img_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f'{img_name}.txt')
        if not os.path.exists(label_path):
            continue
        img = cv2.imread(img_path)
        boxes = load_labels(label_path)

        # 1. 좌우 플립
        flip_img = cv2.flip(img, 1)
        flip_boxes = []
        for box in boxes:
            cls, x, y, bw, bh = box
            flip_boxes.append([cls, 1-x, y, bw, bh])
        cv2.imwrite(f'{out_img_dir}/{img_name}_flip.jpg', flip_img)
        save_labels(f'{out_label_dir}/{img_name}_flip.txt', flip_boxes)

        # 2. 로테이션 (+15도)
        rot_img, rot_boxes = rotate_image_and_boxes(img, boxes, 90)
        cv2.imwrite(f'{out_img_dir}/{img_name}_rot15.jpg', rot_img)
        save_labels(f'{out_label_dir}/{img_name}_rot15.txt', rot_boxes)

        # 3. 로테이션 (-15도)
        rot_img, rot_boxes = rotate_image_and_boxes(img, boxes, -90)
        cv2.imwrite(f'{out_img_dir}/{img_name}_rot-15.jpg', rot_img)
        save_labels(f'{out_label_dir}/{img_name}_rot-15.txt', rot_boxes)

        # 4. 대비 증가
        contrast_img = adjust_contrast(img, alpha=1.8)
        cv2.imwrite(f'{out_img_dir}/{img_name}_contrast.jpg', contrast_img)
        save_labels(f'{out_label_dir}/{img_name}_contrast.txt', boxes)

        # 5. 중앙 크롭
        crop_img, crop_boxes = center_crop(img, boxes, crop_ratio=0.8)
        cv2.imwrite(f'{out_img_dir}/{img_name}_crop.jpg', crop_img)
        save_labels(f'{out_label_dir}/{img_name}_crop.txt', crop_boxes)

    print("증강 완료!")

# 사용 예시
augment_images(
    img_dir='/home/work/wonjun/usc/yolo_proj/dataset/train/images',
    label_dir='/home/work/wonjun/usc/yolo_proj/dataset/train/labels',
    out_img_dir='/home/work/wonjun/usc/yolo_proj/train_aug/images',
    out_label_dir='/home/work/wonjun/usc/yolo_proj/train_aug/labels'
)