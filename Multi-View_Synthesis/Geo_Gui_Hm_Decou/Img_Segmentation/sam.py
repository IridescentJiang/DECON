import torch
from PIL import Image
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def save_masks_with_original_image(image, mask, path_prefix, box_index, i):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)

    # 检查 mask 数据类型并转换为布尔型
    if mask.dtype != bool:
        mask = mask.astype(bool)

    # 创建一个全白的 mask 图像
    mask_img = np.ones_like(image) * 255

    # 将 mask 应用到原图像
    mask_img[mask] = image[mask]

    # 设置输出路径
    if i == 0:
        output_path = f"{path_prefix}_{box_index}_seg.png"
    else:
        output_path = f"{path_prefix}_{box_index}_seg_{i}.png"

    # 将 mask 应用后的图像从 RGB 转换为 BGR（因为 OpenCV 保存时默认是 BGR）
    mask_img_bgr = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)

    # 保存结果
    cv2.imwrite(output_path, mask_img_bgr)

    print(f"Masks have been saved：{output_path}")


def segment_images_in_directory(image_dir, result_dir, predictor):
    # 创建结果目录
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    # 列出目录中所有文件
    for filename in os.listdir(image_dir):
        # 筛选出没有下划线的图像文件
        # if '_' not in filename and filename.endswith('.png'):
        if filename.endswith('.png'):
            base_name = os.path.splitext(filename)[0]
            photoPath = os.path.join(image_dir, filename)

            # 加载图像
            image = cv2.imread(photoPath)
            if image is None:
                print(f"Failed to read image {photoPath}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            # 查找相关的 _box.txt 文件
            box_index = 0
            while True:
                box_file = f"{base_name}_{box_index}_box.txt"
                box_path = os.path.join(image_dir, box_file)

                if not os.path.isfile(box_path):
                    break  # 没有更多的 box 文件

                # 读取边界框坐标
                with open(box_path, 'r') as f:
                    input_box = np.fromstring(f.read().strip('[]'), sep=',')

                # 执行分割预测
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, logits = predictor.predict(
                        box=input_box,
                        multimask_output=False,
                    )

                # show_masks(image, mask, scores, borders=True)

                # 保存分割结果
                output_prefix = f"{base_name}"
                output_path = os.path.join(result_dir, output_prefix)
                for i, mask in enumerate(masks):
                    save_masks_with_original_image(image, mask, output_path, box_index, i)

                box_index += 1


subject_name_list = ["test"]
for subject_name in subject_name_list:
    segment_images_in_directory(f'../Experiment/sports_img/image_estimation/', f'../Experiment/sports_img/image_segmentation/', predictor)

