import os
import trimesh
import numpy as np
import torch
from pytorch3d.renderer import (
    TexturesVertex
)
from PIL import Image
from torchvision.transforms import ToTensor
from thirdparty.lib.common.mesh_utils import TexturedMeshRenderer
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


def load_and_simplify_mesh(mesh, target_reduction):
    # 将 Trimesh 网格及其颜色信息转换为 Open3D 格式
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )

    # 如果存在顶点颜色信息，则将其附加到 Open3D 网格
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors[:, :3] / 255.0  # 转换到 0-1 范围内
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 应用简化算法
    simplified_mesh = o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(len(mesh.faces) * target_reduction))

    # 获取简化后的颜色信息
    simplified_colors = np.asarray(simplified_mesh.vertex_colors) * 255.0

    # 将简化后的网格及其颜色信息转换回 Trimesh 格式
    simplified_trimesh = trimesh.Trimesh(
        vertices=np.asarray(simplified_mesh.vertices),
        faces=np.asarray(simplified_mesh.triangles),
        vertex_colors=simplified_colors.astype(np.uint8)
    )

    return simplified_trimesh


def load_and_align_obj(obj_path, smpl_path, scale_proportion=None):
    '''
    基于高度比例进行网格对齐
    :param obj_path: 待对齐物体的OBJ文件路径
    :param smpl_path: SMPL参考模型的OBJ文件路径
    :return: 对齐后的物体网格
    '''
    # 加载网格
    obj_mesh = trimesh.load(obj_path, force='mesh')
    smpl_mesh = trimesh.load(smpl_path, force='mesh')

    # 计算包围盒参数
    obj_bbox = obj_mesh.bounding_box
    smpl_bbox = smpl_mesh.bounding_box

    # 提取高度信息(Y轴方向)
    obj_height = obj_bbox.extents[1]
    smpl_height = smpl_bbox.extents[1]

    # 计算高度比例
    scale_factor = smpl_height / obj_height if obj_height > 0 else 1.0

    if scale_proportion:
        scale_factor *= scale_proportion

    # 构建变换矩阵
    transform = np.eye(4)

    # 缩放矩阵 (保持各向同性缩放)
    transform[:3, :3] *= scale_factor

    # 计算平移量 (保持底部中心对齐)
    obj_center = obj_bbox.centroid
    target_center = smpl_bbox.centroid

    # 调整后的物体中心 = 原始中心 * 缩放系数
    scaled_center = obj_center * scale_factor

    # 平移向量 = 目标中心 - 缩放后的物体中心
    transform[:3, 3] = target_center - scaled_center

    # 应用变换
    obj_mesh.apply_transform(transform)

    return obj_mesh


def render_meshes(renderer, verts, faces, vertex_colors):
    # 设置仰角和偏角为 1，使得z轴有梯度
    azim_list = [0]
    elev_list = [0]
    renderer.set_cameras(azim_list, elev_list)

    vert_colors = vertex_colors[:, :, :3]

    renderer.load_mesh(verts, faces, vert_colors)

    images, _ = renderer.render_mesh(bg="white", return_mask=True)
    images = images.squeeze(0)

    mask = renderer.render_mesh_mask(bg="white", return_mask=True)

    return images, mask


def compared_render_image_with_grad(images_combined, images_offset, epsilon=2e-5):
    '''
    得到整体模型渲染图中每一个人的部分
    :param images_combined:
    :param images_offset:
    :param epsilon:
    :return:
    '''
    # 计算两个图像之间的绝对差异
    difference = torch.abs(images_combined - images_offset)

    # 检查各个通道的像素是否一致（差异小于阈值）
    similar_pixels = difference < epsilon

    # 检查三个通道的像素值是否都不为1
    not_one = (images_combined != 1).all(dim=0)

    # 只有当三个通道的像素都近似相等，且都不为1时，才保留原颜色
    combined_condition = similar_pixels.all(dim=0) & not_one

    # 在满足条件的位置保留图片原来的颜色，否则用1表示
    consistent_image = torch.where(combined_condition, images_combined,
                                   torch.tensor(1, dtype=images_combined.dtype, device=images_combined.device))

    return consistent_image


def compared_render_mask_with_grad(images_offset, rendered_image):
    """
    得到 rendered_image 的 mask
    即从 images_offset 中提取出 rendered_image 值不等于 1 的部分。

    :param images_offset: torch.Tensor
    :param rendered_image: torch.Tensor
    :return: torch.Tensor，包含从 images_offset 中提取的符合条件的元素。
    """

    # 找出 rendered_image 中不等于 1 的位置
    mask = rendered_image[2, :, :] != 1
    mask = mask.unsqueeze(0)

    # 利用掩码从 images_offset 中提取对应位置的值
    # 使用掩码创建结果张量形状
    result = torch.zeros_like(images_offset)
    result[mask] = images_offset[mask]

    return result


def compute_image_losses(rendered_image, rendered_mask, target_img):
    # 计算像素损失
    pixel_loss = torch.nn.functional.mse_loss(rendered_image, target_img)

    # 创建 masks
    target_mask = (target_img != 1).any(dim=0, keepdim=True)
    white_value = 1.0
    black_value = 0.0
    target_white = torch.where(target_mask, torch.full_like(target_img, white_value),
                               torch.full_like(target_img, black_value))
    target_white = target_white[0, :, :].unsqueeze(0)

    # 计算 Mask Loss
    mask_loss = torch.nn.functional.mse_loss(rendered_mask, target_white)

    mask_weight = 1

    # 可以根据需求给予 mask_loss 特定的权重
    # total_loss = pixel_loss + mask_weight * mask_loss
    total_loss = mask_weight * mask_loss

    return total_loss


def compute_combined_image_losses(rendered_mask, target_img):
    # 创建 masks
    target_mask = (target_img != 1).any(dim=0, keepdim=True)
    white_value = 1.0
    black_value = 0.0
    target_white = torch.where(target_mask, torch.full_like(target_img, white_value),
                               torch.full_like(target_img, black_value))
    target_white = target_white[0, :, :].unsqueeze(0)

    # 计算 Mask Loss
    mask_loss = torch.nn.functional.mse_loss(rendered_mask, target_white)

    mask_weight = 1

    # 可以根据需求给予 mask_loss 特定的权重
    total_loss = mask_weight * mask_loss

    return total_loss


def optimize_each_mesh(meshes, seg_image_paths, vis_diff=False, iterations=300, lr=1e-2):
    '''
    优化模型的相对位置，通过给每个mesh使用可学习偏移量offset，优化Mesh在整合Mesh渲染图中的部分和参考图像一致
    :param meshes:
    :param seg_image_paths:
    :param iterations:
    :param lr:
    :return:
    '''

    target_reduction_ = 0.005

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    offsets = [torch.zeros((1, 3), requires_grad=True, dtype=torch.float32) for _ in meshes]

    # 使用优化器来优化 translation_params
    optimizer = torch.optim.Adam(offsets, lr=lr)

    # 供参考的分割图像
    target_images = [
        ToTensor()(Image.open(path).convert('RGB')).to(device) for path in seg_image_paths
    ]

    # 简化网格数并提前存储网格信息到mesh_info_list，加快优化速度
    mesh_info_list = []
    for mesh in meshes:
        simplified_mesh = load_and_simplify_mesh(mesh, target_reduction=target_reduction_)
        verts = torch.tensor(simplified_mesh.vertices, dtype=torch.float32).unsqueeze(0).to(device)
        faces = torch.tensor(simplified_mesh.faces, dtype=torch.long).unsqueeze(0).to(device)
        vertex_colors = torch.from_numpy(simplified_mesh.visual.vertex_colors / 255.0).contiguous().float().unsqueeze(
            0).to(device)
        mesh_info = {
            "verts": verts,
            "faces": faces,
            "vertex_colors": vertex_colors,
            "mesh_vertices_shape": simplified_mesh.vertices.shape[0]
        }
        mesh_info_list.append(mesh_info)

    best_offsets = offsets
    best_loss = np.inf
    # 开始优化
    with tqdm(total=iterations, desc='Optimizing') as pbar:
        for i in range(iterations):

            optimizer.zero_grad()

            offset_meshes = []
            for offset, mesh_info in zip(offsets, mesh_info_list):
                # 得到偏移后的模型信息
                original_verts = mesh_info["verts"]
                faces = mesh_info["faces"]
                vertex_colors = mesh_info["vertex_colors"]

                # 将偏移量加到模型的vertices上
                # offset_xy = torch.zeros(1, 2)
                # offset = torch.cat((offset_xy, offset), dim=1)
                repeated_offsets = offset.expand(mesh_info["mesh_vertices_shape"], -1).unsqueeze(0)
                verts = original_verts + repeated_offsets.to(device)

                offset_mesh = {"verts": verts, "faces": faces, "vertex_colors": vertex_colors}
                offset_meshes.append(offset_mesh)

            # 创建combined_mesh
            combined_verts = []
            combined_faces = []
            combined_vertex_colors = []
            vertex_offset = 0

            all_meshes = offset_meshes.copy()
            for i_offset_mesh in all_meshes:
                verts = i_offset_mesh["verts"]
                faces = i_offset_mesh["faces"]
                vertex_colors = i_offset_mesh["vertex_colors"]

                combined_verts.append(verts)

                # 调整面索引，并添加到列表中
                adjusted_faces = faces + vertex_offset
                combined_faces.append(adjusted_faces)

                combined_vertex_colors.append(vertex_colors)

                # 更新顶点偏移量以避免索引冲突
                vertex_offset += verts.shape[1]  # 加入当前网格的顶点数量

            # 将列表连接成一个大的张量
            combined_verts = torch.cat(combined_verts, dim=1)
            combined_faces = torch.cat(combined_faces, dim=1)
            combined_vertex_colors = torch.cat(combined_vertex_colors, dim=1)

            combined_mesh = {"verts": combined_verts, "faces": combined_faces, "vertex_colors": combined_vertex_colors}

            # 创建渲染器
            _, image_height, image_width = target_images[0].shape
            renderer = TexturedMeshRenderer(size=(image_height, image_width), device=device)

            # 渲染combined_mesh的图像
            rendered_image_combined, rendered_mask_combined = render_meshes(renderer, combined_mesh["verts"],
                                                                            combined_mesh["faces"],
                                                                            combined_mesh["vertex_colors"])

            # 将参考图像逐像素相加
            target_images_combined = sum(target_images) - len(target_images) + 1
            # 将结果限制在0到1之间的范围
            target_images_combined = torch.clamp(target_images_combined, 0, 1)
            combined_image_loss = compute_combined_image_losses(rendered_mask_combined, target_images_combined)

            if vis_diff:
                rendered_image_vis = rendered_mask_combined
                target_image_vis = target_images_combined

                # 反转通道维度然后转换为 uint8 格式
                rendered_image_np = (rendered_image_vis.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

                # 计算差异图并转换为uint8格式
                target_img_np = target_image_vis.permute(1, 2, 0).detach().cpu().numpy() * 255
                target_img_np = target_img_np.astype(np.uint8)

                difference_image = np.abs(rendered_image_np.astype(int) - target_img_np.astype(int)).astype(np.uint8)

                diff_img = Image.fromarray(difference_image)
                diff_img.save(f"./tmp/combine_mesh_size_opt_res/difference_image_{i}.png")

            total_loss = combined_image_loss

            if best_loss > total_loss:
                best_loss = total_loss
                best_offsets = [offset.clone() for offset in offsets]

            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({'Loss': f'{total_loss:.4f}'})
            pbar.update(1)
            # print(f"Iteration {i + 1}/{iterations}, Loss: {total_loss.item()}")

    return best_offsets


def main(mesh_dir, smpl_dir, output_dir, seg_image_directory, scale_proportions=None):
    meshes = []
    seg_image_paths = []
    filename_list = []

    # 对尺寸比例进行归一化
    scale_proportions_adjust = []
    scale_proportions_normalization = []
    if scale_proportions:
        for scale_proportion in scale_proportions:
            scale_proportion = 1 / scale_proportion      
            scale_proportions_adjust.append(scale_proportion)
        max_proportion = max(scale_proportions_adjust)
        for scale_proportion in scale_proportions_adjust:
            scale_proportion /= max_proportion
            scale_proportions_normalization.append(scale_proportion)

    for filename in os.listdir(mesh_dir):
        if not filename.endswith('_refine.obj'):
            continue
        filename_list.append(filename)
        scale_proportions_normalization.append(1)

    for (filename, scale_proportion) in zip(filename_list, scale_proportions_normalization):

        obj_path = os.path.join(mesh_dir, filename)
        smpl_path = os.path.join(smpl_dir, filename.replace('_refine.obj', '_smpl_mesh.obj'))

        if not os.path.exists(smpl_path):
            print(f"Skipping {obj_path}, no corresponding SMPL model found.")
            continue

        seg_image_path = os.path.join(seg_image_directory, filename.replace('_refine.obj', '_seg.png'))
        if not os.path.exists(seg_image_path):
            print(f"Skipping {obj_path}, no corresponding segmentation image found.")
            continue

        # filename_list.append(filename)
        meshes.append(load_and_align_obj(obj_path, smpl_path, scale_proportion))
        seg_image_paths.append(seg_image_path)

    offsets = optimize_each_mesh(meshes, seg_image_paths, vis_diff=True)

    # 假设 offsets 是一个包含 PyTorch 张量的列表
    offsets_tensor = torch.stack(offsets)  # 将列表转换为一个 PyTorch 张量

    # 找到 offsets 中绝对值最小的偏移并减去它
    min_offset = offsets_tensor.abs().min(dim=0, keepdim=True).values
    adjusted_offsets = offsets_tensor - min_offset

    result_meshes = []  # 如果之前是这样，那这里需要初始化。

    for id, (mesh, offset) in enumerate(zip(meshes, adjusted_offsets)):
        verts = mesh.vertices

        # 创建 z 轴为 0 的偏移量
        # offset_xy = torch.zeros(1, 2)
        # full_offset = torch.cat((offset_xy, offset), dim=1)
        full_offset = offset

        # 将偏移量应用到顶点
        repeated_offsets = full_offset.expand(verts.shape[0], -1)
        offset_verts = verts + repeated_offsets.detach().cpu().numpy()

        # 创建新的三角网格
        result_mesh = trimesh.Trimesh(
            vertices=offset_verts,
            faces=mesh.faces,
            vertex_colors=mesh.visual.vertex_colors
        )

        result_meshes.append(result_mesh)
        output_mesh = os.path.join(output_dir, f"{filename_list[id]}")
        result_mesh.export(output_mesh)

    combined_optimized_mesh = trimesh.util.concatenate(result_meshes)
    output_combined_mesh = os.path.join(output_dir, "size_refined_combined_mesh.obj")
    combined_optimized_mesh.export(output_combined_mesh)
    print(f"Optimized combined mesh exported to {output_combined_mesh}")


if __name__ == '__main__':
    root_dir = './sota_compare_test/Ours'
    # subdir = '42'
    for subdir in os.listdir(root_dir):
        mesh_directory = os.path.join(root_dir, subdir, '6_refined_mesh' )
        smpl_directory = os.path.join(root_dir, subdir, '1_raw_pic_estimated' )
        output_directory = os.path.join(root_dir, subdir, '8_size_refined_combined_mesh')
        seg_image_directory = os.path.join(root_dir, subdir, '2_pic_segmented')
        
        if not os.path.exists(mesh_directory):
            continue

        scale_proportions = None

        main(mesh_directory, smpl_directory, output_directory, seg_image_directory, scale_proportions)
