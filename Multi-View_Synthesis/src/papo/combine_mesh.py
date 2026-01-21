import os
import trimesh
import numpy as np

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

def main(mesh_dir, smpl_dir, output_dir, output_file, suffix, scale_proportions=None):
    combined_mesh = trimesh.Trimesh()
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
        if not filename.endswith(suffix):
            continue
        filename_list.append(filename)
        scale_proportions_normalization.append(1)

    # Iterate over files in directory
    for (filename, scale_proportion) in zip(filename_list, scale_proportions_normalization):
        
        obj_path = os.path.join(mesh_dir, filename)
        smpl_path = os.path.join(smpl_dir, filename.replace(suffix, '_smpl_mesh.obj'))

        # Skip if corresponding SMPL file does not exist
        if not os.path.exists(smpl_path):
            print(f"Skipping {obj_path}, no corresponding SMPL model found.")
            continue

        aligned_mesh = load_and_align_obj(obj_path, smpl_path, scale_proportion)
        output_mesh = os.path.join(output_dir, filename.replace(suffix, '_combine_pos.obj'))
        aligned_mesh.export(output_mesh)
        combined_mesh = trimesh.util.concatenate(combined_mesh, aligned_mesh)

    # Export the combined mesh
    combined_mesh.export(output_file)
    print(f"Combined mesh exported to {output_file}")

if __name__ == '__main__':
    root_dir = './tmp/1'
    combine_for_econ = False

    if combine_for_econ:
        # combined mesh for econ
        econ_dir = './sota_compare_test/ECON'
        suffix = '_full.obj'

        for subdir in os.listdir(econ_dir):
            mesh_directory = os.path.join(econ_dir, subdir)
            smpl_directory = os.path.join(root_dir, subdir, '1_raw_pic_estimated')
            output_dir = os.path.join(econ_dir, subdir)
            output_filename = os.path.join(econ_dir, subdir, 'combined_mesh.obj')

            main(mesh_directory, smpl_directory, output_dir, output_filename, suffix)
    else:
        suffix = '_refine.obj'

        scale_proportions = None

        for subdir in os.listdir(root_dir):
            mesh_directory = os.path.join(root_dir, subdir, '6_refined_mesh' )
            smpl_directory = os.path.join(root_dir, subdir, '1_raw_pic_estimated' )
            output_dir = os.path.join(root_dir, subdir, '7_combined_mesh')
            output_filename = os.path.join(root_dir, subdir, '7_combined_mesh', 'combined_mesh.obj' )

            main(mesh_directory, smpl_directory, output_dir, output_filename, suffix, scale_proportions)

