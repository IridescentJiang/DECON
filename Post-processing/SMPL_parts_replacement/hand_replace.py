import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)
import os
import sys
import json

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from replacement_util import (SMPLX, part_removal, poisson)
from trimesh.collision import CollisionManager

import os
import trimesh
import numpy as np
import torch
import re

torch.backends.cudnn.benchmark = True


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['vertex_indices'], data['faces']


def load_obj(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines


def extract_vertices(obj_lines, vertex_indices):
    vertices = []
    index_mapping = {}
    valid_vertices = [line for line in obj_lines if line.startswith('v ')]
    for new_index, obj_index in enumerate(vertex_indices):
        # Adjust by subtracting 1 because vertex_indices are 1-based
        line = valid_vertices[obj_index - 1]
        if line.startswith('v '):
            # Track the new index mapping
            index_mapping[obj_index] = new_index + 1  # OBJ format requires 1-based indexing
            vertices.append(line.strip())
    return vertices, index_mapping


def extract_faces(face_lines, index_mapping):
    extracted_faces = []
    for face_str in face_lines:
        if face_str.startswith('f '):
            # Parse the face indices
            original_indices = [int(i) for i in face_str.split()[1:]]

            # Remap them to the new indices
            try:
                new_indices = [index_mapping[idx] for idx in original_indices]
                extracted_faces.append(f"f {' '.join(map(str, new_indices))}")
            except KeyError:
                # Skip faces that reference vertices not in index_mapping
                continue
    return extracted_faces


def save_obj(vertices, faces, output_file):
    with open(output_file, 'w') as file:
        for vertex in vertices:
            file.write(vertex + '\n')
        for face in faces:
            file.write(face + '\n')

gpu_device = 0


def process_hand(json_filename, obj_lines, identifier, mesh_dir, json_dir):
    json_file = os.path.join(json_dir, json_filename)
    vertex_indices, faces = load_json(json_file)

    vertices, index_mapping = extract_vertices(obj_lines, vertex_indices)
    hand_faces = extract_faces(faces, index_mapping)

    hand_output = os.path.join(mesh_dir, f'{identifier}_{json_filename.replace(".json", ".obj")}')
    save_obj(vertices, hand_faces, hand_output)

    return trimesh.load(hand_output)


def clean_mesh(mesh):
    cc = mesh.split(only_watertight=False)

    out_mesh = cc[0]
    bbox = out_mesh.bounds
    height = bbox[1, 0] - bbox[0, 0]
    for c in cc:
        bbox = c.bounds
        if height < bbox[1, 0] - bbox[0, 0]:
            height = bbox[1, 0] - bbox[0, 0]
            out_mesh = c

    # mesh decimation for faster rendering
    # current_faces = len(out_mesh.faces)
    # target_reduction = 1 - (50000 / current_faces)  # 计算需要减少的比例

    out_mesh = out_mesh.simplify_quadratic_decimation(100000)

    return out_mesh

def get_vertex_index_map(smpl_mesh, right_hand_mesh):
    # 取得 right_hand_mesh 的所有顶点在 smpl_mesh 中的索引
    indices_map = {tuple(v): idx for idx, v in enumerate(smpl_mesh.vertices)}
    right_hand_indices = [indices_map[tuple(v)] for v in right_hand_mesh.vertices]
    return right_hand_indices


def is_inside(smpl_mesh, hand_mesh):

    # 检查hand_mesh的所有顶点是否在smpl_mesh内部
    contains = smpl_mesh.contains(hand_mesh.vertices)

    # 如果任何点都不在 smpl_mesh 内部，则返回 False
    if not np.any(contains):
        return False
    return True

def process_hand_collision(json_file, obj_lines, identifier, hand_mesh, collision_manager, hand_type, mesh_dir, json_dir):
    # 处理手部网格
    smpl_mesh = process_hand(json_file, obj_lines, identifier, mesh_dir, json_dir)

    # # 设置碰撞管理器
    # collision_manager.add_object(f'smpl_mesh_no_{hand_type}_hand', smpl_mesh)
    # collision_manager.add_object(f'{hand_type}_hand_mesh', hand_mesh)

    # 检查碰撞和内部判断
    # if not collision_manager.in_collision_internal() and not is_inside(smpl_mesh, hand_mesh):
    if not is_inside(smpl_mesh, hand_mesh):
        return hand_mesh
    return None


def refine_and_export_mesh(
        pred_obj_mesh,
        mid_result_path,
        refine_mesh_path,
        smpl_mesh,
        device,
        hand_mesh=None,
        feet_mesh=None,
        face_mesh=None
):
    # 初始化最终网格
    final_mesh_part = pred_obj_mesh.copy()

    # 可选部件处理（按处理顺序排列）
    processing_order = [
        (feet_mesh, 0.06, "feet"),
        (face_mesh, 0.03, "face"),
        (hand_mesh, 0.05, "hand")
    ]

    for mesh, threshold, region in processing_order:
        if mesh is not None:
            final_mesh_part = part_removal(
                final_mesh_part,
                mesh,
                threshold,
                device,
                smpl_mesh,
                region=region
            )

    # 清理中间网格
    final_mesh_part = clean_mesh(final_mesh_part)

    # 动态组装完整网格（自动跳过None值）
    mesh_components = [
        mesh for mesh in [hand_mesh, feet_mesh, face_mesh, final_mesh_part]
        if mesh is not None
    ]

    # 空网格保护
    if not mesh_components:
        raise ValueError("No valid mesh components to assemble")

    full_mesh = sum(mesh_components, start=type(mesh_components[0])())  # 保持类型一致性

    # 泊松重建（带容错）
    return poisson(
        full_mesh,
        mid_result_path,
        refine_mesh_path,
        depth=10,
        decimation=False
    )

# def refine_and_export_mesh(hand_mesh, pred_obj_mesh, mid_result_path, refine_mesh_path, smpl_mesh, feet_mesh, face_mesh, device):
#     if feet_mesh:
#         final_mesh_part = part_removal(pred_obj_mesh, feet_mesh, 0.06, device, smpl_mesh, region="feet")
#     # final_mesh_part = part_removal(final_mesh_part, face_mesh, 0.03, device, smpl_mesh, region="face")
#
#     if hand_mesh:
#         final_mesh_part = part_removal(final_mesh_part, hand_mesh, 0.05, device, smpl_mesh, region="hand")
#
#     final_mesh_part = clean_mesh(final_mesh_part)
#     # full_mesh = hand_mesh + feet_mesh + face_mesh + final_mesh_part
#     full_mesh = hand_mesh + feet_mesh + face_mesh + final_mesh_part
#
#     return poisson(full_mesh, mid_result_path, refine_mesh_path, 10)

def main():
    mesh_dir = './mesh_for_replacement'
    smpl_vertex_id_dir = './smplx_vertex_id'
    mid_result_dir = './mid_results'
    result_dir = './results'

    pattern = re.compile(r'^[A-Za-z0-9]+_[A-Za-z0-9]+\.obj$')

    identifiers = []
    # 遍历目录下所有文件
    for filename in os.listdir(mesh_dir):
        # 检查文件名格式和扩展名
        if pattern.match(filename):
            # 去除扩展名，保留 {M}_{N} 格式的纯文件名
            identifier = os.path.splitext(filename)[0]
            identifiers.append(identifier)

    for test_id in identifiers:
        print(test_id)

        smpl_obj_file = os.path.join(mesh_dir, f'{test_id}_smplx.obj')
        pred_obj_file = os.path.join(mesh_dir, f'{test_id}.obj')

        # Load inputs
        smpl_obj = load_obj(smpl_obj_file)
        device = torch.device(f"cuda:{gpu_device}")

        # Process hands
        right_hand_mesh = process_hand('right_hand.json', smpl_obj, test_id, mid_result_dir, smpl_vertex_id_dir)
        left_hand_mesh = process_hand('left_hand.json', smpl_obj, test_id, mid_result_dir, smpl_vertex_id_dir)
        feet_mesh = process_hand('feet.json', smpl_obj, test_id, mid_result_dir, smpl_vertex_id_dir)
        face_mesh = process_hand('face.json', smpl_obj, test_id, mesh_dir, smpl_vertex_id_dir)

        # Load prediction and SMPL meshes
        pred_obj_mesh = trimesh.load(pred_obj_file)
        smpl_mesh = trimesh.load(smpl_obj_file)

        hand_mesh = None

        # 判断smpl的左手和右手有没有和身体部分重合，如果没有重合则粘贴在mesh上
        # 右手部分
        right_collision_manager = CollisionManager()
        if process_hand_collision('smpl_no_right_hand.json', smpl_obj, test_id, right_hand_mesh,
                                           right_collision_manager, "right", mid_result_dir, smpl_vertex_id_dir):
            hand_mesh += right_hand_mesh

        # 左手部分
        left_collision_manager = CollisionManager()
        if process_hand_collision('smpl_no_left_hand.json', smpl_obj, test_id, left_hand_mesh,
                                           left_collision_manager, "left", mid_result_dir, smpl_vertex_id_dir):
            hand_mesh += left_hand_mesh

        # 网格文件导出路径
        refine_mesh_path = os.path.join(result_dir, f'{test_id}_replaced.obj')

        # 网格细化与导出
        # refined_mesh = refine_and_export_mesh(hand_mesh, pred_obj_mesh, mid_result_dir, refine_mesh_path, smpl_mesh,
        #                                       feet_mesh, face_mesh, device)

        refined_mesh = refine_and_export_mesh(
            pred_obj_mesh=pred_obj_mesh,
            mid_result_path=mid_result_dir,
            refine_mesh_path=refine_mesh_path,
            smpl_mesh=smpl_mesh,
            device=device,
            hand_mesh=None,
            feet_mesh=None,
            face_mesh=None,
        )

        # Export refined mesh
        refined_mesh.export(refine_mesh_path)


if __name__ == "__main__":
    main()
