import trimesh
import numpy as np
from tqdm import tqdm
import os
import re

root_dir = "./"

for subject in tqdm(sorted(os.listdir(root_dir))):
    subject_dir = os.path.join(root_dir, subject)
    
    # 定位到 refined mesh 目录
    refined_mesh_dir = os.path.join(subject_dir, "6_refined_mesh")
    if not os.path.exists(refined_mesh_dir):
        print(f"跳过缺失目录：{refined_mesh_dir}")
        continue
    
    # 扫描所有符合 M_N_replaced.obj 格式的文件
    for filename in os.listdir(refined_mesh_dir):
        match = re.fullmatch(r"([a-zA-Z0-9]+)_(\d+)_replaced\.obj", filename)
        if not match:
            continue
            
        M, N = match.groups()
        print(f"处理 {M}_{N} 组合")
        
        # 构建完整文件路径
        source_path = os.path.join(refined_mesh_dir, filename)
        output_filename = f"{M}_{N}_refine.obj"
        output_path = os.path.join(refined_mesh_dir, output_filename)
        
        # 处理网格
        try:
            # 加载并清理网格
            mesh_lst = trimesh.load_mesh(source_path, process=False, maintains_order=True)
            mesh_lst = mesh_lst.split(only_watertight=False)
            
            # 选择最大组件
            comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
            mesh_clean = mesh_lst[comp_num.index(max(comp_num))]

            # 网格简化和平滑
            mesh = mesh_clean.simplify_quadratic_decimation(200000)
            mesh = trimesh.smoothing.filter_humphrey(
                mesh, alpha=0.1, beta=0.5, iterations=10, laplacian_operator=None
            )
            
            # 导出处理后的网格
            mesh.export(output_path)
            print(f"已生成：{output_path}")
            
        except Exception as e:
            print(f"处理失败 {filename}: {str(e)}")
            continue

print("全部处理完成！")
