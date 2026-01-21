import trimesh
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import os
root_dir = "/apdcephfs_cq10/share_1330077/hexu/NVS/test_set"
# root_dir = "/apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/test_set"
for subject in tqdm(sorted(os.listdir(root_dir))):
    subject_dir = os.path.join(root_dir, subject)
    ##### 需要翻转
    # source_mesh_path = f"{subject_dir}/mv/gt_gen/nvs_dual_branch@step2_union_v1/neus/save/it15000-mc512-colored.obj"
    # target_mesh_dir = f"{subject_dir}/mv/gt_gen/nvs_dual_branch@step2_union_v1/neus" # 下面存mesh和textured_mesh
    
    ##### 需要翻转
    # source_mesh_path = f"{subject_dir}/mv/free_gen/neus/save/it15000-mc320-colored.obj"
    # target_mesh_dir = f"{subject_dir}/mv/free_gen/neus" # 下面存mesh和textured_mesh
    
    # # 不翻转
    # source_mesh_path = f"{subject_dir}/mv/optmized_gen/neus/save/it15000-mc320-colored.obj"
    # target_mesh_dir = f"{subject_dir}/mv/optmized_gen/neus" # 下面存mesh和textured_mesh
    
    # 不翻转
    # source_mesh_path = f"{subject_dir}/mv/optimized_wo_normal_gen/neus/save/it15000-mc320-colored.obj"
    # target_mesh_dir = f"{subject_dir}/mv/optimized_wo_normal_gen/neus" # 下面存mesh和textured_mesh
    
    # source_mesh_path = f"{subject_dir}/mv/optimized_wo_normal_gen/neus/save/it15000-mc320-colored.obj"
    # target_mesh_dir = f"{subject_dir}/mv/optimized_wo_normal_gen/neus" # 下面存mesh和textured_mesh
    
    source_mesh_path = f"{subject_dir}/mv/pymafx_gen/neus/save/it10000-mc320-colored.obj"
    target_mesh_dir = f"{subject_dir}/mv/pymafx_gen/neus" # 下面存mesh和textured_mesh

    # clean mesh清除漂浮
    mesh_lst = trimesh.load_mesh(source_mesh_path, process=False, maintains_order=True)
    mesh_lst = mesh_lst.split(only_watertight=False)
    comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
    mesh_clean = mesh_lst[comp_num.index(max(comp_num))]

    ########## 对于testset上的gt_gen和free_gen要把z反向 其他不用!
    # faces = mesh_clean.faces
    # vertices = np.zeros_like(mesh_clean.vertices)
    # vertices[:, 0] = mesh_clean.vertices[:, 0]
    # vertices[:, 1] = mesh_clean.vertices[:, 1]
    # vertices[:, 2] = -mesh_clean.vertices[:, 2]
    # vertex_colors = mesh_clean.visual.vertex_colors
    # faces = np.array(faces)[:,::-1] # 这里改变faces的顺序 防止内外表面颠倒
    # vertices = np.array(vertices)
    # vertex_colors = np.array(vertex_colors)
    # mesh_clean = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
    
    '''
    这里是自适应修复正方面 但是可能会失败 最好是face直接颠倒顺序
    # mesh_clean.fix_normals() # 翻转z轴后会导致内外表面反了
    # mesh_clean.export("clean_mesh.obj")
    '''
    
    # remesh 简化点数以及平滑
    mesh = mesh_clean.simplify_quadratic_decimation(200000)
    mesh = trimesh.smoothing.filter_humphrey(
            mesh, alpha=0.1, beta=0.5, iterations=10, laplacian_operator=None
        )

    # 这里得到的mesh没有颜色 先存一次
    mesh.export(f"{target_mesh_dir}/mesh.obj")

    # 利用clean_mesh中最近的点的颜色弥补简化后mesh的颜色
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(mesh_clean.vertices)
    _, indices = nbrs.kneighbors(mesh.vertices)
    simplified_colors = mesh_clean.visual.vertex_colors[indices].reshape(-1, 4)
    mesh.visual.vertex_colors = simplified_colors

    mesh.export(f"{target_mesh_dir}/mesh_textured.obj")
    
    print(subject, "Done!")
