import argparse
import json
import os
import csv
from tqdm import tqdm
# -----
# [{'scan': {'rgb': , 'normal':, 'depth':, 'disparity':, 'mask':,}
#   'smplx': {'normal':, 'depth':, 'disparity':, 'mask':, } },
#  { }]
# -----
# python tools/extract_meta_info.py --root_path /apdcephfs_cq10/share_1330077/hexu/NVS/data_rendering/THuman2/ortho_5 --dataset_name thuman2_ortho_5
# -----
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="/apdcephfs_cq10/share_1330077/hexu/NVS/data_rendering/CustomHumans/ortho_6")
parser.add_argument("--dataset_name", type=str, default="customhumans_ortho_6")
parser.add_argument("--meta_info_name", type=str)

args = parser.parse_args()

if args.meta_info_name is None:
    args.meta_info_name = args.dataset_name

thuman2_list = []
customhumans_list = []
with open("data_split.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过标题行
    for row in reader:
            if row[0]:  # 检查是否为空
                thuman2_list.append(f"{int(row[0]):04d}")
            if row[1]:
                customhumans_list.append(row[1])
if "thuman2" in args.dataset_name:
    test_list = thuman2_list
if "customhumans" in args.dataset_name:
    test_list = customhumans_list

meta_infos = []
# collect all video_folder paths
for idx, person in tqdm(enumerate(sorted(os.listdir(args.root_path)))):
    if person in test_list:
        continue
    person_path = os.path.join(args.root_path, person)
    if not os.path.isdir(person_path):
        continue
    scan_rgb_path = os.path.join(person_path, "scan", "rgb")
    scan_normal_path = os.path.join(person_path, "scan", "normal")
    scan_depth_path = os.path.join(person_path, "scan", "depth")
    scan_disparity_path = os.path.join(person_path, "scan", "disparity")
    scan_mask_path = os.path.join(person_path, "scan", "mask")
    
    smplx_semantic_path = os.path.join(person_path, "smplx", "semantic")
    smplx_normal_path = os.path.join(person_path, "smplx", "normal")
    smplx_depth_path = os.path.join(person_path, "smplx", "depth")
    smplx_disparity_path = os.path.join(person_path, "smplx", "disparity")
    smplx_mask_path = os.path.join(person_path, "smplx", "mask")
    
    # 参考normal和depth 用marigold估计的
    ref_normal_path = os.path.join(person_path, "marigold", "normal")
    ref_depth_path = os.path.join(person_path, "marigold", "depth")
    
    
    assert os.path.exists(scan_rgb_path) 
    assert os.path.exists(scan_normal_path)
    assert os.path.exists(scan_depth_path)
    assert os.path.exists(scan_disparity_path)
    assert os.path.exists(scan_mask_path)
    assert os.path.exists(smplx_semantic_path)
    assert os.path.exists(smplx_normal_path)
    assert os.path.exists(smplx_depth_path)
    assert os.path.exists(smplx_disparity_path)
    assert os.path.exists(smplx_mask_path)
    assert os.path.exists(ref_normal_path)
    assert os.path.exists(ref_depth_path)

    meta_infos.append({
                    'scan': 
                        {'rgb': scan_rgb_path, 
                         'normal': scan_normal_path, 
                         'depth': scan_depth_path, 
                         'disparity': scan_disparity_path, 
                         'mask': scan_mask_path,},
                    'smplx': 
                        {'semantic': smplx_semantic_path,
                        'normal': smplx_normal_path, 
                        'depth': smplx_depth_path, 
                        'disparity': smplx_disparity_path, 
                        'mask': smplx_mask_path, },
                    'ref':
                        {'normal': ref_normal_path,
                        'depth': ref_depth_path,},
                        })
    

print(len(meta_infos))
json.dump(meta_infos, open(f"./data/{args.meta_info_name}_meta.json", "w"))
