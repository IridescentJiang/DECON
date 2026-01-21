import os
import pickle
from tqdm.auto import tqdm
import torch
import numpy as np
from utils.transforms import unNormalize
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, vis_meshes_img_depth, output_meshes, vis_boxes, vis_sat, vis_scale_img, get_colors_rgb, vis_points
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
import time
import cv2
import trimesh
import json


def inference(model, infer_dataloader, conf_thresh, results_save_path=None,
              distributed=False, accelerator=None):
    assert results_save_path is not None
    assert accelerator is not None

    accelerator.print(f'Results will be saved at: {results_save_path}')
    os.makedirs(results_save_path, exist_ok=True)
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model

    progress_bar = tqdm(total=len(infer_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('inference')

    for itr, (samples, targets) in enumerate(infer_dataloader):
        samples = [sample.to(device=cur_device, non_blocking=True) for sample in samples]
        with torch.no_grad():
            outputs = model(samples, targets)
        bs = len(targets)
        for idx in range(bs):
            img_size = targets[idx]['img_size'].detach().cpu().int().numpy()
            img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]

            # pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()
            pred_joint2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()
            pred_poses = outputs['pred_poses'][idx][select_queries_idx].detach().cpu().numpy()
            pred_betas = outputs['pred_betas'][idx][select_queries_idx].detach().cpu().numpy()
            pred_transls = outputs['pred_transl'][idx][select_queries_idx].detach().cpu().numpy()

            for i, (pred_pose, pred_beta, pred_transl) in enumerate(zip(pred_poses, pred_betas, pred_transls)):
                output_dict = {
                    "poses": pred_pose.reshape(24,3).tolist(),
                    "betas": pred_beta.tolist(),
                    # "transl": [0, 0, 0],
                    "transl": pred_transl.tolist(),
                }
                filename = os.path.join(results_save_path, f"{img_name}_{i}_smpl_para.json")
                os.makedirs(results_save_path, exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(output_dict, f, indent=2)

            ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
            ori_img[img_size[0]:, :, :] = 255
            ori_img[:, img_size[1]:, :] = 255
            ori_img[img_size[0]:, img_size[1]:, :] = 255
            ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)

            sat_img = vis_sat(ori_img.copy(),
                              input_size=model.input_size,
                              patch_size=14,
                              sat_dict=outputs['sat'],
                              bid=idx)[:img_size[0], :img_size[1]]

            height, width, channels = ori_img.shape
            white_img = np.ones((height, width, channels), dtype=np.uint8) * 255

            colors = get_colors_rgb(len(pred_verts))
            pred_mesh_imgs = []
            for pred_vert, color in zip(pred_verts, colors):
                pred_vert = np.expand_dims(pred_vert, axis=0)
                color = np.expand_dims(color, axis=0)
                
                # 输出SMPL模型渲染图（彩色）
                # pred_mesh_img = vis_meshes_img(img=white_img.copy(),
                #                                verts=pred_vert,
                #                                smpl_faces=smpl_layer.faces,
                #                                cam_intrinsics=outputs['pred_intrinsics'][idx].reshape(3,
                #                                                                                       3).detach().cpu(),
                #                                colors=color)[:img_size[0], :img_size[1]]
                # 输出SMPL模型深度图
                
                pred_mesh_img = vis_meshes_img_depth(img=white_img.copy(),
                                               verts=pred_vert,
                                               smpl_faces=smpl_layer.faces,
                                               cam_intrinsics=outputs['pred_intrinsics'][idx].reshape(3,
                                                                                                      3).detach().cpu(),
                                               colors=color)[:img_size[0], :img_size[1]]
                pred_mesh_imgs.append(pred_mesh_img)

            smpl_meshes = output_meshes(
                                meshes=pred_verts,
                                face=smpl_layer.faces)  
            smpl_meshes_path = os.path.join(results_save_path, f"{img_name}_smpl_meshes.obj")
            smpl_meshes.export(smpl_meshes_path)
            
            for i, pred_vert in enumerate(pred_verts):
                pred_vert = np.expand_dims(pred_vert, axis=0)
                smpl_mesh = output_meshes(
                        meshes=pred_vert,
                        face=smpl_layer.faces)  
                smpl_mesh_path = os.path.join(results_save_path, f"{img_name}_{i}_smpl_mesh.obj")
                smpl_mesh.export(smpl_mesh_path)

            pred_meshes_img = vis_meshes_img(img=ori_img.copy(),
                                           verts=pred_verts,
                                           smpl_faces=smpl_layer.faces,
                                           cam_intrinsics=outputs['pred_intrinsics'][idx].reshape(3, 3).detach().cpu(),
                                           colors=colors)[:img_size[0], :img_size[1]]

            # 渲染SMPL
            if 'enc_outputs' not in outputs:
                pred_scale_img = np.zeros_like(ori_img)[:img_size[0], :img_size[1]]
            else:
                enc_out = outputs['enc_outputs']
                h, w = enc_out['hw'][idx]
                flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                scale_map = torch.zeros((h, w, 2))
                scale_map[ys, xs] = flatten_map

                pred_scale_img = vis_scale_img(img=ori_img.copy(),
                                               scale_map=scale_map,
                                               conf_thresh=model.sat_cfg['conf_thresh'],
                                               patch_size=28)[:img_size[0], :img_size[1]]

            cv2.imwrite(os.path.join(results_save_path, f'{img_name}.png'),
                        ori_img[:img_size[0], :img_size[1]])

            for i, pred_mesh_img in enumerate(pred_mesh_imgs):
                cv2.imwrite(os.path.join(results_save_path, f'{img_name}_{i}_smpl.png'),
                            pred_mesh_img)

            # 保存并渲染2d joint
            # for i, pred_joint2d in enumerate(pred_joint2ds):
            #     img_with_joints = vis_points(ori_img.copy(), pred_joint2d, color=(0, 0, 255), radius=5, thickness=2)[:img_size[0], :img_size[1]]
            #     cv2.imwrite(os.path.join(results_save_path, f'{img_name}_{i}_joints.png'),
            #                 img_with_joints)
            #     joint_txt_path = os.path.join(results_save_path, f"{img_name}_{i}_joints.txt")
            #     pred_joint2d_tolist = pred_joint2d.tolist()
            #     joint_json = json.dumps(pred_joint2d_tolist)
            #     with open(joint_txt_path, 'w') as f:
            #         f.write(joint_json)

            # 保存并渲染边界框
            pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
            pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
            pred_box_imgs = vis_boxes(ori_img.copy(), pred_boxes, color=(255, 0, 255))[:img_size[0], :img_size[1]]

            for i, box in enumerate(pred_boxes):
                box_list = []
                # 获取原边界框的宽高
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                # 计算宽高各5%的扩展量（每个方向向外扩展2.5%，总尺寸扩大5%）
                enlarge_x = w * 0.025  # 横向总扩大5% (左缩2.5%，右扩2.5%)
                enlarge_y = h * 0.025  # 纵向总扩大5% (上缩2.5%，下扩2.5%)

                # 调整边界框
                box = torch.tensor([
                    x1 - enlarge_x,
                    y1 - enlarge_y,
                    x2 + enlarge_x,
                    y2 + enlarge_y
                ])

                # 限制边界在图像范围内
                box[0] = torch.clamp(box[0], min=0)
                box[1] = torch.clamp(box[1], min=0)
                box[2] = torch.clamp(box[2], max=img_size[1])  # 假设 img_size[1] 是图像宽度
                box[3] = torch.clamp(box[3], max=img_size[0])  # 假设 img_size[0] 是图像高度

                box_list.append(box)
                pred_box_img = vis_boxes(ori_img.copy(), box_list, color=(255, 0, 255))[:img_size[0], :img_size[1]]
                cv2.imwrite(os.path.join(results_save_path, f'{img_name}_{i}_box.png'),
                            pred_box_img)
                box_txt_path = os.path.join(results_save_path, f"{img_name}_{i}_box.txt")
                box_tolist = box.tolist()
                box_json = json.dumps(box_tolist)
                with open(box_txt_path, 'w') as f:
                    f.write(box_json)

            cv2.imwrite(os.path.join(results_save_path, f'{img_name}_vis.png'),
                        np.vstack([np.hstack([pred_box_imgs, pred_meshes_img]),
                                   np.hstack([pred_scale_img, sat_img])]))

        progress_bar.update(1)
