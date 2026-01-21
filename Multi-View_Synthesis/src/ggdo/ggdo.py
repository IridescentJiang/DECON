import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import trimesh

from PIL import Image

from lib.options import BaseOptions
from lib.data import SmplOptDataset_Multi
from lib.model.vol.Voxelize import Voxelization
from lib.model.vol import util as util, constant as const
from lib.model.vol.TetraSmpl import TetraSMPL
from lib.mesh_util import projection

from rendering_script.Render import Render

seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0

generate_for_inference = True

if generate_for_inference:
    batch_size = 1
else:
    batch_size = 2

test_script_activate = True  # Set to True to generate the test subject meshes

current_path = os.path.dirname(os.path.abspath(__file__))

generate_smpl_model = False
generate_smpl_para = False
calculate_cd_p2s = False  # only worker when generate_smpl_model is True

if generate_for_inference:
    calculate_cd_p2s = False


def calculate_rescaled_size(original_size, max_long_edge=512):
    """
    保持宽高比缩放，确保最长边等于max_long_edge
    """
    w, h = original_size
    scale = max_long_edge / max(w, h)

    # 如果原始尺寸已经小于等于最大边长，则保持原尺寸
    if scale >= 1.0:
        return (w, h)

    # 计算新尺寸并保持整数
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return (new_w, new_h)

class Smpl_optimizer(object):
    def __init__(self, device, faces, size):
        super(Smpl_optimizer, self).__init__()
        self.device = device

        self.smpl_faces = faces

        basicModel_path = os.path.join(current_path,
                                       "../lib/model/vol/smpl_data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
        tetra_smpl_path = os.path.join(current_path, "../lib/model/vol/smpl_data/tetra_smpl.npz")
        self.tet_smpl = TetraSMPL(basicModel_path,
                                  tetra_smpl_path).to(self.device)


        self.render_rescale = Render(size=calculate_rescaled_size(size), device=device)
        self.render_ori_scale = Render(size=size, device=device)

    def optm_smpl_param_by_depth(self, pose_transl, shape, iter_num, name,
                                 rotation_matrix, mask, opt, calib):
        assert iter_num > 0

        bs = pose_transl.size()[0]
        pose = pose_transl[:, :24, :]
        transl = pose_transl[:, 24:25, :]
        pose = pose.view(bs, 72)

        theta_new = torch.nn.Parameter(pose)
        theta_new_limb = []
        theta_new_limb.append(torch.nn.Parameter(pose[:, 0:3]))  # orient_global
        theta_new_limb.append(torch.nn.Parameter(pose[:, 3:9]))  # thigh
        theta_new_limb.append(torch.nn.Parameter(pose[:, 12:18]))  # knees
        theta_new_limb.append(torch.nn.Parameter(pose[:, 39:45]))  # shoulders
        theta_new_limb.append(torch.nn.Parameter(pose[:, 48:60]))  # arms

        beta_new = torch.nn.Parameter(shape)
        transl_new = torch.nn.Parameter(transl)

        theta_orig = theta_new.clone().detach()
        transl_orig = transl_new.clone().detach()

        parameters_mask = [transl_new]

        optimizer_smpl_mask = torch.optim.Adam(parameters_mask, lr=3e-2, amsgrad=True)

        scheduler_smpl_mask = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl_mask,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=5,
        )

        # parameters_mask_limb = [theta_new_limb[i] for i in range(5)] + [beta_new]
        parameters_mask_limb = [theta_new] + [beta_new]

        optimizer_smpl_mask_limb = torch.optim.Adam(parameters_mask_limb, lr=2e-3, amsgrad=True)

        scheduler_smpl_depth = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl_mask_limb,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=5,
        )

        pos_weight = 1 / 2
        pbar_mask_pos = tqdm(range(int(iter_num * pos_weight)), desc='Smpl Mask Position Optimizing:')
        pbar_mask_limb = tqdm(range(int(iter_num * (1 - pos_weight))), desc='Smpl Mask Limb Optimizing:')

        best_loss_fitting_first = torch.inf
        best_loss_fitting_second = torch.inf
        best_i = -1

        refer_mask = mask * 100

        for i in pbar_mask_pos:
            theta = theta_new

            with torch.no_grad():
                transl_new.data[:, :, 2] = transl_orig[:, :, 2]
            transl_new_ = transl_new

            verts, faces = self.get_smpl_verts_face(pose=theta, shape=beta_new,
                                                    transl=transl_new_,
                                                    rotation_matrix=rotation_matrix, calib=calib)

            self.render_rescale.load_meshes(verts, faces)

            mask_F, mask_B = self.rendering_smpl_mask(self.render_rescale)

            if i % (iter_num / 10) == 0 or i == (iter_num - 1):
                save_path_diff_mask_depth = '%s/%s/opt_smpl_diff_%s_1_%s.png' % (
                    opt.results_path, opt.name, name, i)
                diff_mask = abs(mask_F.squeeze(0) - refer_mask.squeeze(0).squeeze(0))
                diff_mask_output = diff_mask.cpu().detach().numpy()
                diff_mask_output = diff_mask_output + 255 / 4
                diff_mask_output = diff_mask_output.astype(np.uint8)
                diff_mask_output = Image.fromarray(diff_mask_output, 'L')
                diff_mask_output.save(save_path_diff_mask_depth)

            loss_mask_fitting = torch.mean(torch.abs(mask_F - refer_mask))

            loss = loss_mask_fitting
            pbar_mask_pos.set_postfix(loss='TL: {0:.06f}'.format(loss))

            # 判断是否找到更好的解
            if loss < best_loss_fitting_first:
                best_i = i
                best_loss_fitting_first = loss
                best_theta_new = theta.clone()

            optimizer_smpl_mask.zero_grad()

            loss.backward()

            optimizer_smpl_mask.step()
            scheduler_smpl_mask.step(loss)

        for i in pbar_mask_limb:
            theta = theta_new

            # theta = torch.cat([theta_new_limb[0], theta_new_limb[1], theta_orig[:, 9:12],
            #                    theta_new_limb[2], theta_orig[:, 18:39],
            #                    theta_new_limb[3], theta_orig[:, 45:48],
            #                    theta_new_limb[4], theta_orig[:, 60:]], dim=1)

            transl_new_ = torch.cat([transl_new[:, :, 0:2], transl_orig[:, :, 2].unsqueeze(2)], dim=2)
            # transl_new_ = transl_new

            verts, faces = self.get_smpl_verts_face(pose=theta, shape=beta_new,
                                                    transl=transl_new_,
                                                    rotation_matrix=rotation_matrix,
                                                    calib=calib)

            self.render_rescale.load_meshes(verts, faces)

            mask_F, mask_B = self.rendering_smpl_mask(self.render_rescale)

            if i % (iter_num / 10) == 0 or i == (iter_num - 1):
                save_path_diff_mask_depth = '%s/%s/opt_smpl_diff_%s_2_%s.png' % (
                    opt.results_path, opt.name, name, i)
                diff_mask = abs(mask_F.squeeze(0) - refer_mask.squeeze(0).squeeze(0))
                diff_mask_output = diff_mask.cpu().detach().numpy()
                diff_mask_output = diff_mask_output + 255 / 4
                diff_mask_output = diff_mask_output.astype(np.uint8)
                diff_mask_output = Image.fromarray(diff_mask_output, 'L')
                diff_mask_output.save(save_path_diff_mask_depth)

            loss_mask_fitting = torch.mean(torch.abs(mask_F - refer_mask))
            loss_bias = torch.mean((theta_orig - theta) ** 2)

            loss_mask_fitting_weight = 1
            loss_bias_weight = 50

            loss = loss_mask_fitting * loss_mask_fitting_weight + loss_bias * loss_bias_weight
            pbar_mask_limb.set_postfix(loss='TL: {0:.06f}, ML: {1:.06f}, LB: {2:.06f}'.format(loss, loss_mask_fitting, loss_bias))

            # 判断是否找到更好的解
            if loss < best_loss_fitting_second:
                best_i = i
                best_loss_fitting_second = loss
                best_theta_new = theta.clone()

            optimizer_smpl_mask_limb.zero_grad()

            loss.backward()

            optimizer_smpl_mask_limb.step()
            scheduler_smpl_depth.step(loss)

        print('BL: {0:.06f}, {1:d}'.format(best_loss_fitting_second, best_i))

        # save the final_smpl_mask
        verts, faces = self.get_smpl_verts_face(pose=best_theta_new, shape=beta_new,
                                                transl=transl_new_,
                                                rotation_matrix=rotation_matrix, calib=calib)

        self.render_ori_scale.load_meshes(verts, faces)

        mask_F, mask_B = self.rendering_smpl_mask(self.render_ori_scale)

        save_path_final_smpl_mask = '%s/%s/%s_smpl.png' % (
            opt.results_path, opt.name, name)
        smpl_mask = abs(mask_F.squeeze(0))
        smpl_mask_output = smpl_mask.cpu().detach().numpy()
        smpl_mask_output = smpl_mask_output.astype(np.uint8)
        smpl_mask_output = Image.fromarray(smpl_mask_output, 'L')
        smpl_mask_output.save(save_path_final_smpl_mask)

        best_theta_new = best_theta_new.reshape(bs, 24, 3)

        return best_theta_new, beta_new, transl_new_

    def get_smpl_verts_face(self, pose, shape, transl, rotation_matrix, calib):
        vert_tetsmpl_new_cam = self.tet_smpl(pose, shape)
        vert_tetsmpl_new_cam = self.tet_smpl.rot_vert(vert_tetsmpl_new_cam, transl.squeeze(0))

        verts = vert_tetsmpl_new_cam[0]
        rotation_inv = np.linalg.inv(rotation_matrix.cpu().detach().numpy())
        rotation_matrix = torch.from_numpy(rotation_inv).to(self.device)
        verts = torch.matmul(verts, rotation_matrix.t())
        verts = projection(verts, calib)
        verts[:, 1] *= -1
        verts = verts.unsqueeze(0)

        faces = self.smpl_faces
        bs = verts.size()[0]
        faces = np.expand_dims(faces, axis=0).repeat(bs, axis=0)
        faces = torch.from_numpy(faces).to(self.device)

        return verts, faces

    def rendering_smpl_mask(self, render):

        # F_mask, B_mask = render.get_image(type="mask", scale=0.85, render_type="Perspective")
        F_mask, B_mask = render.get_image(type="mask", scale=0.15, render_type="Perspective")
        F_mask = F_mask.squeeze(0) * 100

        return F_mask, B_mask

def optimize(opt):
    global gen_test_counter
    global lr

    if torch.cuda.is_available():
        # set cuda
        device = 'cuda:0'

    else:
        device = 'cpu'

    print("using device {}".format(device))

    if generate_for_inference:
        train_dataset = SmplOptDataset_Multi.SmplOptInferenceDataset(opt)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print('train loader size: ', len(train_data_loader))

    smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = util.read_smpl_constants('./smpl_data')
    voxelization = Voxelization(smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras,
                                volume_res=const.vol_res,
                                sigma=const.semantic_encoding_sigma,
                                smooth_kernel_size=const.smooth_kernel_size,
                                batch_size=1)

    if (not os.path.exists(opt.checkpoints_path)):
        os.makedirs(opt.checkpoints_path)
    if (not os.path.exists(opt.results_path)):
        os.makedirs(opt.results_path)
    if (not os.path.exists('%s/%s' % (opt.checkpoints_path, opt.name))):
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name))
    if (not os.path.exists('%s/%s' % (opt.results_path, opt.name))):
        os.makedirs('%s/%s' % (opt.results_path, opt.name))

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    # optimizering script

    from evaluate_model import quick_get_chamfer_and_surface_dist

    print('generate smpl (test) ...')
    train_dataset.is_train = False

    len_to_iterate = len(train_dataset) // 1

    total_chamfer_distance = []
    total_point_to_surface_distance = []
    num_samples_to_use = 5000

    for gen_idx in tqdm(range(len_to_iterate)):

        index_to_use = gen_test_counter % len(train_dataset)
        gen_test_counter += 1  # 11 is the number of images for each class
        train_data = train_dataset.get_item(index=index_to_use)
        save_path = '%s/%s/test_%s.obj' % (
            opt.results_path, opt.name, train_data['name'])
        result_file_path = '%s/%s/result.txt' % (opt.results_path, opt.name)
        result_file = open(result_file_path, 'a')

        name = train_data['name']
        id = train_data['id']
        mask = train_data['mask_low_pifu'].to(device=device)
        smpl_shapes_rect = train_data['smpl_shapes_rect'].to(device=device)
        smpl_poses_rect = train_data['smpl_poses_rect'].to(device=device)
        rotation_matrix = train_data['rotation_matrix'].to(device=device)
        calib = train_data['calib'].to(device=device)
        size = train_data['img_size']

        smpl_optimizer = Smpl_optimizer(device, smpl_faces, size)

        id = str(id)

        smpl_shapes_rect = smpl_shapes_rect.unsqueeze(0)
        smpl_poses_rect = smpl_poses_rect.unsqueeze(0)
        mask = mask.unsqueeze(0)

        smpl_rect_vs, smpl_rect_faces = voxelization.para_to_smpl_faces(pose=smpl_poses_rect,
                                                                        shape=smpl_shapes_rect)
        smpl_rect_vs = smpl_rect_vs.squeeze(0).cpu().detach().numpy()
        smpl_rect_vs *= 2
        smpl_rect_faces = smpl_rect_faces.squeeze(0).cpu().detach().numpy()

        if generate_smpl_model:
            save_smpl_ori_path = save_path[:-4] + '_' + id + '_smpl_ori.obj'
            with open(save_smpl_ori_path, 'w') as fp:
                for i, v in enumerate(smpl_rect_vs):
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                for t in smpl_rect_faces + 1:
                    fp.write('f %d %d %d\n' % (t[0] + 1, t[2] + 1, t[1] + 1))
                    fp.write('f %d %d %d\n' % (t[0] + 1, t[3] + 1, t[2] + 1))
                    fp.write('f %d %d %d\n' % (t[0] + 1, t[1] + 1, t[3] + 1))
                    fp.write('f %d %d %d\n' % (t[1] + 1, t[2] + 1, t[3] + 1))

        # generating smpl para
        pose, shape, transl = smpl_optimizer.optm_smpl_param_by_depth(pose_transl=smpl_poses_rect,
                                                                      shape=smpl_shapes_rect,
                                                                      iter_num=200,
                                                                      name=name,
                                                                      rotation_matrix=rotation_matrix,
                                                                      mask=mask,
                                                                      opt=opt,
                                                                      calib=calib)


        # 保存数据到 JSON 文件

        if generate_smpl_para:
            save_smpl_opt_para_path = save_path[:-4] + '_' + id + '_smpl_opt_para.json'
            data = {
                "poses": pose.tolist(),
                "betas": shape.tolist(),
                "transl": transl.tolist()
            }
            with open(save_smpl_opt_para_path, 'w') as file:
                json.dump(data, file, indent=4)

        # generating smpl model
        if generate_smpl_model:

            pose = torch.cat((pose, transl), dim=1)

            smpl_opt_vs, smpl_opt_faces = voxelization.para_to_smpl_faces_reverse(pose=pose, shape=shape * 0.1)

            smpl_opt_vs = smpl_opt_vs.squeeze(0).cpu().detach().numpy()
            smpl_opt_vs *= 2
            smpl_opt_faces = smpl_opt_faces.squeeze(0).cpu().detach().numpy()

            save_smpl_opt_path = save_path[:-4] + '_' + id + '_smpl_opt.obj'
            with open(save_smpl_opt_path, 'w') as fp:
                for i, v in enumerate(smpl_opt_vs):
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                for t in smpl_opt_faces + 1:
                    fp.write('f %d %d %d\n' % (t[0] + 1, t[2] + 1, t[1] + 1))
                    fp.write('f %d %d %d\n' % (t[0] + 1, t[3] + 1, t[2] + 1))
                    fp.write('f %d %d %d\n' % (t[0] + 1, t[1] + 1, t[3] + 1))
                    fp.write('f %d %d %d\n' % (t[1] + 1, t[2] + 1, t[3] + 1))

            subject = train_data['name']
            subject = subject.replace('.obj', '')
            if not generate_for_inference:
                smpl_directory = "rendering_script/smpl_gt"
                GT_mesh = trimesh.load(os.path.join(smpl_directory, subject, "smpl_{0:03d}.obj".format(train_data['id'])))
                save_smpl_gt_path = save_path[:-4] + '_' + id + '_smpl_gt.obj'
                GT_mesh.export(save_smpl_gt_path)

        if calculate_cd_p2s and generate_smpl_model:
            try:
                source_mesh = trimesh.load(save_smpl_opt_path)

                chamfer_distance, point_to_surface_distance = quick_get_chamfer_and_surface_dist(
                    src_mesh=source_mesh, tgt_mesh=GT_mesh, num_samples=num_samples_to_use)
                print('{0} - CD: {1:.08f} P2S: {2:.08f}'.format(train_data['name'], chamfer_distance,
                                                                point_to_surface_distance), file=result_file)
                print('{0} - CD: {1:.08f} P2S: {2:.08f}'.format(train_data['name'], chamfer_distance,
                                                                point_to_surface_distance))
                total_chamfer_distance.append(chamfer_distance)
                total_point_to_surface_distance.append(point_to_surface_distance)
            except:
                print('Unable to compute chamfer_distance and/or point_to_surface_distance!', file=result_file)
                print('Unable to compute chamfer_distance and/or point_to_surface_distance!')

    if calculate_cd_p2s and generate_smpl_model:
        if len(total_chamfer_distance) == 0:
            average_chamfer_distance = 0
        else:
            average_chamfer_distance = np.mean(total_chamfer_distance)

        if len(total_point_to_surface_distance) == 0:
            average_point_to_surface_distance = 0
        else:
            average_point_to_surface_distance = np.mean(total_point_to_surface_distance)

        print("[Testing] Overall - Avg CD: {0:.08f}; Avg P2S: {1:.08f}".format(average_chamfer_distance,
                                                                               average_point_to_surface_distance),
              file=result_file)
        print("[Testing] Overall - Avg CD: {0:.08f}; Avg P2S: {1:.08f}".format(average_chamfer_distance,
                                                                               average_point_to_surface_distance))

    print("Optimizing is Done! Exiting...")
    sys.exit()


if __name__ == '__main__':
    optimize(opt)
