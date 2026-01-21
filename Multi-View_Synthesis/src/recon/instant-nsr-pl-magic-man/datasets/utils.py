import numpy as np
WARNED = False

def get_c2w_from_up_and_look_at(
    up,
    look_at,
    pos,
    opengl=False,
):
    up = up / np.linalg.norm(up)
    z = look_at - pos # z是相机指向物体（原点）
    z = z / np.linalg.norm(z)
    y = -up # y是相机的下方向
    x = np.cross(y, z) # x是相机的右方向
    x /= np.linalg.norm(x)
    y = np.cross(z, x) # 这里的x y应该是图像成像坐标系 z是相机光轴

    c2w = np.zeros([4, 4], dtype=np.float32) # 下面以0度的相机为例
    c2w[:3, 0] = x # 相机的x轴 对应世界010 也就是世界Y轴
    c2w[:3, 1] = y # 相机的y轴 对应世界00-1 也就是世界-Z轴
    c2w[:3, 2] = z # 相机的z轴 对应世界-100 也就是世界-X轴
    c2w[:3, 3] = pos
    c2w[3, 3] = 1.0

    # opencv to opengl
    if opengl:
        c2w[..., 1:3] *= -1

    return c2w


def get_uniform_poses(num_frames=20, radius=2.0, elevation=0.0, opengl=False):
    T = num_frames
    azimuths = np.deg2rad(np.linspace(0, 360, T + 1)[:T]) * (-1.0) # 这里是逆时针旋转
    elevations = np.full_like(azimuths, np.deg2rad(elevation))
    cam_dists = np.full_like(azimuths, radius)

    campos = np.stack(
        [   
            cam_dists * np.cos(elevations) * np.sin(azimuths),
            - cam_dists * np.cos(elevations) * np.cos(azimuths),
            cam_dists * np.sin(elevations),
        ],
        axis=-1,
    )

    center = np.array([0, 0, 0], dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)
    poses = []
    for t in range(T):
        poses.append(get_c2w_from_up_and_look_at(up, center, campos[t], opengl=opengl))

    return [pose[:3, :] for pose in poses] # 只保留RT 3*4