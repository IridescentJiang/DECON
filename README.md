This is the code for "**DECON: Reconstruction of Clothed-Geometric Multiple Humans from a Single Image via Geometry-Guided Decoupling**".



## Template

Save visualization results for each step to facilitate review.



## Human SMPL Estimation

**Human pose estimation** uses [SAT-HMR: Real-Time Multi-Person 3D Mesh Estimation via Scale-Adaptive Tokens](https://github.com/ChiSu001/SAT-HMR)

**Source code** is located in `Multi-View_Synthesis/Geo_Gui_Hm_Decou/Human_SMPL_Estimation`

We use the **official training weights **：

1. Download SMPL-related weights.
   - Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicModel_m_lbs_10_207_0_v1.0.0.pkl`, and `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) (female & male) and [here](http://smplify.is.tue.mpg.de/) (neutral) to `Human_SMPL_Estimation/weights/smpl_data/smpl`. Please rename them as `SMPL_FEMALE.pkl`, `SMPL_MALE.pkl`, and `SMPL_NEUTRAL.pkl`, respectively.
   - Download others from [Google drive](https://drive.google.com/drive/folders/1wmd_pjmmDn3eSl3TLgProgZgCQZgtZIC?usp=sharing) and put them to `Human_SMPL_Estimation/weights/smpl_data/smpl`.
2. Download DINOv2 pretrained weights from [their official repository](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models). We use `ViT-B/14 distilled (without registers)`. Please put `dinov2_vitb14_pretrain.pth` to `Human_SMPL_Estimation/weights/dinov2`. These weights will be used to initialize our encoder. **You can skip this step if you are not going to train SAT-HMR.**
3. Download pretrained weights for inference and evaluation from [Google drive](https://drive.google.com/drive/folders/1L09zt5lQ2RVK2MS2DwKODpdTs9K6CQPC?usp=sharing) or [🤗HuggingFace](https://huggingface.co/ChiSu001/SAT-HMR/tree/main/weights/sat_hmr). Please put them to `Human_SMPL_Estimation/weights/sat_hmr`.



## Image Segmentation

**Image segmentation** uses [SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)

**Source code** is located in `Multi-View_Synthesis/Geo_Gui_Hm_Decou/Img_Segmentation`

We use the **official training weights **：

All the model checkpoints can be downloaded by running:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```



## Human Image Inpainting

**Human image inpainting** employs [PowerPaint: A Versatile Image Inpainting Model](https://github.com/open-mmlab/PowerPaint)

**Source code** is located in `Multi-View_Synthesis/Geo_Gui_Hm_Decou/Img_Inpainting`

We use the **official training weights **：

```bash
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint_v2/ ./checkpoints/ppt-v2

python app.py --share --version ppt-v2 --checkpoint_dir checkpoints/ppt-v2
```





## Multi-View Synthesis Reconstruction

**Novel view synthesis** can be found in the `Multi-View_Synthesis` directory

**Reconstruction code** is located in `Multi-View_Synthesis/src/recon`



## Geometry-Guided Decoupling Optimization (GGDO)
See the `Multi-View_Synthesis/src/ggdo` directory for details



## Perspective-Aware Position Optimization (PAPO)

See the `Multi-View_Synthesis/src/papo` directory for details



## Post-processing

 **SMPL parts replacement** 

See the ``Post-processing/SMPL_parts_replacement`` directory for details

**mesh refinement**

See the ``Post-processing/mesh_refinement`` directory for details