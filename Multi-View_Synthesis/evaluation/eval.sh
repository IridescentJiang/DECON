# # zero123
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/zero123 --dataset wild --name zero123
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/zero123 --dataset shhq --name zero123

# # zero123-xl
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/zero123-xl --dataset wild --name zero123-xl
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/zero123-xl --dataset shhq --name zero123-xl

# # sv3d
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/sv3d --dataset wild --name sv3d
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/sv3d --dataset shhq --name sv3d

# # syncdreamer
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/syncdreamer --dataset wild --name syncdreamer
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/syncdreamer --dataset shhq --name syncdreamer

# # wonder3d
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/wonder3d --dataset wild --name wonder3d
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/wonder3d --dataset shhq --name wonder3d

# # wonder3d_trained
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/wonder3d_trained --dataset wild --name wonder3d_trained
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/wonder3d_trained --dataset shhq --name wonder3d_trained

# # champ
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/champ --dataset wild --name champ
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/champ --dataset shhq --name champ

# # animate-anyone
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/animate-anyone --dataset wild --name animate-anyone
# python eval_wild_nvs_baseline.py --pr /apdcephfs_cq10/share_1330077/hexu/NVS-in-the-wild/0_nvs_baseline_wild/animate-anyone --dataset shhq --name animate-anyone

# # w/_predicted_smpl(pixie)
# python eval_nvs_ours.py --dataset thuman --name pixie_gen --mode full
# python eval_nvs_ours.py --dataset customhumans --name pixie_gen --mode full
# python eval_nvs_ours.py --dataset thuman --name pixie_gen --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name pixie_gen --mode ortho

# # w/_predicted_smpl(pymafx)
# python eval_nvs_ours.py --dataset thuman --name pymafx_gen --mode ortho
# python eval_nvs_ours.py --dataset thuman --name pymafx_gen --mode full
# python eval_nvs_ours.py --dataset customhumans --name pymafx_gen --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name pymafx_gen --mode full

# # w/o attention
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@stage1 --mode ortho
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@stage1 --mode full
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@stage1 --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@stage1 --mode full

# w/ temporal attention
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_time --mode ortho
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_time --mode full
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_time --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_time --mode full

# # w/ full attention
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_view --mode ortho
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_view --mode full
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_view --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_view --mode full

# # w/ reverse attention (v3)
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_union_v3 --mode ortho
# python eval_nvs_ours.py --dataset thuman --name gt_gen/nvs_dual_branch@step2_union_v3 --mode full
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_union_v3 --mode ortho
# python eval_nvs_ours.py --dataset customhumans --name gt_gen/nvs_dual_branch@step2_union_v3 --mode full

# in-the-wild只有attention/normal消融需要测指标
# w/o attention
# python eval_wild_nvs_ours.py --name gt_gen/nvs_dual_branch@stage1 --dataset wild
# python eval_wild_nvs_ours.py --name gt_gen/nvs_dual_branch@stage1 --dataset shhq

# # w/ temporal attention
# python eval_wild_nvs_ours.py --name gt_gen/nvs_dual_branch@step2_time --dataset wild
# python eval_wild_nvs_ours.py --name gt_gen/nvs_dual_branch@step2_time --dataset shhq

# # # w/ full attention
# python eval_wild_nvs_ours.py --name gt_gen/nvs_dual_branch@step2_view --dataset wild
# python eval_wild_nvs_ours.py --name gt_gen/nvs_dual_branch@step2_view --dataset shhq

# w/ reverse attention (v3)
python eval_wild_nvs_ours.py --name gt_gen/nvs_dual_branch@step2_union_v3 --dataset wild
python eval_wild_nvs_ours.py --name gt_gen/nvs_dual_branch@step2_union_v3 --dataset shhq