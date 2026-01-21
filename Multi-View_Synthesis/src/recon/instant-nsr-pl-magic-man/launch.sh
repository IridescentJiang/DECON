#!/bin/bash

# 设置root_dir和trial_name变量
root_dir="../recon"
trial_name="recon@test"

# start_from="0041_00015_12_00121"
# end_at="0515"
# device=0
start_from="2-1_0_result"
end_at="5_2_result"
device=0

# export CUDA_VISIBLE_DEVICES=1 这里没用！！！在下面的python命令里改gpu
## 改上面

# 循环遍历root_dir下的所有子目录
for subject in $(ls -v "$root_dir"); do
    if [ "$subject" == "$start_from" ]; then
        processing=true
    fi
    # 从start_from开始
    if [ "$processing" = true ]; then
    # 检查是否为目录
        if [ -d "$root_dir/$subject" ]; then
            # 获取subject目录名
            subject_name=$(basename "$subject")

            # 打印subject名称
            echo "Processing subject: $subject_name"

            # 运行python命令
            CUDA_LAUNCH_BLOCKING=1 \
            python launch.py --config configs/human-nvs.yaml --gpu $device \
            --train --scene "$subject_name" \
            dataset.scene="$subject_name" \
            trial_name="$trial_name" \
            dataset.ref_dir="$root_dir/$subject_name/ref" \
            dataset.cond_dir="$root_dir/$subject_name/cond/gt" \
            dataset.mv_dir="$root_dir/$subject_name/$trial_name"
        fi
    fi

    if [ "$subject" == "$end_at" ]; then
        break
    fi

done


# root_dir="/apdcephfs_cq10/share_1330077/hexu/NVS/test_set"

# echo "Listing subjects in natural order:"
# for subject in $(ls -v "$root_dir"); do
#     echo "$subject"
# done
