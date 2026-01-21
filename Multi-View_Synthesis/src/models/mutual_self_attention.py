# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional

import torch
from einops import rearrange

from src.models.attention import TemporalBasicTransformerBlock

from .attention import BasicTransformerBlock


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

def torch_named_dfs(model: torch.nn.Module, name: str='unet'): # 返回实例的同时返回名称
    result = [(name, model)]
    for child_name, child_model in model.named_children():
        result += torch_named_dfs(child_model, f'{name}.{child_name}')
    return result

# 初始化ReferenceAttentionControl的时候，里面包含了一个Unet（ref或denoise）
# 根据mode的不同，初始化的时候给unet里面的basicTransformerBlock添加hook，绑定新的forward
# 使得在forward的时候，可以根据mode的不同，对hidden_states进行不同的处理
# 1. 如果是write，将hidden_states添加到bank中
# 2. 如果是read，将bank中的hidden_states拼接到hidden_states中，然后进行forward
class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read_cross_attn", "read_concat_attn", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            reference_attn,
            batch_size=batch_size,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        reference_attn=True,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        reference_attn = reference_attn # True
        num_images_per_prompt = num_images_per_prompt
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else: # 这里
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2) #2
                .to(device)
                .bool()
            ) # [0,0]

        def hacked_basic_transformer_inner_forward(
            self, # 这里的self是BasicTransformerBlock或者TemporalBasicTransformerBlock 被绑定了
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
        ):
            # 1. Self-Attention
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            # write read的区别只在于attn1 这里判断选择不同的attn1  之后统一走attn2和forward等操作
            if MODE == "write":
                # attn0(self attn)
                norm_hidden_states = self.norm0(hidden_states) # b l c
                hidden_states  = (
                    self.attn0(
                    norm_hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                ) 
                    + hidden_states 
                )
                # attn1 (self attn)
                norm_hidden_states = self.norm1(hidden_states)
                self.bank.append(norm_hidden_states.clone())
                hidden_states  = (
                    self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=None, # None
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                ) 
                    + hidden_states 
                )
            elif "read" in MODE: # ["read_concat_attn", "read_cross_attn"] 
                # 用统一的索引走attn0(multi/single self attn),
                # 区分单纯cross或者concat的attn1
                # 走一样的clip attn2（实际上都删掉了）
                
                ''' attn0 index决定的single/multi self attn 传入的是(b f) (l=hw) c的hidden_states'''
                norm_hidden_states = self.norm0(hidden_states)
                
                attention_index = self.attention_index * video_length //360 # 把角度形式转换成帧索引
                attention_index_matrix = torch.arange(video_length).unsqueeze(1) + attention_index  # stage1是1x1 stage2是FxS
                                                                                                            # F是总帧数 S是采样帧数
                attention_index_matrix = attention_index_matrix % video_length  # 取模循环
                norm_hidden_states = rearrange(norm_hidden_states, "(b f) l c -> b l f c", f=video_length)
                sample_norm_hidden_states = norm_hidden_states[:, :, attention_index_matrix, :] # b l f s c
                
                norm_hidden_states = rearrange(norm_hidden_states, "b l f c -> (b f) l c")
                sample_norm_hidden_states = rearrange(sample_norm_hidden_states, "b l f s c -> (b f) (l s) c")
                
                hidden_states  = (
                    self.attn0(
                    norm_hidden_states,
                    encoder_hidden_states=sample_norm_hidden_states, # 和sample的做cross attn实现了multi/single view
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                ) 
                    + hidden_states 
                ) # (b f) l c
                
                ''' 区分attn1 # 每个basicTransBlock里面的bank存了一个feature map，就是norm之后、过selfatt之前的特征图'''
                if MODE == "read_concat_attn":  # concat的情况 
                    # attn1 ref-attn
                    norm_hidden_states = self.norm1(hidden_states)
                    bank_fea = [
                        rearrange(
                            d.unsqueeze(1).repeat(1, video_length, 1, 1), # stage1 video_length=1 
                            "b t l c -> (b t) l c",                         # stage2 b中每个样本室视频，有length，但是都用同一个ref_feat，所以要repeat
                        )
                        for d in self.bank
                    ]
                    modify_norm_hidden_states = torch.cat(
                        [norm_hidden_states] + bank_fea, dim=1
                    ) # (bt) l c 后按照l维度拼接 (bt) 2l c 
                    # modify_norm_hidden_states = bank_fea[0] # 不拼接 直接和ref feature map做cross attn，cfg要改，比较麻烦
                    hidden_states_uc = ( # 训练都走这里，cond的时候bank里有东西，拼接att。uncond的时候bank空，没拼东西，等于self att （如果不用拼接直接改cross attn这里要改空 比较麻烦）
                        self.attn1( # 推理走这里的时候，因为推理前一定update了bank，所以这里一定是拼接kv
                            norm_hidden_states, # 原始特征图做q
                            encoder_hidden_states=modify_norm_hidden_states, # 拼接后的结果做kv
                            attention_mask=attention_mask,
                        )
                        + hidden_states
                    )
                    if do_classifier_free_guidance: # 推理的cfg都是true，所以走这里
                        hidden_states_c = hidden_states_uc.clone() ### ***
                        _uc_mask = uc_mask.clone()
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor(
                                    [1] * (hidden_states.shape[0] // 2)
                                    + [0] * (hidden_states.shape[0] // 2)
                                )
                                .to(device)
                                .bool()
                            )
                        # 这里_uc_mask用于选择前半batch的样本，前半无条件做self att，不拼接；后半在上面的***行沿用了拼接结果
                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=norm_hidden_states[_uc_mask], # 这里是self att 不拼接
                                attention_mask=attention_mask,
                            )
                            + hidden_states[_uc_mask]
                        )
                        hidden_states = hidden_states_c.clone()
                    else: # 训练走这里
                        hidden_states = hidden_states_uc
                        
                if MODE == "read_cross_attn":  # 单纯cross的情况
                    # attn1 ref-attn
                    norm_hidden_states = self.norm1(hidden_states)
                    if len(self.bank) > 0: # 也就是没有drop掉参考图的时候
                        assert len(self.bank) == 1 # 暂时只考虑单张参考图     
                        bank_fea = [
                            rearrange(
                                d.unsqueeze(1).repeat(1, video_length, 1, 1), # stage1 video_length=1 
                                "b t l c -> (b t) l c",                         # stage2 b中每个样本室视频，有length，但是都用同一个ref_feat，所以要repeat
                            )
                            for d in self.bank
                        ]
                        cross_norm_hidden_states = bank_fea[0] # (bt) l c 直接用ref的特征图 不拼接
                    else:
                        cross_norm_hidden_states = torch.zeros_like(norm_hidden_states) # cfg的时候没有参考图，用一张全0的空白图 
                        
                    # modify_norm_hidden_states = bank_fea[0] # 不拼接 直接和ref feature map做cross attn，cfg要改，比较麻烦
                    hidden_states_uc = ( # 训练都走这里，cond的时候bank里有东西，拼接att。uncond的时候bank空，没拼东西，等于self att （如果不用拼接直接改cross attn这里要改空 比较麻烦）
                        self.attn1( # 推理走这里的时候，因为推理前一定update了bank，所以这里一定是拼接kv
                            norm_hidden_states, # 原始特征图做q
                            encoder_hidden_states=cross_norm_hidden_states, # 拼接后的结果做kv
                            attention_mask=attention_mask,
                        )
                        + hidden_states
                    )
                    if do_classifier_free_guidance: # 推理的cfg都是true，所以走这里
                        hidden_states_c = hidden_states_uc.clone() ### ***
                        _uc_mask = uc_mask.clone()
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor(
                                    [1] * (hidden_states.shape[0] // 2)
                                    + [0] * (hidden_states.shape[0] // 2)
                                )
                                .to(device)
                                .bool()
                            )
                        # 这里_uc_mask用于选择前半batch的样本，前半无条件做self att，不拼接；后半在上面的***行沿用了拼接结果
                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=norm_hidden_states[_uc_mask], # 这里是self att 不拼接
                                attention_mask=attention_mask,
                            )
                            + hidden_states[_uc_mask]
                        )
                        hidden_states = hidden_states_c.clone()
                    else: # 训练走这里
                        hidden_states = hidden_states_uc
                        
            # 不管哪种writeread，都要过cross attn2
            if self.attn2 is not None:
                # Cross-Attention
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm2(hidden_states)
                )
                hidden_states = (
                    self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                    )
                    + hidden_states
                )

            # Feed-forward
            hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

            return hidden_states
            # def forward 结束
            
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, BasicTransformerBlock
                    )
                if isinstance(module, TemporalBasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, TemporalBasicTransformerBlock
                    )

                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))
                
                
    def update(self, writer, dtype=torch.float16):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (
                        torch_dfs(writer.unet.mid_block)
                        + torch_dfs(writer.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full": # 走这里
            ## 这里的named_dfs同时返回名称和实例 方便对照查看两个unet返回的顺序（其实用dfs也可以）
                reader_attn_modules = [
                    (name, module)
                    for name, module in torch_named_dfs(self.unet) # denoise_net
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    (name, module)
                    for name, module in torch_named_dfs(writer.unet) # ref_net
                    if isinstance(module, BasicTransformerBlock)
                ]
            # (name, module)取0是name 取1是module实例
            reader_attn_modules = sorted( # 按照实例的norm1层维度排序(sort是稳定排序 所以key相同的保持原本顺序)
                reader_attn_modules, key=lambda x: -x[1].norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x[1].norm1.normalized_shape[0]
            )
            
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r[1].bank = [v.clone().to(dtype) for v in w[1].bank]
                # w.bank.clear()
            
            ## 原本只返回实例的dfs
            #     reader_attn_modules = [
            #         module
            #         for module in torch_dfs(self.unet) # denoise_net
            #         if isinstance(module, TemporalBasicTransformerBlock)
            #     ]
            #     writer_attn_modules = [
            #         module
            #         for module in torch_dfs(writer.unet) # ref_net
            #         if isinstance(module, BasicTransformerBlock)
            #     ]
            # reader_attn_modules = sorted(
            #     reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            # )
            # writer_attn_modules = sorted(
            #     writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            # )
            # for r, w in zip(reader_attn_modules, writer_attn_modules):
            #     r.bank = [v.clone().to(dtype) for v in w.bank]
            #     # w.bank.clear()

    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            ) # 这里和reader还是writer没关系了，只是代码命名问题。不管是reader还是writer调用clear都会把自己unet里面basicTrans的bank清空。每次iter都需要清空
            for r in reader_attn_modules:
                r.bank.clear()
