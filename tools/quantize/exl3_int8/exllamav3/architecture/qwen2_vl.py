"""Qwen2-VL: full-attention vision tower with LayerNorm and non-gated MLP."""
from types import SimpleNamespace
import torch
from .qwen2_5_vl import Qwen2_5VLConfig,Qwen2_5VLModel,Qwen2_5VLVisionModel
from ..model.model import Model
from ..modules import Conv,TransformerBlock,Attention,MLP,LayerNorm,Linear,Module
from ..util.rope import RopeSettings,RoPE,RopeStyle


class Qwen2VLConfig(Qwen2_5VLConfig):
    arch_string="Qwen2VLForConditionalGeneration"

    def __init__(self,directory,**kwargs):
        super().__init__(directory,model_classes={"text":Qwen2VLModel,"vision":Qwen2VLVisionModel},**kwargs)

    def read_vision_config(self,c):
        if c.get('model_type') not in (None,'qwen2_vl','qwen2_vl_vision'):
            raise ValueError('Expected Qwen2-VL vision configuration')
        # Original HF checkpoints often omit fields with these Qwen2-VL defaults.
        embed=c.get('embed_dim',1280);heads=c.get('num_heads',16)
        if embed%heads:raise ValueError('Vision width must divide its head count')
        return SimpleNamespace(depth=c.get('depth',32),hidden_size=embed,num_heads=heads,
            head_dim=embed//heads,intermediate_size=int(embed*c.get('mlp_ratio',4)),
            hidden_act=c.get('hidden_act','quick_gelu'),out_hidden_size=c.get('hidden_size',self.hidden_size),
            num_channels=c.get('in_chans',c.get('in_channels',3)),patch_size=c.get('patch_size',14),
            spatial_merge_size=c.get('spatial_merge_size',2),temporal_patch_size=c.get('temporal_patch_size',2),
            rope_theta=c.get('rope_parameters',{}).get('rope_theta',10000),rms_norm_eps=1e-6)


class Qwen2VLModel(Qwen2_5VLModel):
    config_class=Qwen2VLConfig


class Qwen2VLPatchMerger(Module):
    def __init__(self,config,key):
        super().__init__(config,key,None)
        v=config.vision;self.merge_width=v.hidden_size*v.spatial_merge_size**2
        self.norm=LayerNorm(config,key+'.ln_q',layernorm_eps=1e-6,out_dtype=torch.float)
        self.up=Linear(config,key+'.mlp.0',self.merge_width,self.merge_width,pad_to=1,out_dtype=torch.float)
        self.down=Linear(config,key+'.mlp.2',self.merge_width,v.out_hidden_size,pad_to=1,out_dtype=torch.half)
        for m in [self.norm,self.up,self.down]:self.register_submodule(m)

    def weights_numel(self):return sum(m.weights_numel() for m in [self.norm,self.up,self.down])
    def optimizer_targets(self):raise NotImplementedError('Qwen2-VL vision conversion is not validated')

    def forward(self,x,params,out_dtype=None):
        bsz=x.shape[0]
        y=self.norm.forward(x,params).to(torch.half).reshape(bsz,-1,self.merge_width)
        y=torch.nn.functional.gelu(self.up.forward(y,params),approximate='none')
        y=self.down.forward(y.to(torch.half),params)
        return y.to(out_dtype) if out_dtype is not None else y


class Qwen2VLVisionModel(Qwen2_5VLVisionModel):
    def __init__(self,config,**kwargs):
        Model.__init__(self,config,**kwargs);self.config=config;self.caps.update({'image_input':True})
        v=config.vision
        self.modules=[Conv(config,'visual.patch_embed.proj',v.num_channels,v.hidden_size,
            kernel_size=(v.temporal_patch_size,v.patch_size,v.patch_size),flat=True,out_dtype=torch.float)]
        for i in range(v.depth):
            key=f'visual.blocks.{i}'
            self.modules.append(TransformerBlock(config,key,layer_idx=i,
                attn_norm=LayerNorm(config,key+'.norm1',layernorm_eps=1e-6,out_dtype=torch.float),
                attn=Attention(config,key+'.attn',layer_idx=i,hidden_size=v.hidden_size,head_dim=v.head_dim,
                    num_q_heads=v.num_heads,num_kv_heads=v.num_heads,
                    rope_settings=RopeSettings(head_dim=v.head_dim,rope_style=RopeStyle.NEOX),
                    key_fused_qkv='qkv',key_o='proj',use_cu_seqlens=True),
                mlp_norm=LayerNorm(config,key+'.norm2',layernorm_eps=1e-6,out_dtype=torch.float),
                mlp=MLP(config,key+'.mlp',v.hidden_size,v.intermediate_size,key_up='fc1',key_down='fc2',
                    activation_fn=v.hidden_act,pad_to=1,out_dtype=torch.float)))
        self.modules.append(Qwen2VLPatchMerger(config,'visual.merger'))

    def image_attention_layout(self,grid_thw):
        t,h,w=grid_thw;merge=self.config.vision.spatial_merge_size**2
        return torch.arange(t*h*w//merge,dtype=torch.long),[i*h*w for i in range(t+1)]
