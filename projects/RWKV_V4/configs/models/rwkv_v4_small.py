from libai.config import LazyCall
from projects.RWKV_V4.rwkv_v4_model import RWKV_V4

rwkv_v4_cfg = dict(
    vocab_size=50304,
    dim=768,
    depth=12,
    dim_head=64,
    num_heads=12,
    ffn_mult=4,
    initializer_range=0.02,
    layernorm_eps=1e-12,
    amp_enabled=False,
)

model = LazyCall(RWKV_V4)(cfg=rwkv_v4_cfg)
