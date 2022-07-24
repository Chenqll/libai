import numpy as np
import oneflow as flow
import oneflow.nn.functional as F
from oneflow import einsum, nn
import math, os
import logging
from libai.config import configurable
from libai.layers import LayerNorm, Linear, LMLogits, ParallelCrossEntropyLoss, VocabEmbedding
from libai.models.utils import init_method_normal
from libai.utils import distributed as dist
logger = logging.getLogger(__name__)

RWKV_HEAD_QK_DIM = 256
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n')

class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
       
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd

        attn_sz = config.n_embd

        with flow.no_grad(): # fancy init
            ratio_0_to_1 = (layer_id / (config.n_layer - 1)) # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer)) # 1 to ~0
            
            # fancy time_decay
            decay_speed = flow.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz-1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = (flow.tensor([(i+1)%3 - 1 for i in range(attn_sz)]) * 0.5)
            self.time_first = nn.Parameter(flow.ones(attn_sz) * math.log(0.3) + zigzag)
            
            # fancy time_mix
            x = flow.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd
            self.time_mix_k = nn.Parameter(flow.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(flow.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(flow.pow(x, 0.5 * ratio_1_to_almost0))
            
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = Linear(config.n_embd, attn_sz, bias=False)
        self.value = Linear(config.n_embd, attn_sz, bias=False)
        self.receptance = Linear(config.n_embd, attn_sz, bias=False)

        self.output = Linear(attn_sz, config.n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size() # x = (Batch,Time,Channel)

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x) # self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        rwkv = flow.sigmoid(r) * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
  
        return rwkv


class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
      
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with flow.no_grad(): # fancy init of time_mix
            ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer)) # 1 to ~0

            x = flow.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd

            self.time_mix_k = nn.Parameter(flow.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(flow.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * config.n_embd
        self.key = Linear(config.n_embd, hidden_sz, bias=False)
        self.receptance = Linear(config.n_embd, config.n_embd, bias=False)
        self.value = Linear(hidden_sz, config.n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = flow.square(flow.relu(k))
        kv = self.value(k)

        rkv = flow.sigmoid(self.receptance(xr)) * kv
        return rkv

########################################################################################################
# The GPT Model with our blocks
########################################################################################################


class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = LayerNorm(config.n_embd)
        self.ln2 = LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = LayerNorm(config.n_embd)

        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(config, layer_id+1000)
        else:
            self.att = RWKV_TimeMix(config, layer_id)

        self.ffn = RWKV_ChannelMix(config, layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)        
        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            x = x + self.ffnPre(self.ln1(x))  # better in some cases
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.step = 0
        self.config = config

        self.emb = VocabEmbedding(config.vocab_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config, i)
                                    for i in range(config.n_layer)])

        self.ln_out = LayerNorm(config.n_embd)
        self.head = Linear(config.n_embd, config.vocab_size, bias=False)

        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_q.scale_init = 0
            self.head_k = Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
            self.head_k.scale_init = 0.1
            self.register_buffer("copy_mask", flow.tril(
                flow.ones(config.ctx_len, config.ctx_len)))

        self.ctx_len = config.ctx_len

        try:
            if os.environ['RWKV_LOAD_MODEL'] == str(False):
                RWKV_Init(self, config) 
        except:
            pass

        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))

    def get_ctx_len(self):
        return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        no_decay = set()

        for mn, m in self.named_modules():  # here we disable weight_decay
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = Fflow.optim.Optimizer

        return optimizer

    def forward(self, idx, targets=None):
        idx = idx.to(self.emb.weight.device)

        self.step += 1
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)

        if RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
            c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).half()
            x = self.head(x) + c
        else:
            x = self.head(x)

        loss = None
        if targets is not None:
            loss = F.ParallelCrossEntropyLoss(x.view(-1, x.size(-1)), targets.to(x.device).view(-1))

        return x, loss
