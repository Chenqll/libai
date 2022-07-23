from omegaconf import OmegaConf

from libai.config import get_config
from libai.config import LazyCall
from libai.tokenizer import GPT2Tokenizer

# 配置 model
from projects.RWKV_V4.modeling.model import GPT 
# 配置 dataloader `build_image_train_loader` 和 `build_image_test_loader` 是 LiBai 提供的用于创建图像数据的训练集和测试集 DataLoader 的两个函数
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
# 导入自定义的 dataset
from projects.RWKV_V4.dataset import RWKVDataset

graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train
optim = get_config("common/optim.py").optim

# 配置两个file
vocab_file = "/home/zhangxiaoyu/shan/RWKV-LM/RWKV-v4/vocab.json"
merge_files = "/home/zhangxiaoyu/shan/RWKV-LM/libai/projects/RWKV_V4/dataset/gpt2-merges.txt"

tokenization = get_config("common/data/gpt_dataset.py").tokenization
tokenization.tokenizer.vocab_file = vocab_file
tokenization.tokenizer.merges_file = merge_files

optim = get_config("common/optim.py").optim
model_cfg = get_config("common/models/gpt.py").cfg
graph = get_config("common/models/graph.py").graph



data_prefix = "./projects/PaLM/gpt_dataset/loss_compara_content_sentence"

# 配置model
model = LazyCall(GPT)(cfg=model_cfg)
model.cfg.embedding_dropout_prob = 0.1
model.cfg.attention_dropout_prob = 0.1
model.cfg.num_attention_heads = 16
model.cfg.hidden_size = 384
model.cfg.ffn_hidden_size = 1536
model.cfg.num_layers = 6
model.cfg.max_seq_length = 1024

# 训练过程
train = get_config("common/train.py").train
train.input_placement_device = "cpu"
train.dist.pipeline_num_layers = model.cfg.num_layers


# 获得一个 DataLoader 的配置对象
dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_nlp_train_loader)(
    dataset=[
        LazyCall(RWKVDataset)(
            data="/home/zhangxiaoyu/shan/RWKV-LM/data/enwik8",
            ctx_len=1024,
            epoch_length_fixed=9996,
        ),
    ],
    num_workers=4,
)
for ds in dataloader.train.dataset:
    ds.max_seq_length = model.cfg.max_seq_length







