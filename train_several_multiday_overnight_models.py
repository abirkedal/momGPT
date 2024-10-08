"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import datetime
import math
import pickle
from contextlib import nullcontext

import numpy as np
import pandas as pd
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import OvnMomGPTConfig, OvnMomGPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 10
log_interval = 1
eval_iters = 1000
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

#important times of day
intraday_start = datetime.time(10,00,00)
intraday_end = datetime.time(15,45,00)
afternoon_start = datetime.time(15,45,00)
isos_split_date = datetime.date(2014,12,31)

# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 1024 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size1 = 4

# model
subtract_ovn_linear = False
drop_any_nan = True
input_clip = 0.5
n_layer = 2
n_head = 1
n_embd = 5
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 15000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
# lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
lr_decay_iters = max_iters
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
print('DDP:', ddp)
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter1 = gradient_accumulation_steps * ddp_world_size * batch_size * block_size1
print(f"tokens per iteration will be: {tokens_per_iter1:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
data = np.load('/home/andreas/momGPT/data/firstratedata/multi_ticker_dict_intraday.npy', allow_pickle='TRUE').item()
data_df = pd.concat(data, axis=0)
data_df = data_df.reset_index(level=[0])
data_df = data_df.rename(columns={'level_0': 'ticker'})

# Other potential input data:
# past volatility, volume, day of week/month/quarter, closeness to earnings announcement/dividend date, closeness to options expiry date, closeness to index rebal date
# We can also try adding something like the mom-rev signal from https://arxiv.org/pdf/1107.0036
pred_scale = 0.0157 # multiplier to put a (0,1) predictor on the scale of overnight returns. Is just the std of overnight_return
data_df['day_qtr'] = (pd.to_datetime(data_df.index) - pd.PeriodIndex(pd.to_datetime(data_df.index), freq='Q').start_time).days + 1
data_df['day_wk'] = pd.to_datetime(data_df.index).dayofweek
data_df['day_yr'] = pd.to_datetime(data_df.index).dayofyear
pred_cols= []
for c in ['day_qtr', 'day_wk', 'day_yr']:
    data_df[c+'_pred'] = pred_scale * (data_df[c] - data_df[c].median())/data_df[c].max()
    pred_cols.append(c+'_pred')
skip_val = len(pred_cols)
return_cols = ['overnight_return', 'into_close_safe_return', 'intraday_return', 'future_overnight_return']

ret_cols = pred_cols + return_cols

block_size1 = len(return_cols)
block_size2 = len(ret_cols)

train_data_df = data_df[data_df.index <= isos_split_date][['ticker'] + ret_cols]
train_data_gb = train_data_df.groupby('ticker')
tickers = list(train_data_gb.groups.keys())

val_data_df = data_df[data_df.index > isos_split_date][['ticker'] + ret_cols]
val_data_gb = val_data_df.groupby('ticker')

train_data_vals_dict = {}
val_data_vals_dict = {}
for ticker in tickers:
    if drop_any_nan:
        train_data_vals_dict[ticker] = train_data_gb.get_group(ticker)[ret_cols].replace(0.0, np.nan).dropna().astype('float').values
        val_data_vals_dict[ticker]= val_data_gb.get_group(ticker)[ret_cols].replace(0.0, np.nan).dropna().astype('float').values
    else:
        train_data_vals_dict[ticker]= train_data_gb.get_group(ticker)[ret_cols].replace(0.0, np.nan).dropna(how='all').fillna(0).astype('float').values
        val_data_vals_dict[ticker]= val_data_gb.get_group(ticker)[ret_cols].replace(0.0, np.nan).dropna(how='all').fillna(0).astype('float').values

if drop_any_nan:
    train_data_df = data_df[data_df.index <= isos_split_date][ret_cols].replace(0.0, np.nan).dropna().fillna(0).astype('float')
    val_data_df = data_df[data_df.index > isos_split_date][ret_cols].replace(0.0, np.nan).dropna().fillna(0).astype('float')
else:
    train_data_df = data_df[data_df.index <= isos_split_date][ret_cols].replace(0.0, np.nan).dropna(how='all').fillna(0).astype('float')
    val_data_df = data_df[data_df.index > isos_split_date][ret_cols].replace(0.0, np.nan).dropna(how='all').fillna(0).astype('float')
train_data = train_data_df.values

corr_data = train_data_df.copy()

corr_data[ret_cols] = corr_data[ret_cols].clip(lower=-input_clip, upper=input_clip)
corr_mat = corr_data.corr()
cov_mat = corr_data.cov()
linear_ind = 1+len(pred_cols)
train_ovn_linear_r2 = (corr_mat.values[linear_ind][-1])**2
train_ovn_linear_beta = cov_mat.values[linear_ind][-1]/cov_mat.values[linear_ind][linear_ind]
train_ovn_linear_r2_vals = []
train_ovn_linear_beta_vals = []
for i in range(len(ret_cols)-1):
    train_ovn_linear_r2_vals.append((corr_mat.values[i][-1])**2)
    train_ovn_linear_beta_vals.append(cov_mat.values[i][-1]/cov_mat.values[i][i])

val_data = val_data_df.values
val_corr_data = val_data_df.copy()
val_corr_data[ret_cols] = val_corr_data[ret_cols].clip(lower=-input_clip, upper=input_clip)
val_ovn_linear_r2 = (val_corr_data[ret_cols].corr().values[linear_ind][-1])**2
val_corr_mat = val_corr_data.corr()
val_cov_mat = val_corr_data.cov()
val_ovn_linear_r2_vals = []
val_ovn_linear_beta_vals = []
for i in range(len(ret_cols)-1):
    val_ovn_linear_r2_vals.append((val_corr_mat.values[i][-1])**2)
    val_ovn_linear_beta_vals.append(val_cov_mat.values[i][-1]/val_cov_mat.values[i][i])

print(train_data.shape, train_ovn_linear_r2_vals, train_ovn_linear_beta_vals, val_ovn_linear_r2_vals, val_ovn_linear_beta_vals)

def get_batch(split, skip=0):
    t = time.time()
    if split == 'train':
        dfd = train_data_vals_dict
    else:
        dfd = val_data_vals_dict
    
    tickers = list(dfd.keys())
    
    ix = torch.randint(len(tickers), (2 * batch_size,))

    xl = []
    yl = []
    zl = []

    xl2 = []
    yl2 = []
    zl2 = []

    t1 = time.time()
    
    count = 0
    for i in ix:
        if count < batch_size:
            ticker = tickers[i]

            data = dfd[ticker]

            ds0 = data.shape[0]
            if ds0 - gptconf1.n_embd > 0:

                j = torch.randint(ds0 - gptconf1.n_embd, (1,))[0]
                xd = np.clip(data[j:(j+gptconf1.n_embd), :-1], -input_clip, input_clip)
                yd = data[j:(j+gptconf1.n_embd), 1:].copy()
                zd = data[j:(j+gptconf1.n_embd), 1:].copy()
                
                if skip > 0:
                    xd2 = np.clip(data[j:(j+gptconf1.n_embd), :-1][:, skip:], -input_clip, input_clip)
                    yd2 = data[j:(j+gptconf1.n_embd), (skip+1):].copy()
                    zd2 = yd2.copy()
                else:
                    xd2 = np.clip(data[j:(j+gptconf1.n_embd), :-1], -input_clip, input_clip)
                    yd2 = data[j:(j+gptconf1.n_embd), 1:].copy()
                    zd2 = data[j:(j+gptconf1.n_embd), 1:].copy()
                if subtract_ovn_linear:
                    yd[:, -1] = yd[:, -1] - train_ovn_linear_beta * xd[:, skip+1]
                    yd2[:, -1] = yd2[:, -1] - train_ovn_linear_beta * xd2[:, 1]
                    
                xl.append(torch.from_numpy((xd)))
                yl.append(torch.from_numpy((yd)))
                zl.append(torch.from_numpy((zd)))
                    
                xl2.append(torch.from_numpy((xd2)))
                yl2.append(torch.from_numpy((yd2)))
                zl2.append(torch.from_numpy((zd2)))
                
                count += 1

    t2 = time.time()
    
    x = torch.stack(xl)
    y = torch.stack(yl)
    x2 = torch.stack(xl2)
    y2 = torch.stack(yl2)
                      
    x = x.bfloat16()
    y = y.bfloat16()
    x2 = x2.bfloat16()
    y2 = y2.bfloat16()
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, x2, y2 = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), x2.pin_memory().to(device, non_blocking=True), y2.pin_memory().to(device, non_blocking=True)
    else:
        x, y, x2, y2 = x.to(device), y.to(device), x2.to(device), y2.to(device)
    if len(zl) > 0:
        z = torch.stack(zl)
        z = z.bfloat16()
        z2 = torch.stack(zl2)
        z2 = z2.bfloat16()
        
        if device_type == 'cuda':
            z = z.pin_memory().to(device, non_blocking=True)
            z2 = z2.pin_memory().to(device, non_blocking=True)
        else:
            z = z.to(device)
            z2 = z2.to(device)

    t3 = time.time()

    if len(zl) > 0:
        return x, y, z, x2, y2, z2
    else:
        return x, y, None, x2, y2, None

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
models = {}
models_args = {}
model1_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size1,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
model2_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size2,
                  bias=bias, vocab_size=None, dropout=dropout)
models_args['model1'] = model1_args
models_args['model2'] = model2_args

print('Using the following model_args:', model1_args, model2_args)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of 1")
    for m in models_args.keys():
        models_args[m]['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 1
    model1_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 1
    model2_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 1
    gptconf1 = OvnMomGPTConfig(**model1_args)
    gptconf2 = OvnMomGPTConfig(**model2_args)
    gptconfigs = {}
    for m in models_args.keys():
        gptconfigs[m] = OvnMomGPTConfig(**models_args[m])
        torch.manual_seed(0)
        models[m] = OvnMomGPT(gptconfigs[m])

elif init_from == 'resume':
    # !!!! Needs to be updated to use models dict, etc
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt_without.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model1_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf1 = OvnMomGPTConfig(**model1_args)
    gptconf2 = OvnMomGPTConfig(**model2_args)

    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model1.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

for m in models.keys():
    if gptconfigs[m].block_size < models[m].config.block_size:
        models[m].crop_block_size(gptconfigs[m].block_size)
        models_args[m]['block_size'] = gptconfigs[m].block_size
    models[m].to(device)

# optimizer
optimizers = {}
scalers = {}
for m in models.keys():
    scalers[m] = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizers[m] = models[m].configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# compile the model
unoptimized_models = {}
if compile:
    print("compiling the model... (takes a ~minute)")
    for m in models.keys():
        unoptimized_models[m] = models[m]
        models[m] = torch.compile(models[m])
    
# wrap model into DDP container
if ddp:
    for m in models.keys():
        models[m] = DDP(models[m], device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    # It isn't ideal to manually set the seed, this is only to understand a performance difference
    # torch.manual_seed(0)
    out = {}
    target_out = {}
    cov_means_out = {}
    corrcoef_means_out = {}
    corrcoef_std_out2 = {}

    for m in models.keys():
        models[m].eval()
        
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        target_losses = torch.zeros(eval_iters)
        corrcoefs_all = torch.zeros((len(models.keys()), eval_iters, 5, 5))
        cov_all = torch.zeros((len(models.keys()), eval_iters, 5, 5))
        
        nan_count = 0
        for k in range(eval_iters):
            X1, Y1, Z1, X2, Y2, Z2 = get_batch(split, skip=skip_val)
            for mod_ind in range(len(models.keys())):
                with ctx:
                    mod = list(models.keys())[mod_ind]
                    # print(mod_ind, mod, gptconfigs[mod])
                    if mod_ind==1:
                        X = X1.bfloat16()
                        Y = Y1.bfloat16()
                        Z = Z1.bfloat16()
                        linear_pred = X[:, -1, skip_val+1]
                    else:
                        X = X2.bfloat16()
                        Y = Y2.bfloat16()
                        Z = Z2.bfloat16()
                        linear_pred = X[:, -1, 1]
                        
                    logits, loss = models[list(models.keys())[mod_ind]](X, Y)
                    
                    prediction = torch.add(logits[:, -1, :].view(-1), linear_pred,
                                           alpha=train_ovn_linear_beta)
                    new_mat = torch.row_stack((prediction, train_ovn_linear_beta * linear_pred,
                                               logits[:, -1, :].view(-1), Y[:, -1, -1], Z[:, -1, -1]))
                    corrcoefs_mat = torch.corrcoef(new_mat)
                    cov_mat = torch.cov(new_mat)
                    
                    corrcoefs_all[mod_ind, k] = corrcoefs_mat.data
                    cov_all[mod_ind, k] = cov_mat.data

            if not np.isnan(loss.item()):   
                losses[k] = loss.item()
                target_losses[k] = models[list(models.keys())[mod_ind]].get_target_loss(Y).item()

            else:
                nan_count += 1

        out[split] = losses.mean()
        target_out[split] = target_losses.mean()

        for mod_ind in range(len(models.keys())):
            cov_final = torch.zeros((5,5))
            corrcoefs_final = torch.zeros((5,5))
            corrcoefs_final_std = torch.zeros((5,5))
            for i in range(5):
                for j in range(5):
                    cov_final[i][j] = cov_all[mod_ind, :, i, j].mean()
                    corrcoefs_final[i][j] = corrcoefs_all[mod_ind, :, i, j].mean()
                    corrcoefs_final_std[i][j] = corrcoefs_all[mod_ind, :, i, j].std()

            cov_means_out[split] = cov_final
            corrcoef_means_out[split] = corrcoefs_final
            corrcoef_std_out2[split] = corrcoefs_final_std

            print(mod_ind, split, 'means')
            # print(cov_means_out[split])
            beta1_string = 'Z '
            beta2_string = 'Y '
            corr1_string = 'Z '
            corr2_string = 'Y '

            labels = ['pred', 'lin', 'nonlin']
            for i in range(3):
                b1 = cov_means_out[split][i][-1].item()/cov_means_out[split][i][i].item()
                b2 = cov_means_out[split][i][-2].item()/cov_means_out[split][i][i].item()
                c1 = cov_means_out[split][i][-1].item()/np.sqrt(cov_means_out[split][i][i].item()*cov_means_out[split][-1][-1].item())
                c2 = cov_means_out[split][i][-2].item()/np.sqrt(cov_means_out[split][i][i].item()*cov_means_out[split][-2][-2].item())
                beta1_string = beta1_string + labels[i] + ": {:.3}, ".format(b1)
                beta2_string = beta2_string + labels[i] + ": {:.3}, ".format(b2)
                corr1_string = corr1_string + labels[i] + ": {:.3}, ".format(c1)
                corr2_string = corr2_string + labels[i] + ": {:.3}, ".format(c2)
            print(mod_ind, split, 'betas', beta1_string, beta2_string)
            print(mod_ind, split, 'corrs', corr1_string, corr2_string)
            # print(X[:, -1, 1])
            # print(X2[:, -1, 1])

    # for i in range(len(models)):
        # models[i].train()
    for m in models.keys():
        models[m].train()

    return out, target_out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
torch.manual_seed(0)

X1, Y1, Z1, X2, Y2, Z2 = get_batch('train', skip=skip_val) # fetch the very first batch

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model1 = models['model1'].module if ddp else models['model1'] # unwrap DDP container if needed
raw_model2 = models['model2'].module if ddp else models['model2'] # unwrap DDP container if needed
running_mfu = -1.0
running_mfu2 = -1.0
# models = [model2, model1]

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for op in optimizers.keys():
        for param_group in optimizers[op].param_groups:
            param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        # for mod_ind in range(len(models)):
        losses, target_losses = estimate_loss()
        train_r2 = 1.0 - losses['train']/target_losses['train']
        val_r2 = 1.0 - losses['val']/target_losses['val']

        print(f"step {iter_num}: train loss {losses['train']:.8f}, target train loss {target_losses['train']:.8f}, train r2 {train_r2:.8f}, train ovn linear r2 {train_ovn_linear_r2:.8f}, train ovn linear beta {train_ovn_linear_beta:.8f}, val loss {losses['val']:.8f}, target val loss {target_losses['val']:.8f}, val r2 {val_r2:.8f} val ovn linear r2 {val_ovn_linear_r2:.8f}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu2*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model2.state_dict(),
                    'optimizer': optimizers['model2'].state_dict(),
                    'model_args': model2_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model1.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            model2.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            mod_list = list(models.keys())
            for mod_ind in range(len(models.keys())):
                if mod_ind == 1:
                    logits, loss = models[mod_list[mod_ind]](X1,Y1)
                    loss = loss / gradient_accumulation_steps
                else:
                    logits2, loss2 = models[mod_list[mod_ind]](X2,Y2)
                    loss2 = loss2 / gradient_accumulation_steps
                    
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X1, Y1, Z1, X2, Y2, Z2 = get_batch('train', skip=skip_val)
        # backward pass, with gradient scaling if training in fp16
        mod_list = list(models.keys())
        for mod_ind in range(len(models.keys())):
            m = mod_list[mod_ind]
            if mod_ind == 1:
                if not np.isnan(loss.item()):
                    scalers[m].scale(loss).backward()
            else:
                if not np.isnan(loss2.item()):
                    scalers[m].scale(loss2).backward()
                    
    # clip the gradient
    if grad_clip != 0.0:
        for m in models.keys():
            scalers[m].unscale_(optimizers[m])
            torch.nn.utils.clip_grad_norm_(models[m].parameters(), grad_clip)
        
    # step the optimizer and scaler if training in fp16
    for m in models.keys():
        scalers[m].step(optimizers[m])
        scalers[m].update()
        optimizers[m].zero_grad(set_to_none=True)
        
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        if not np.isnan(loss.item()):
            lossf = loss.item() * gradient_accumulation_steps
            lossf2 = loss2.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model1.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            mfu2 = raw_model2.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            running_mfu2 = mfu2 if running_mfu2 == -1.0 else 0.9*running_mfu2 + 0.1*mfu2
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}, loss2 {lossf2:.4f}, mfu2 {running_mfu2*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
