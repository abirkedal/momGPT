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
block_size = 4

# model
subtract_ovn_linear = True
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
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

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
# train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
# val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
# print(train_data.shape)
data = np.load('/home/andreas/momGPT/data/firstratedata/multi_ticker_dict_intraday.npy', allow_pickle='TRUE').item()
data_df = pd.concat(data, axis=0)
data_df = data_df.reset_index(level=[0])
data_df = data_df.rename(columns={'level_0': 'ticker'})

ret_cols = ['overnight_return', 'into_close_safe_return', 'intraday_return', 'future_overnight_return']
train_data_df = data_df[data_df.index <= isos_split_date][['ticker'] + ret_cols]
train_data_gb = train_data_df.groupby('ticker')
tickers = list(train_data_gb.groups.keys())
train_data_vals_dict = {}
for ticker in tickers:
    train_data_vals_dict[ticker]= train_data_gb.get_group(ticker)[ret_cols].replace(0.0, np.nan).dropna(how='all').fillna(0).astype('float').values

val_data_df = data_df[data_df.index > isos_split_date][['ticker'] + ret_cols]
val_data_gb = val_data_df.groupby('ticker')
tickers = list(val_data_gb.groups.keys())
val_data_vals_dict = {}
for ticker in tickers:
    val_data_vals_dict[ticker]= val_data_gb.get_group(ticker)[ret_cols].replace(0.0, np.nan).dropna(how='all').fillna(0).astype('float').values

train_data_df = data_df[data_df.index <= isos_split_date][ret_cols].replace(0.0, np.nan).dropna(how='all').fillna(0).astype('float')
train_data = train_data_df.values

corr_data = train_data_df[['into_close_safe_return', 'future_overnight_return']].copy()
corr_data['into_close_safe_return'] = corr_data['into_close_safe_return'].clip(lower=-input_clip, upper=input_clip)
corr_mat = corr_data.corr()
cov_mat = corr_data.cov()
train_ovn_linear_r2 = (corr_mat.values[0][1])**2
train_ovn_linear_beta = cov_mat.values[0][1]/cov_mat.values[0][0]

val_data_df = data_df[data_df.index > isos_split_date][ret_cols].replace(0.0, np.nan).dropna(how='all').fillna(0).astype('float')
val_data = val_data_df.values
val_corr_data = val_data_df[['into_close_safe_return', 'future_overnight_return']].copy()
val_corr_data['into_close_safe_return'] = val_corr_data['into_close_safe_return'].clip(lower=-input_clip, upper=input_clip)
val_ovn_linear_r2 = (val_corr_data[['into_close_safe_return', 'future_overnight_return']].corr().values[0][1])**2
print(train_data.shape, train_ovn_linear_r2, train_ovn_linear_beta, val_ovn_linear_r2)

def get_batch(split):
    t = time.time()
    if split == 'train':
        # datadf = train_data_df
        dfd = train_data_vals_dict
    else:
        # datadf = val_data_df
        dfd = val_data_vals_dict
    
    # tickers = datadf['ticker'].unique()
    tickers = list(dfd.keys())
    
    ix = torch.randint(len(tickers), (2 * batch_size,))

    xl = []
    yl = []
    zl = []

    t1 = time.time()
    # print('get_batch prep', t1 - t)
    
    count = 0
    for i in ix:
        if count < batch_size:
            ticker = tickers[i]
            # My old version that used the following is incredibly slow, since we are pawing through the entire
            # datadf each iteration:
            # data = datadf[datadf.ticker == ticker][ret_cols].replace(0.0, np.nan).dropna(how='all').fillna(0).astype('float').values
            # data = gb.get_group(ticker)[ret_cols].replace(0.0, np.nan).dropna(how='all').fillna(0).astype('float').values
            data = dfd[ticker]
            
            # print('j', data.shape[0], gptconf.n_embd)
            ds0 = data.shape[0]
            if ds0 - gptconf.n_embd > 0:
                # if count < batch_size:
                j = torch.randint(ds0 - gptconf.n_embd, (1,))[0]
                xd = np.clip(data[j:(j+gptconf.n_embd), :-1], -input_clip, input_clip)
                yd = data[j:(j+gptconf.n_embd), 1:].copy()
                zd = data[j:(j+gptconf.n_embd), 1:].copy()
                if subtract_ovn_linear:
                    yd[:, -1] = yd[:, -1] - train_ovn_linear_beta * xd[:, 1]
                    zl.append(torch.from_numpy((zd)))
                    # print('yd', yd)
                    # print('zd', zd)
                xl.append(torch.from_numpy((xd)))
                yl.append(torch.from_numpy((yd)))
                
                count += 1

    t2 = time.time()
    # print('get_batch making xl and yl', t2 - t1)
    
    x = torch.stack(xl)
    y = torch.stack(yl)
                      
    x = x.bfloat16()
    y = y.bfloat16()
    
    # print(len(data), ix)
    # print('x', x.shape, y.shape)
    # print(y)
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    if len(zl) > 0:
        z = torch.stack(zl)
        z = z.bfloat16()
        
        if device_type == 'cuda':
            z = z.pin_memory().to(device, non_blocking=True)
        else:
            z = z.to(device)
    # print('ix: ', ix)
    t3 = time.time()
    # print('get_batch completion', t3 - t2)

    if len(zl) > 0:
        return x, y, z
    else:
        return x, y, None

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
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
print('Using the following model_args:', model_args)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of 1")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 1
    gptconf = OvnMomGPTConfig(**model_args)
    model = OvnMomGPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = OvnMomGPTConfig(**model_args)
    model = OvnMomGPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

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

    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        target_losses = torch.zeros(eval_iters)
        corrcoefs_all = torch.zeros((eval_iters, 5, 5))
        cov_all = torch.zeros((eval_iters, 5, 5))

        nan_count = 0
        for k in range(eval_iters):
            X, Y, Z = get_batch(split)
            with ctx:
                X = X.bfloat16()
                Y = Y.bfloat16()
                
                logits, loss = model(X, Y)

                linear_pred = X[:, -1, 1]
                
                if subtract_ovn_linear:
                    Z = Z.bfloat16()
                    prediction = torch.add(logits[:, -1, :].view(-1), linear_pred,
                                           alpha=train_ovn_linear_beta)
                    new_mat = torch.row_stack((prediction, train_ovn_linear_beta * linear_pred,
                                               logits[:, -1, :].view(-1), Y[:, -1, -1], Z[:, -1, -1]))
                    corrcoefs_mat = torch.corrcoef(new_mat)
                    cov_mat = torch.cov(new_mat)
                                              
                else:
                    prediction = torch.add(logits[:, -1, :].view(-1), linear_pred,
                                           alpha=train_ovn_linear_beta)
                    new_mat = torch.row_stack((prediction, train_ovn_linear_beta * linear_pred,
                                               logits[:, -1, :].view(-1), Y[:, -1, -1], Z[:, -1, -1]))
                    corrcoefs_mat = torch.corrcoef(new_mat)
                    cov_mat = torch.cov(new_mat)
                                              
                corrcoefs_all[k] = corrcoefs_mat.data
                cov_all[k] = cov_mat.data

            if not np.isnan(loss.item()):   
                losses[k] = loss.item()
                target_losses[k] = model.get_target_loss(Y).item()
            else:
                nan_count += 1

        out[split] = losses.mean()
        target_out[split] = target_losses.mean()

        cov_final = torch.zeros((5,5))
        corrcoefs_final = torch.zeros((5,5))
        corrcoefs_final_std = torch.zeros((5,5))
        for i in range(5):
            for j in range(5):
                cov_final[i][j] = cov_all[:, i, j].mean()
                corrcoefs_final[i][j] = corrcoefs_all[:, i, j].mean()
                corrcoefs_final_std[i][j] = corrcoefs_all[:, i, j].std()

        cov_means_out[split] = cov_final
        corrcoef_means_out[split] = corrcoefs_final
        corrcoef_std_out2[split] = corrcoefs_final_std

        print(split, 'means')
        print(cov_means_out[split])
        print(cov_means_out[split][0][-1].item()/cov_means_out[split][0][0].item(),cov_means_out[split][1][-1].item()/cov_means_out[split][1][1].item(),cov_means_out[split][2][-1].item()/cov_means_out[split][2][2].item(),cov_means_out[split][0][-2].item()/cov_means_out[split][0][0].item(),cov_means_out[split][1][-2].item()/cov_means_out[split][1][1].item(),cov_means_out[split][2][-2].item()/cov_means_out[split][2][2].item())
        print(corrcoef_means_out[split])
        print(split, 'std')
        print(corrcoef_std_out2[split])
        # print(split, 'means Z', corrcoef_means_out[split][-1], 'std Z', corrcoef_std_out2[split][-1])
        # print(split, 'means Y', corrcoef_means_out[split][-2], 'std Y', corrcoef_std_out2[split][-2])
        
    model.train()

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

# Here the Y is the same as the X, but shifted one to the left
data = np.load('/home/andreas/momGPT/data/firstratedata/multi_ticker_dict_intraday.npy', allow_pickle='TRUE').item()
data_df = pd.concat(data, axis=0)
data_df = data_df.reset_index(level=[0])
data_df = data_df.rename(columns={'level_0': 'ticker'})

train_data = data_df[data_df.index <= isos_split_date][['overnight_return', 'into_close_safe_return', 'intraday_return', 'future_overnight_return']].fillna(0).astype('float').values

X, Y, Z = get_batch('train') # fetch the very first batch
print(X.size(), Y.size(), torch.tensor(train_data).size())
# print(X)
# print(Y)
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
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
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
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
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, Z = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        if not np.isnan(loss.item()):
            scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        if not np.isnan(loss.item()):
            lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
