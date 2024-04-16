# Trainer file

import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data import rules
from model import Model

parser = argparse.ArgumentParser(description='Rule MLP')
parser.add_argument('--search-version', type=int, default=1, choices=(1,2))
parser.add_argument('--gt-rules', type=int, default=2)
parser.add_argument('--data-seed', type=int, default=0)
parser.add_argument('--seq-len', type=int, default=10)

parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--iterations', type=int, default=50000)

parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--att-dim', type=int, default=512)
parser.add_argument('--model', type=str, default='Monolithic', choices=('Monolithic', 'Modular', 'GT_Modular'))
parser.add_argument('--num-heads', type=int, default=2)
parser.add_argument('--num-rules', type=int, default=2)
parser.add_argument('--op', action='store_true', default=False)

parser.add_argument('--scheduler', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

sns.color_palette("dark", as_cmap=True)
sns.set(style="darkgrid")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(args.seed)

if args.seq_len == 10:
    test_lens = [3, 5, 10]  # [3, 5, 10, 20, 30]
else:
    test_lens = [3, 5, 10]    # [10, 20, 30, 40, 50]

config = vars(args)
device = torch.device('cuda')

if args.op:
    extras = f'_operation-only_'
else:
    extras = '_'

if args.scheduler:
    ext='_scheduler'
else:
    ext=''

name = f'Sequence_{args.seq_len}{ext}/Search-Version_{args.search_version}/Data-Seed_{args.data_seed}/GT_Rules_{args.gt_rules}/{args.model}{extras}{args.dim}_{args.att_dim}_{args.num_heads}_{args.num_rules}_{args.seed}'

if not os.path.exists(name):
    os.makedirs(name)
else:
    print(name)
    print("Folder Already Exists")
    if os.path.exists(os.path.join(name, 'loss.png')):
        print("Model Already Exists")
        exit()

if args.search_version == 1:
    in_dim = args.num_rules * 5
elif args.search_version == 2:
    in_dim = args.num_rules * 7
else:
    print("Search Version Not Supported")
    exit()

model = Model(
    dim = args.dim,
    att_dim = args.att_dim,
    num_heads = args.num_heads,
    in_dim = in_dim,
    model = args.model,
    num_rules = args.num_rules,
    op = args.op
).to(device)

gt_ticks = [f'Ground Truth Rule {i}' for i in range(1, args.gt_rules+1)]

num_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"Number of Parameters: {num_params}")
with open(os.path.join(name, 'log.txt'), 'w') as f:
    f.write(f"Number of Parameters: {num_params}\n")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.5)
criterion = nn.BCEWithLogitsLoss()

df = pd.DataFrame(columns=["Iterations", "Loss", "Accuracy"])

def eval_step(eval_len=args.seq_len, ood=False, n_evals=10):
    model.eval()
    total_loss = 0.
    total_acc = 0.

    for _ in range(n_evals):
        data, label, op = rules(args.batch_size, eval_len, args.gt_rules, 2, \
                            args.search_version, args.data_seed, ood, noise_mean=0, noise_std=2.0)

        data = torch.Tensor(data).to(device)
        label = torch.Tensor(label).to(device)
        op = torch.Tensor(op).to(device)

        out, score = model(data, op)

        loss = criterion(out, label)
        acc = torch.eq(out >= 0., label).double().mean()

        total_loss += loss.item()
        total_acc += acc.item()

    return total_loss / float(n_evals), total_acc * 100. / float(n_evals)


def train_step():
    model.train()
    model.zero_grad()

    data, label, op = rules(args.batch_size, args.seq_len, args.gt_rules, 2, \
                            args.search_version, args.data_seed, noise_mean=0, noise_std=2.0)

    data = torch.Tensor(data).to(device)
    label = torch.Tensor(label).to(device)
    op = torch.Tensor(op).to(device)

    out, score = model(data, op)

    loss = criterion(out, label)
    acc = torch.eq(out >= 0., label).double().mean()

    loss.backward()
    optimizer.step()

    return loss.item(), acc.item() * 100.


eval_log = f'Iteration: 0 | '
train_log = f'Iteration: 0 | '
eval_ood_log = f'Iteration: 0 | '

for seq_len in test_lens:
    eval_loss, eval_acc = eval_step(seq_len)
    eval_ood_loss, eval_ood_acc = eval_step(seq_len, True)

    if seq_len == args.seq_len:
        df.loc[-1] = [0, eval_loss, eval_acc]
        df.index = df.index + 1

    eval_log += f'Seq. Len: {seq_len} - Eval Loss: {eval_loss} - Eval Acc: {eval_acc} | '
    train_log += f'Seq. Len: {seq_len} - Train Loss: {eval_loss} - Train Acc: {eval_acc} | '
    eval_ood_log += f'Seq. Len: {seq_len} - Eval OoD Loss: {eval_ood_loss} - Train OoD Acc: {eval_ood_acc} | '

log = train_log + '\n' + eval_log + '\n' + eval_ood_log + '\n'
print(log)

with open(os.path.join(name, 'log.txt'), 'a') as f:
    f.write(log)

best_val = 0.
for i in range(1, args.iterations+1):
    if i % 5000 == 0:
        eval_loss, eval_acc = eval_step()
        val_loss, val_acc = eval_step()

        df.loc[-1] = [i, eval_loss, eval_acc]
        df.index = df.index + 1

    train_loss, train_acc = train_step()

    if i % 5000 == 0:
        if args.scheduler:
            scheduler.step()

        log = f'Iteration: {i} | Train Loss: {train_loss} - Train Acc: {train_acc}\n' \
              f'Iteration: {i} | Eval Loss: {eval_loss} - Eval Acc: {eval_acc}\n'
        print(log)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(name, 'model_best.pt'))

        torch.save(model.state_dict(), os.path.join(name, 'model_last.pt'))

        with open(os.path.join(name, 'log.txt'), 'a') as f:
            f.write(log)

sns.lineplot(data=df, x="Iterations", y="Loss")
plt.savefig(os.path.join(name, 'loss.png'), bbox_inches='tight')
plt.close()

sns.lineplot(data=df, x="Iterations", y="Accuracy")
plt.savefig(os.path.join(name, 'acc.png'), bbox_inches='tight')
plt.close()