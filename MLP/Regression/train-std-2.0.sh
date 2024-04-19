#!/usr/bin/env bash

srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 2 --model Monolithic --num-rules 2 --seed 0 --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 4 --model Monolithic --num-rules 4 --seed 0 --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 8 --model Monolithic --num-rules 8 --seed 0 --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 16 --model Monolithic --num-rules 16 --seed 0 --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 32 --model Monolithic --num-rules 32 --seed 0 --data-seed 0 --noise-std 2.0

srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 2 --model Modular --num-rules 2 --seed 0 --op --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 4 --model Modular --num-rules 4 --seed 0 --op --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 8 --model Modular --num-rules 8 --seed 0 --op --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 16 --model Modular --num-rules 16 --seed 0 --op --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 32 --model Modular --num-rules 32 --seed 0 --op --data-seed 0 --noise-std 2.0

srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 2 --model Modular --num-rules 2 --seed 0 --joint --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 4 --model Modular --num-rules 4 --seed 0 --joint --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 8 --model Modular --num-rules 8 --seed 0 --joint --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 16 --model Modular --num-rules 16 --seed 0 --joint --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 32 --model Modular --num-rules 32 --seed 0 --joint --data-seed 0 --noise-std 2.0

srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 2 --model GT_Modular --num-rules 2 --seed 0 --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 4 --model GT_Modular --num-rules 4 --seed 0 --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 8 --model GT_Modular --num-rules 8 --seed 0 --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 16 --model GT_Modular --num-rules 16 --seed 0 --data-seed 0 --noise-std 2.0
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --iterations 20000 --gt-rules 32 --model GT_Modular --num-rules 32 --seed 0 --data-seed 0 --noise-std 2.0
