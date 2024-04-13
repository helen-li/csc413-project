#!/usr/bin/env bash

cd Regression

srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 2 --model Monolithic --num-rules 2
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 4 --model Monolithic --num-rules 4
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 8 --model Monolithic --num-rules 8
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 16 --model Monolithic --num-rules 16
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 32 --model Monolithic --num-rules 32

srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 2 --model Modular --num-rules 2
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 4 --model Modular --num-rules 4
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 8 --model Modular --num-rules 8
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 16 --model Modular --num-rules 16
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 32 --model Modular --num-rules 32

srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 2 --model Modular --num-rules 2 --op --joint
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 4 --model Modular --num-rules 4 --op --joint
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 8 --model Modular --num-rules 8 --op --joint
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 16 --model Modular --num-rules 16 --op --joint
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 32 --model Modular --num-rules 32 --op --joint

srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 2 --model GT_Modular --num-rules 2
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 4 --model GT_Modular --num-rules 4
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 8 --model GT_Modular --num-rules 8
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 16 --model GT_Modular --num-rules 16
srun -p csc413 --gres gpu -c 2 --pty python3 main.py --gt-rules 32 --model GT_Modular --num-rules 32
