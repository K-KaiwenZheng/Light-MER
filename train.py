import os
import time
import random
import argparse
import logging
import numpy as np

import torch
from datetime import datetime
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf

import my_affectgpt.tasks as tasks
from my_affectgpt.common.config import Config
from my_affectgpt.common.dist_utils import get_rank, init_distributed_mode
from my_affectgpt.common.logger import setup_logger
from my_affectgpt.common.registry import registry
from my_affectgpt.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *

def setup_seeds(config): 
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--options",  nargs="+", help="overwrite params in xxx.config")
    args = parser.parse_args()
    return args

def get_runner_class(cfg):
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    return runner_cls


def build_teacher_model(model_cfg):
    """
    构建教师模型用于 OT 蒸馏。
    覆盖 LLM 和编码器配置以匹配教师 checkpoint 的架构。
    """
    # [Teacher model construction removed - paper under review]
    # Builds teacher model from config, loads checkpoint, freezes parameters
    raise NotImplementedError("Contact authors for details")

def main():

    args = parse_args()
    cfg = Config(args)

    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    job_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
    job_id = f"{job_name}_{datetime.now().strftime('%Y%m%d%H%M')[:-1]}"

    print(job_id)

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger() 
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    # ====== OT Distillation: Load Teacher Model ======
    teacher_cfg = cfg.model_cfg.get("teacher", None)
    if teacher_cfg is not None:
        teacher_model = build_teacher_model(cfg.model_cfg)
        if teacher_model is not None:
            ot_weight = teacher_cfg.get("ot_weight", 1.0)
            ot_epsilon = teacher_cfg.get("ot_epsilon", 0.1)
            ot_common_dim = teacher_cfg.get("ot_common_dim", 256)
            ot_num_iters = teacher_cfg.get("ot_num_iters", 20)
            ot_ramp_steps = teacher_cfg.get("ot_ramp_steps", 5000)
            kl_weight = teacher_cfg.get("kl_weight", 0.0)
            kl_temperature = teacher_cfg.get("kl_temperature", 2.0)

            model.set_teacher(
                teacher_model,
                ot_weight=ot_weight,
                ot_epsilon=ot_epsilon,
                ot_common_dim=ot_common_dim,
                ot_num_iters=ot_num_iters,
                ot_ramp_steps=ot_ramp_steps,
                kl_weight=kl_weight,
                kl_temperature=kl_temperature,
            )
            print(f"OT distillation enabled: ot_weight={ot_weight}, "
                  f"epsilon={ot_epsilon}, common_dim={ot_common_dim}, kl_weight={kl_weight}")
    # ==================================================

    runner = get_runner_class(cfg)(
        cfg=cfg,
        job_id=job_id, 
        task=task, 
        model=model, 
        datasets=datasets
    )

    # ====== Pre-training Sanity Check ======
    print("\n====== Pre-training Sanity Check ======")
    print(f"  Student device: {next(model.parameters()).device}")
    if getattr(model, 'teacher', None) is not None:
        print(f"  Teacher device: {next(model.teacher.parameters()).device}")
        opt_param_ids = {id(p) for g in runner.optimizer.param_groups for p in g["params"]}
        teacher_param_ids = {id(p) for p in model.teacher.parameters()}
        overlap = len(opt_param_ids & teacher_param_ids)
        print(f"  Teacher params in optimizer: {overlap} (should be 0)")
        if overlap > 0:
            print("  WARNING: Teacher parameters found in optimizer!")
        ot_proj_total = sum(p.numel() for p in model.ot_hidden_proj.parameters())
        ot_proj_trainable = sum(p.numel() for p in model.ot_hidden_proj.parameters() if p.requires_grad)
        print(f"  OT projector params: {ot_proj_trainable:,} trainable / {ot_proj_total:,} total")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Student trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"  Grad accumulation: {runner.accum_grad_iters}")
    print(f"  Effective batch size: {cfg.run_cfg.batch_size_train} × {runner.accum_grad_iters} = {cfg.run_cfg.batch_size_train * runner.accum_grad_iters}")
    print("====== Sanity Check Done ======\n")

    runner.train()

if __name__ == "__main__":
    main()
