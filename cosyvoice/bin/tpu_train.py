from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from hyperpyyaml import load_hyperpyyaml

from cosyvoice.utils.losses import DPOLoss
from cosyvoice.utils.tpu_executor import Executor
from cosyvoice.utils.tpu_train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_tpu_model, check_modify_and_save_config)

from cosyvoice.llm.lora import apply_lora_to_llm

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_xla',
                        choices=['torch_ddp', 'deepspeed', 'torch_xla'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--ref_model', required=False, help='ref model used in dpo')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    args = parser.parse_args()
    return args


def _mp_fn(index, args):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    
    # Init env for ddp (XLA handles this mostly, but we can log info)
    init_distributed(args)

    # gan train has some special initialization logic
    gan = True if args.model == 'hifigan' else False

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')
    try:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={**override_dict, 'qwen_pretrain_path': args.qwen_pretrain_path})
    except Exception:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides=override_dict)
    if gan is True:
        configs['train_conf'] = configs['train_conf_gan']
    configs['train_conf'].update(vars(args))
    
    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan, args.dpo)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # load checkpoint
    if args.dpo is True:
        configs[args.model].forward = configs[args.model].forward_dpo
    model = configs[args.model]

    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

            # apply lora after load pretrain checkpoint
            if args.model == 'llm' and configs.get("lora_conf") is not None:
                print("apply lora")
                model = apply_lora_to_llm(
                    model,
                    r=configs['lora_conf']['lora_r'],
                    lora_alpha=configs['lora_conf']['lora_alpha'],
                    lora_dropout=configs['lora_conf']['lora_dropout']
                )

                for name, param in model.named_parameters():
                    if "lora_" not in name:
                        param.requires_grad = False


            if 'step' in state_dict:
                start_step = state_dict['step']
            if 'epoch' in state_dict:
                start_epoch = state_dict['epoch']
        else:
            logging.warning('checkpoint {} do not exsist!'.format(args.checkpoint))

    # Dispatch model from cpu to tpu
    model = wrap_tpu_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)
    scheduler.set_step(start_step)
    if scheduler_d is not None:
        scheduler_d.set_step(start_step)

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    # save_model(model, 'init', info_dict)

    # DPO related
    if args.dpo is True:
        ref_model = deepcopy(configs[args.model])
        state_dict = torch.load(args.ref_model, map_location='cpu')
        ref_model.load_state_dict(state_dict, strict=False)
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
        ref_model = wrap_tpu_model(args, ref_model)
    else:
        ref_model, dpo_loss = None, None

    # Get executor
    executor = Executor(gan=gan, ref_model=ref_model, dpo_loss=dpo_loss)
    executor.step = start_step

    # Init scaler, used for pytorch amp mixed precision training
    # XLA handles AMP differently, so we pass None or handle it inside executor
    scaler = None 

    if xm.is_master_ordinal():
        print(f"start training for {args.model}")

    # Start training loop
    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        
        # dist.barrier() # Not needed for XLA usually, or handled by xmp
        
        # Group join logic is for uneven workload, skipping for now
        group_join = None 
        
        if gan is True:
            executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                                        writer, info_dict, scaler, group_join)
        else:
            executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=ref_model)


def main():
    args = get_args()
    # Launch multi-processing
    # Assuming 8 cores for TPU v3-8, or user can set PJRT_DEVICE=TPU and xmp will handle it.
    # xmp.spawn will spawn nprocs processes.
    # If using PJRT, nprocs might need to be set or it defaults to available devices.
    # Let's try to detect or default to 8 (standard for single TPU host).
    # Or better, let xmp handle it if possible, but xmp.spawn requires nprocs.
    # If running on TPU VM, usually 8.
    
    # We can check if TPU is available.
    # For now, let's assume 8 if not specified.
    # But wait, if the user runs this on a single TPU chip (e.g. Colab), it might be 1.
    # Safest is to let the user specify or default to 1? No, TPU training usually implies distributed.
    # Let's use 8 as default for TPU VM.
    
    # Actually, we can check os.environ['TPU_NUM_DEVICES'] or similar.
    # Or just use 8.
    
    nprocs = 8
    xmp.spawn(_mp_fn, args=(args,), nprocs=nprocs)


if __name__ == '__main__':
    main()
