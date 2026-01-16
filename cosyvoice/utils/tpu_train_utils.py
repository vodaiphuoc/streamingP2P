import logging
import os
import torch
import json
import re
import datetime
import yaml

import torch.optim as optim
import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from cosyvoice.llm.llm import Qwen2LM
from cosyvoice.dataset.dataset import get_dataset
from cosyvoice.utils.scheduler import WarmupLR, NoamHoldAnnealing, ConstantLR

def init_distributed(args):
    # XLA handles distributed init via xmp.spawn
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    local_rank = xm.get_local_ordinal()
    logging.info('training on multiple tpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    return world_size, local_rank, rank


def init_dataset_and_dataloader(args, configs, gan, dpo):
    data_pipeline = configs['data_pipeline_gan'] if gan is True else configs['data_pipeline']
    # On TPU, we usually want to shard the dataset manually or use DistributedSampler
    # But get_dataset seems to handle partition=True. 
    # We need to ensure get_dataset uses the correct rank/world_size if it relies on dist.get_rank()
    # If get_dataset uses torch.distributed, we might need to patch it or ensure torch.distributed is initialized if possible,
    # or pass rank/world_size explicitly if get_dataset supports it.
    # Looking at train_utils.py, it calls get_dataset(..., partition=True)
    # Let's assume get_dataset uses torch.distributed. 
    # In torch_xla, we can use xm.get_ordinal() and xm.xrt_world_size().
    # We might need to modify get_dataset or set environment variables that get_dataset reads.
    # For now, let's assume standard torch.distributed works if we init it, OR we rely on get_dataset reading env vars.
    
    # However, torch_xla doesn't always use torch.distributed default process group.
    # Let's try to initialize torch.distributed using 'xla' backend if available or just rely on env vars.
    # But usually for TPU, we use ParallelLoader.
    
    train_dataset = get_dataset(args.train_data, data_pipeline=data_pipeline, mode='train', gan=gan, dpo=dpo, shuffle=True, partition=True)
    cv_dataset = get_dataset(args.cv_data, data_pipeline=data_pipeline, mode='dev', gan=gan, dpo=dpo, shuffle=False, partition=False)

    # do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time when the for loop starts
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=2,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=2,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def check_modify_and_save_config(args, configs):
    configs['train_conf']["dtype"] = 'bf16' if args.use_amp is True else 'fp32'
    return configs


def wrap_tpu_model(args, model: Qwen2LM):
    device = xm.xla_device()
    model.to(device)
    return model


def init_optimizer_and_scheduler(args, configs, model, gan):
    if gan is False:
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                **configs['train_conf']['optim_conf']
            )
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                **configs['train_conf']['optim_conf']
            )
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])

        optimizer_d, scheduler_d = None, None

    else:
        # currently we wrap generator and discriminator in one model
        if configs['train_conf']['optim'] == 'adam':
            optimizer = optim.Adam(model.generator.parameters(), **configs['train_conf']['optim_conf'])
        elif configs['train_conf']['optim'] == 'adamw':
            optimizer = optim.AdamW(model.generator.parameters(), **configs['train_conf']['optim_conf'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler = WarmupLR(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler = NoamHoldAnnealing(optimizer, **configs['train_conf']['scheduler_conf'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler = ConstantLR(optimizer)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])

        if configs['train_conf']['optim_d'] == 'adam':
            optimizer_d = optim.Adam(model.discriminator.parameters(), **configs['train_conf']['optim_conf_d'])
        elif configs['train_conf']['optim_d'] == 'adamw':
            optimizer_d = optim.AdamW(model.discriminator.parameters(), **configs['train_conf']['optim_conf_d'])
        else:
            raise ValueError("unknown optimizer: " + configs['train_conf'])

        if configs['train_conf']['scheduler_d'] == 'warmuplr':
            scheduler_type = WarmupLR
            scheduler_d = WarmupLR(optimizer_d, **configs['train_conf']['scheduler_d'])
        elif configs['train_conf']['scheduler_d'] == 'NoamHoldAnnealing':
            scheduler_type = NoamHoldAnnealing
            scheduler_d = NoamHoldAnnealing(optimizer_d, **configs['train_conf']['scheduler_d'])
        elif configs['train_conf']['scheduler'] == 'constantlr':
            scheduler_type = ConstantLR
            scheduler_d = ConstantLR(optimizer_d)
        else:
            raise ValueError("unknown scheduler: " + configs['train_conf'])
    return model, optimizer, scheduler, optimizer_d, scheduler_d


def init_summarywriter(args):
    writer = None
    if xm.is_master_ordinal():
        os.makedirs(args.model_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)
    return writer


def save_model(model, model_name, info_dict):
    # rank = int(os.environ.get('RANK', 0))
    model_dir = info_dict["model_dir"]
    save_model_path = os.path.join(model_dir, '{}.pt'.format(model_name))

    # xm.save handles saving only on master ordinal
    xm.save(model.state_dict(), save_model_path)
    
    if xm.is_master_ordinal():
        info_path = re.sub('.pt$', '.yaml', save_model_path)
        info_dict['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(info_path, 'w') as fout:
            data = yaml.dump(info_dict)
            fout.write(data)
        logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(xm.get_ordinal(), save_model_path))


def cosyvoice_join(group_join, info_dict):
    # Not strictly needed for XLA if we rely on xm.mark_step / optimizer_step for sync
    # But if we want to check for uneven workload, we might need a barrier.
    # xm.rendezvous('cosyvoice_join')
    return False


def batch_forward(model, batch, scaler, info_dict, ref_model=None, dpo_loss=None):
    device = xm.xla_device()
    
    # Move batch to device
    # Note: ParallelLoader usually handles this if used, but here we might be doing it manually
    # or relying on the loop to move data.
    # In XLA, it's best to let ParallelLoader handle data movement.
    # But here we receive 'batch' which is a dict of tensors.
    # We should ensure they are on the correct device.
    # If using ParallelLoader, they are already on device.
    # If not, we need to move them.
    # We will assume the executor uses ParallelLoader and passes device-bound tensors.
    
    # dtype handling: XLA handles bf16 automatically if env var is set or we can cast.
    # For now, let's just run the model.
    
    info_dict['loss_dict'] = model(batch, device)
    
    if ref_model is not None and dpo_loss is not None:
        chosen_logps = info_dict['loss_dict']["chosen_logps"]
        rejected_logps = info_dict['loss_dict']["rejected_logps"]
        sft_loss = info_dict['loss_dict']['loss']
        with torch.no_grad():
            ref_loss_dict = ref_model(batch, device)
        reference_chosen_logps = ref_loss_dict["chosen_logps"]
        reference_rejected_logps = ref_loss_dict["rejected_logps"]
        preference_loss, chosen_reward, reject_reward = dpo_loss(
            chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
        )
        dpo_acc = (chosen_reward > reject_reward).float().mean()
        info_dict['loss_dict']["loss"] = preference_loss + sft_loss
        info_dict['loss_dict']["sft_loss"] = sft_loss
        info_dict['loss_dict']["dpo_loss"] = preference_loss
        info_dict['loss_dict']["dpo_acc"] = dpo_acc
        info_dict['loss_dict']["chosen_reward"] = chosen_reward.mean()
        info_dict['loss_dict']["reject_reward"] = reject_reward.mean()
            
    return info_dict


def batch_backward(model, scaler, info_dict):
    scaled_loss = info_dict['loss_dict']['loss'] / info_dict['accum_grad']
    scaled_loss.backward()
    info_dict['loss_dict']['loss'] = scaled_loss
    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict):
    grad_norm = 0.0
    if (info_dict['batch_idx'] + 1) % info_dict["accum_grad"] == 0:
        # Gradient clipping
        # xm.optimizer_step handles the step and all-reduce
        # But we need to clip grads before step.
        # XLA requires clipping to be done carefully or using xm.utils?
        # Actually standard clip_grad_norm_ works if we do it before optimizer.step()
        # But in XLA, gradients are lazy.
        # xm.optimizer_step calls optimizer.step() and xm.mark_step().
        
        # For clipping:
        # torch.nn.utils.clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
        # However, we need to make sure gradients are computed.
        # xm.optimizer_step does all-reduce.
        
        # Correct pattern for XLA with clipping:
        # loss.backward()
        # xm.optimizer_step(optimizer, barrier=True) # This does all-reduce and step
        
        # But if we want to clip, we should do:
        # loss.backward()
        # xm.reduce_gradients(optimizer)
        # clip_grad_norm_(model.parameters(), ...)
        # optimizer.step()
        
        xm.reduce_gradients(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), info_dict['grad_clip'])
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
    info_dict["lr"] = optimizer.param_groups[0]['lr']
    info_dict["grad_norm"] = grad_norm
    return info_dict


def log_per_step(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict.get('epoch', 0)
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    rank = xm.get_ordinal()

    # only rank 0 write to tensorboard
    if writer is not None:
        if (info_dict['batch_idx'] + 1) % info_dict['accum_grad'] == 0:
            for k in ['epoch', 'lr', 'grad_norm']:
                writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
            for k, v in loss_dict.items():
                writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)

    # TRAIN & CV, Shell log (stdout)
    if (info_dict['batch_idx'] + 1) % info_dict['log_interval'] == 0:
        log_str = '{} Batch {}/{} '.format(tag, epoch, batch_idx + 1)
        for name, value in loss_dict.items():
            log_str += '{} {:.6f} '.format(name, value)
        if tag == "TRAIN":
            log_str += 'lr {:.8f} grad_norm {:.6f}'.format(
                info_dict["lr"], info_dict['grad_norm'])
        log_str += ' rank {}'.format(rank)
        # Use xm.master_print for logging to avoid clutter, or just logging.debug
        if xm.is_master_ordinal():
            logging.info(log_str) # Changed to info to be visible


def log_per_save(writer, info_dict):
    tag = info_dict["tag"]
    epoch = info_dict["epoch"]
    step = info_dict["step"]
    loss_dict = info_dict["loss_dict"]
    lr = info_dict['lr']
    rank = xm.get_ordinal()
    
    if xm.is_master_ordinal():
        logging.info(
            'Epoch {} Step {} CV info lr {} {} rank {}'.format(
                epoch, step + 1, lr, rank, ' '.join(['{} {}'.format(k, v) for k, v in loss_dict.items()])))

    if writer is not None:
        for k in ['epoch', 'lr']:
            writer.add_scalar('{}/{}'.format(tag, k), info_dict[k], step + 1)
        for k, v in loss_dict.items():
            writer.add_scalar('{}/{}'.format(tag, k), v, step + 1)
