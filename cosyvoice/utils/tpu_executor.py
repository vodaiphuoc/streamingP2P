import logging
from contextlib import nullcontext
import os
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm

from cosyvoice.utils.tpu_train_utils import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, batch_backward, save_model, cosyvoice_join


class Executor:

    def __init__(self, gan: bool = False, ref_model: torch.nn.Module = None, dpo_loss: torch.nn.Module = None):
        self.gan = gan
        self.ref_model = ref_model
        self.dpo_loss = dpo_loss
        self.step = 0
        self.epoch = 0
        self.rank = xm.get_ordinal()
        self.device = xm.xla_device()

    def train_one_epoc(
            self, 
            model, 
            optimizer, 
            scheduler, 
            train_data_loader, 
            cv_data_loader, 
            writer, 
            info_dict, 
            scaler, 
            group_join, 
            ref_model=None
        ):
        ''' Train one epoch
        '''
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        
        model.train()
        if self.ref_model is not None:
            self.ref_model.eval()
            
        # XLA doesn't need model.join context for DDP usually, or it's handled differently.
        # We'll use nullcontext for now.
        model_context = nullcontext
        
        # ParallelLoader is usually used here for performance, but we'll stick to standard loader for now
        # and assume the user might optimize later or we can add it if needed.
        # Ideally: train_data_loader = pl.ParallelLoader(train_data_loader, [self.device]).per_device_loader(self.device)
        # But let's keep it simple first.
        
        with model_context():
            for batch_idx, batch_dict in tqdm(enumerate(train_data_loader)):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                
                # cosyvoice_join is for uneven workload check, skipping for now on TPU
                if cosyvoice_join(group_join, info_dict):
                    break

                # Gradient accumulation logic
                # In XLA, we don't need no_sync context usually if we use xm.optimizer_step correctly?
                # Actually, if we want to accumulate gradients without syncing, we need to be careful.
                # Standard DDP uses no_sync. XLA's DDP (if used) might support it.
                # But here we are likely using XLA's native distributed training where gradients are reduced during optimizer_step.
                # So we don't need no_sync context because we only call optimizer_step (which triggers reduction) every accum_grad steps.
                # However, we need to make sure gradients are NOT reduced automatically on backward().
                # In XLA, backward() computes gradients but doesn't reduce them until optimizer_step (which calls all_reduce).
                # So we are good! We just don't call optimizer_step until accumulation is done.
                
                context = nullcontext

                with context():
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict, ref_model=self.ref_model, dpo_loss=self.dpo_loss)
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                log_per_step(writer, info_dict)
                
                if info_dict['save_per_step'] > 0 and \
                    (self.step + 1) % info_dict['save_per_step'] == 0 and \
                    (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    
                    xm.rendezvous('save_per_step')
                    print('run self.cv')
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1

        xm.rendezvous('epoch_end')
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    def train_one_epoc_gan(self, model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                           writer, info_dict, scaler, group_join):
        ''' Train one epoch
        '''
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        
        model.train()
        model_context = nullcontext
        
        with model_context():
            for batch_idx, batch_dict in tqdm(enumerate(train_data_loader)):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                context = nullcontext

                with context():
                    batch_dict['turn'] = 'discriminator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer_d, scheduler_d, scaler, info_dict)
                optimizer.zero_grad()
                log_per_step(writer, info_dict)
                
                with context():
                    batch_dict['turn'] = 'generator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                optimizer_d.zero_grad()
                log_per_step(writer, info_dict)
                
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    xm.rendezvous('save_per_step_gan')
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        xm.rendezvous('epoch_end_gan')
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        ''' Cross validation on
        '''
        logging.info('Epoch {} Step {} on_batch_end {} CV rank {}'.format(self.epoch, self.step + 1, on_batch_end, self.rank))
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan is True:
                batch_dict['turn'] = 'generator'
            info_dict = batch_forward(model, batch_dict, None, info_dict)

            for k, v in info_dict['loss_dict'].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                total_loss_dict[k].append(v.mean().item() * num_utts)
            log_per_step(None, info_dict)
            
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict['loss_dict'] = total_loss_dict
        log_per_save(writer, info_dict)
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        save_model(model, model_name, info_dict)
