from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
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


def _mp_fn(index):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    
    # Init env for ddp (XLA handles this mostly, but we can log info)
    init_distributed(args = 0)

    print(f'this is map function at index {index}')

def main():
    xmp.spawn(_mp_fn, nprocs=None)


if __name__ == '__main__':
    main()
