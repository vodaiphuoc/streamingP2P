from peft import LoraConfig, get_peft_model
from peft.utils import TaskType
from cosyvoice.llm.llm import Qwen2LM

def apply_lora_to_llm(
    model: Qwen2LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
)->Qwen2LM:
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )
    
    model.llm.model = get_peft_model(model.llm.model, lora_config)
    model.llm.model.print_trainable_parameters()

    return model
