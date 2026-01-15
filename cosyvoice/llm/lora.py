from peft import LoraConfig, get_peft_model, PeftModel
from peft.utils import TaskType
from cosyvoice.llm.llm import Qwen2LM

def apply_lora_to_llm(
    model: Qwen2LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
)->PeftModel:
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
    if not hasattr(model, "prepare_inputs_for_generation"):
        def _prepare_inputs_for_generation(*args, **kwargs):
            raise RuntimeError(
                "prepare_inputs_for_generation should NOT be called in CosyVoice2"
            )
        model.prepare_inputs_for_generation = _prepare_inputs_for_generation
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model
