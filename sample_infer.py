import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.cli.model import CosyVoice2Model


tts_text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
instruct_text = '用四川话说这句话<|endofprompt|>'
prompt_wav = f'{ROOT_DIR}/asset/zero_shot_prompt.wav'

cosyvoice_wrapper = AutoModel(model_dir="FunAudioLLM/CosyVoice2-0.5B")

cosyvoice2model: CosyVoice2Model = cosyvoice_wrapper.model

instruct_text = cosyvoice_wrapper.frontend.text_normalize(instruct_text, split=False, text_frontend=True)
for i in cosyvoice_wrapper.frontend.text_normalize(tts_text, split=True, text_frontend=True):
    print(i)
    model_input = cosyvoice_wrapper.frontend.frontend_instruct2(
        tts_text= i, 
        instruct_text = instruct_text, 
        prompt_wav = prompt_wav, 
        resample_rate = cosyvoice_wrapper.sample_rate, 
        zero_shot_spk_id = ''
    )
