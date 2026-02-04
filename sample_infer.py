import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.cli.model import CosyVoice2Model


tts_text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
instruct_text = '用四川话说这句话<|endofprompt|>'


cosyvoice_wrapper = AutoModel(model_dir="iic/CosyVoice2-0.5B")

cosyvoice2model: CosyVoice2Model = cosyvoice_wrapper.model

for i in cosyvoice_wrapper.frontend.text_normalize(tts_text, split=True, text_frontend=True):
    print(i)