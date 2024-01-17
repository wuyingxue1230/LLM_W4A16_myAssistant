import os
from modelscope.hub.snapshot_download import snapshot_download
from transformers.utils import logging


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
logger = logging.get_logger(__name__)


def download_w4a16_chat7b():
    # modelscope download 
    quant_model_path = "/home/xlab-app-center/hf"
    if not os.path.exists(quant_model_path):
        snapshot_download(model_id='sccHyFuture/LLM_W4A16_7B', cache_dir=quant_model_path)
        
    quant_model_final_path = f'{quant_model_path}/sccHyFuture/LLM_W4A16_7B'
    if os.path.exists(f'{quant_model_final_path}/configuration.json'):
        os.system(f'rm {quant_model_final_path}/configuration.json')
    
    return quant_model_final_path
    