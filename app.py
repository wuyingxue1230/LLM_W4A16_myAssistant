# python3
# Create Date: 2024-01-12
# Author: Scc_hy
# Func: web demo
# ==============================================================================
import gradio as gr
from dataclasses import asdict
from lmdeploy import turbomind as tm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from appPrepare.download_model import download_w4a16_chat7b, logger
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system(f'sh ./appPrepare/env_prepare.sh')
logger.info('1- Prepare w4a16-chat-7B-Model')
model_path = download_w4a16_chat7b()
logger.info('2- Download and Merged Adapater & base mode')
user_prompt = "<|User|>:{user}\n"
robot_prompt = "<|Bot|>:{robot}<eoa>\n"
cur_query_prompt = "<|User|>:{user}<eoh>\n<|Bot|>:"


# class Model_center():
#     def __init__(self):
#         # 构造函数，加载检索问答链
#         self.model = (
#             AutoModelForCausalLM.from_pretrained(mg_path, trust_remote_code=True)
#             .to(torch.bfloat16)
#             .cuda()
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(mg_path, trust_remote_code=True)

#     def qa_answer(self, question: str, chat_history: list = []):
#             if question == None or len(question) < 1:
#                 return "", chat_history
#             try:
#                 question = question.replace(" ", '')
#                 response, history = self.model.chat(
#                     self.tokenizer, 
#                     question, 
#                     history=chat_history
#                 )
#                 chat_history.append((question, response))
#                 return "", chat_history
#             except Exception as e:
#                 return e, chat_history


class Model_center():
    def __init__(self):
        # 构造函数，加载检索问答链
        self.tm_model = tm.TurboMind.from_pretrained(model_path, model_name='internlm-chat-7b')
    
    def _prompt(self, query):
        generator = self.tm_model.create_instance()
        # process query
        # query = "你是谁"
        prompt = self.tm_model.model.get_prompt(query)
        input_ids = self.tm_model.tokenizer.encode(prompt)
        # inference
        for outputs in generator.stream_infer(
                session_id=0,
                input_ids=[input_ids]):
            res, tokens = outputs[0]

        response = self.tm_model.tokenizer.decode(res.tolist())
        return response

    def qa_answer(self, question: str, chat_history: list = []):
            if question == None or len(question) < 1:
                return "", chat_history
            try:
                question = question.replace(" ", '')
                response = self._prompt(question)
                chat_history.append((question, response))
                return "", chat_history
            except Exception as e:
                return e, chat_history



# 实例化核心功能对象
model_center = Model_center()
# 创建一个 Web 界面
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            # 展示的页面标题
            gr.Markdown("""<h1><center>InternLMSccAssistant</center></h1>
                <center>书生浦语-SccAssistant</center>
                """)
    with gr.Row():
        with gr.Column(scale=4):
            # 创建一个聊天机器人对象
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
    
        db_wo_his_btn.click(model_center.qa_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)


gr.close_all()
# 直接启动
demo.launch()