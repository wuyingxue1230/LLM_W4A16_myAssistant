from lmdeploy import turbomind as tm

# load model
model_path = "/root/personal_assistant/quant_minmax_info"
tm_model = tm.TurboMind.from_pretrained(model_path, model_name='internlm-chat-7b')
generator = tm_model.create_instance()

# process query
query = "你是谁"
prompt = tm_model.model.get_prompt(query)
input_ids = tm_model.tokenizer.encode(prompt)

# inference
for outputs in generator.stream_infer(
        session_id=0,
        input_ids=[input_ids]):
    res, tokens = outputs[0]

response = tm_model.tokenizer.decode(res.tolist())
print(response)


res_ = """
[TM][INFO] [forward] Enqueue requests
[TM][INFO] [forward] Wait for requests to complete ...
[TM][WARNING] [ProcessInferRequests] Request for 0 received.
[TM][INFO] [Forward] [0, 1), dc_bsz = 0, pf_bsz = 1, n_tok = 106, max_q = 106, max_k = 106
[TM][INFO] ------------------------- step = 110 -------------------------
[TM][INFO] ------------------------- step = 120 -------------------------
[TM][INFO] [Interrupt] slot = 0, id = 0
[TM][INFO] [forward] Request complete for 0, code 0
>>> response = tm_model.tokenizer.decode(res.tolist())
>>> print(response)
我是Scc_hy的小助手，内在是上海AI实验室书生·浦语的7B大模型哦
>>> 
>>> 
"""