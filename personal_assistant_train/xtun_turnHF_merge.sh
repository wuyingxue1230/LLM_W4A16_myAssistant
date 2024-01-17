
# 微调
xtuner train internlm_chat_7b_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2

# 转hf
cd /root/personal_assistant
mkdir hf

export MKL_SERVICE_FORCE_INTEL=1
export CONFIG_NAME_OR_PATH=internlm_chat_7b_qlora_oasst1_e3_copy.py
export PTH=./work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_1.pth
export SAVE_PATH=./hf
xtuner convert pth_to_hf $CONFIG_NAME_OR_PATH $PTH $SAVE_PATH

# merge 
cd /root/personal_assistant
mkdir hf_merge

export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'
export NAME_OR_PATH_TO_LLM=./internlm-chat-7b
export NAME_OR_PATH_TO_ADAPTER=./hf
export SAVE_PATH=./hf_merge

xtuner convert merge \
    $NAME_OR_PATH_TO_LLM \
    $NAME_OR_PATH_TO_ADAPTER \
    $SAVE_PATH \
    --max-shard-size 2GB

# test
xtuner chat ./hf_merge --prompt-template internlm_chat




