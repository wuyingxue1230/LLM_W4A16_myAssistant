cd /root/personal_assistant

# 1- 计算统计量
lmdeploy lite calibrate \
  --model /root/personal_assistant/hf_merge \
  --calib_dataset "c4" \
  --calib_samples 128 \
  --calib_seqlen 2048 \
  --work_dir ./quant_minmax_info

# 2- 量化
lmdeploy lite auto_awq \
  --model  /root/personal_assistant/hf_merge \
  --w_bits 4 \
  --w_group_size 128 \
  --work_dir ./quant_minmax_info

# 3- 转TurboMind 直接用Python转
# lmdeploy convert  internlm-chat-7b ./quant_minmax_info \
#     --model-format awq \
#     --group-size 128 \
#     --dst_path ./workspace_w4a16

