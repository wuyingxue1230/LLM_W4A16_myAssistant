# 1- 执行目录 /root/personal_assistant
# 2- 创建 configuration.json # cp config.json configuration.json
#     创建 README.md
# 3- apt install git git-lfs -y
# git lfs install
# 4- ModelScop创建模型
from modelscope.hub.api import HubApi

# 请从ModelScope个人中心->访问令牌获取'
YOUR_ACCESS_TOKEN = 'e8da05a6-909f-4655-a38a-794ab95a89f4'

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
api.push_model(
    model_id="sccHyFuture/LLM_W4A16_7B", 
    model_dir="./quant_minmax_info"
)