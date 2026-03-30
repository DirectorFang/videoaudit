from dotenv import load_dotenv

load_dotenv(dotenv_path="/tmp/pycharm_project_803/.env")
import os
from dashscope import MultiModalConversation
import dashscope
from prompts import prompt
# 各地域配置不同，请根据实际地域修改
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

# 将xxxx/test.mp4替换为你本地视频的绝对路径
local_path = "/tmp/pycharm_project_803/video/6be4f12931e69092bdfa22c49245e8fb_raw.mp4"
video_path = f"file://{local_path}"
messages = [
                {'role':'user',
                # fps参数控制视频抽帧数量，表示每隔1/fps 秒抽取一帧
                'content': [{'video': video_path,"fps":2},
                            {'text': prompt}]}]
print(messages)
response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen3.5-plus',
    messages=messages)
print(response.output.choices[0].message.content[0]["text"])