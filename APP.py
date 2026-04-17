from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.responses import JSONResponse

from src.utils import AlibabaLLM
from src.prompts import prompt1
import uvicorn

app = FastAPI()

llm = AlibabaLLM()


# ========================
# 请求体定义（强烈推荐）
# ========================
class VideoRequest(BaseModel):
    video_url: HttpUrl   # 自动校验URL是否合法
    fps: int = 2


# ========================
# 统一返回结构
# ========================
def success(data=None):
    return JSONResponse(
        status_code=200,
        content={
            "code": 0,
            "message": "success",
            "data": data
        }
    )


def error(code, message, http_status=200):
    return JSONResponse(
        status_code=http_status,
        content={
            "code": code,
            "message": message,
            "data": None
        }
    )


# ========================
# 接口
# ========================
@app.post("/score/video")
async def score_video(req: VideoRequest):
    """
    视频评分接口（URL模式）
    video_url: HttpUrl   # 视频url地址
    fps: int = 2 #视频解析速率 一般不用传，取默认速度

    return: 当前直接返回文本 后续若需其他格式再改
    """
    try:
        result = await llm.async_chat_with_video(
            prompt=prompt1,
            video_path=str(req.video_url),  # ⚠️ 传URL
            fps=req.fps,
        )

        return success(result)

    except Exception as e:
        return error(4001, f"LLM调用失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)