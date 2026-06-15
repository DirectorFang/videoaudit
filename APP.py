from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.responses import JSONResponse

from src.utils import AlibabaLLM
from src.prompts import (
    pit_fissure_sealing as pit_fissure_sealing_prompt,
    periodontal_probing as periodontal_probing_prompt,
    oral_mucosa_disinfection as oral_mucosa_disinfection_prompt,
    general_inspection_visual_examination as general_inspection_visual_examination_prompt,
    submandibular_gland_examination as submandibular_gland_examination_prompt,
)
import uvicorn

app = FastAPI()

llm = AlibabaLLM()


# ========================
# 请求体定义（强烈推荐）
# ========================
class VideoRequest(BaseModel):
    video_url: HttpUrl  # 自动校验URL是否合法
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


def build_prompt_with_audio_transcript(prompt: str, audio_transcript: str) -> str:
    print(audio_transcript)
    transcript = audio_transcript.strip() or "未识别到明确的音频转写内容。"
    return f"""{prompt}

--------------------------------------------------
【音频转写内容】
以下内容来自视频音轨自动转写，仅用于辅助判断口述内容、探诊结果描述、疼痛反馈等需要听音频才能确认的评分项。
如果转写内容与画面存在冲突，应结合视频画面审慎判断，并在报告中说明依据。

{transcript}
"""


# ========================
# 接口
# ========================
@app.post("/score/pit_fissure_sealing")
async def pit_fissure_sealing(req: VideoRequest):
    """
    视频评分接口（URL模式）
    video_url: HttpUrl   # 视频url地址
    fps: int = 2 #视频解析速率 一般不用传，取默认速度

    return: 当前直接返回文本 后续若需其他格式再改
    """
    try:
        result = await llm.async_chat_with_video(
            prompt=pit_fissure_sealing_prompt,
            video_path=str(req.video_url),  # ⚠️ 传URL
            fps=req.fps,
        )

        return success(result)

    except Exception as e:
        return error(4001, f"LLM调用失败: {str(e)}")


# ========================
# 接口
# ========================
@app.post("/score/periodontal_probing")
async def periodontal_probing(req: VideoRequest):
    """
    牙周探诊视频评分接口（URL模式）
    video_url: HttpUrl   # 视频url地址
    fps: int = 2 #视频解析速率 一般不用传，取默认速度

    return: 当前直接返回文本 后续若需其他格式再改
    """
    try:
        audio_transcript = await llm.async_transcribe_audio_from_url(
            file_url=str(req.video_url),
        )
        prompt = build_prompt_with_audio_transcript(
            periodontal_probing_prompt,
            audio_transcript,
        )
        result = await llm.async_chat_with_video(
            prompt=prompt,
            video_path=str(req.video_url),  # ⚠️ 传URL
            fps=req.fps,
        )

        return success(result)

    except Exception as e:
        return error(4001, f"LLM调用失败: {str(e)}")


# ========================
# 接口
# ========================
@app.post("/score/oral_mucosa_disinfection")
async def oral_mucosa_disinfection(req: VideoRequest):
    """
    口腔粘膜消毒视频评分接口（URL模式）
    video_url: HttpUrl   # 视频url地址
    fps: int = 2 #视频解析速率 一般不用传，取默认速度

    return: 当前直接返回文本 后续若需其他格式再改
    """
    try:
        audio_transcript = await llm.async_transcribe_audio_from_url(
            file_url=str(req.video_url),
        )
        prompt = build_prompt_with_audio_transcript(
            oral_mucosa_disinfection_prompt,
            audio_transcript,
        )
        result = await llm.async_chat_with_video(
            prompt=prompt,
            video_path=str(req.video_url),  # ⚠️ 传URL
            fps=req.fps,
        )

        return success(result)

    except Exception as e:
        return error(4001, f"LLM调用失败: {str(e)}")


# ========================
# 接口
# ========================
@app.post("/score/general_inspection_visual_examination")
async def general_inspection_visual_examination(req: VideoRequest):
    """
    一般检查-视诊视频评分接口（URL模式）
    video_url: HttpUrl   # 视频url地址
    fps: int = 2 #视频解析速率 一般不用传，取默认速度

    return: 当前直接返回文本 后续若需其他格式再改
    """
    try:
        audio_transcript = await llm.async_transcribe_audio_from_url(
            file_url=str(req.video_url),
        )
        prompt = build_prompt_with_audio_transcript(
            general_inspection_visual_examination_prompt,
            audio_transcript,
        )
        result = await llm.async_chat_with_video(
            prompt=prompt,
            video_path=str(req.video_url),  # ⚠️ 传URL
            fps=req.fps,
        )

        return success(result)

    except Exception as e:
        return error(4001, f"LLM调用失败: {str(e)}")


# ========================
# 接口
# ========================
@app.post("/score/submandibular_gland_examination")
async def submandibular_gland_examination(req: VideoRequest):
    """
    下颌下腺检查视频评分接口（URL模式）
    video_url: HttpUrl   # 视频url地址
    fps: int = 2 #视频解析速率 一般不用传，取默认速度

    return: 当前直接返回文本 后续若需其他格式再改
    """
    try:
        audio_transcript = await llm.async_transcribe_audio_from_url(
            file_url=str(req.video_url),
        )
        prompt = build_prompt_with_audio_transcript(
            submandibular_gland_examination_prompt,
            audio_transcript,
        )
        result = await llm.async_chat_with_video(
            prompt=prompt,
            video_path=str(req.video_url),  # ⚠️ 传URL
            fps=req.fps,
        )

        return success(result)

    except Exception as e:
        return error(4001, f"LLM调用失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
