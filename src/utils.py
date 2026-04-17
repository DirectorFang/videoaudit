# -*- coding: utf-8 -*-
"""
@Time ： 2026/3/20 15:03
@Auth ： chensonglin
@IDE ：PyCharm
  Describe:

"""
# -*- coding: utf-8 -*-
"""
@Time ： 2026/2/4 17:58
@Auth ： chensonglin
@IDE ：PyCharm
  Describe:

"""
from src.config import settings

from dashscope import Generation, MultiModalConversation,AioMultiModalConversation
from dashscope.aigc import AioGeneration
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type


class AlibabaLLM:
    def __init__(self):
        self.api_key = settings.dashscope.api_key

    # ========================
    # 通用响应解析（核心优化点）
    # ========================
    def _parse_response(self, response, model: str):
        if (
            response.status_code == 200
            and response.output
            and response.output.choices
            and len(response.output.choices) > 0
            and response.output.choices[0].message
        ):
            return response.output.choices[0].message.content[0]['text']
        else:
            raise Exception(f"{model} 调用失败: {response}")

    # ========================
    # 通用 message 构造
    # ========================
    def _build_messages(self, prompt, system_content):
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

    # ========================
    # 异步文本
    # ========================
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def async_chat(
        self,
        prompt: str,
        model="qwen-plus",
        system_content="You are a helpful assistant.",
        result_format="message",
        response_format=None,
    ):
        messages = self._build_messages(prompt, system_content)

        response = await AioGeneration.call(
            api_key=self.api_key,
            model=model,
            messages=messages,
            result_format=result_format,
            enable_thinking=False,
            response_format=response_format,
        )

        return self._parse_response(response, model)

    # ========================
    # 同步文本
    # ========================
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat(
        self,
        prompt,
        model="qwen-plus",
        system_content="You are a helpful assistant.",
        result_format="message",
        response_format=None,
    ):
        messages = self._build_messages(prompt, system_content)

        response = Generation.call(
            api_key=self.api_key,
            model=model,
            messages=messages,
            result_format=result_format,
            enable_thinking=False,
            response_format=response_format,
        )

        return self._parse_response(response, model)

    # ========================
    # 视频（多模态）
    # ========================
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat_with_video(
        self,
        prompt,
        video_path,
        fps=2,
        model="qwen3.5-plus",
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"video": video_path, "fps": fps},
                    {"text": prompt},
                ],
            }
        ]

        response = MultiModalConversation.call(
            api_key=self.api_key,
            model=model,
            messages=messages,
        )

        return self._parse_response(response, model)

        # ========================
        # 视频（多模态）
        # ========================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def async_chat_with_video(
            self,
            prompt,
            video_path,
            fps=2,
            model="qwen3.5-plus",
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"video": video_path, "fps": fps},
                    {"text": prompt},
                ],
            }
        ]

        response = await AioMultiModalConversation.call(
            api_key=self.api_key,
            model=model,
            messages=messages,
        )

        return self._parse_response(response, model)
#
# class OSSUTILS:
#     def __init__(self, username, password):
#         self.username = username
#         self.password = password
#
#
#
#     parser = argparse.ArgumentParser(description="Async put object from file sample")
#     parser.add_argument('--region', help='The region in which the bucket is located.', required=True)
#     parser.add_argument('--bucket', help='The name of the bucket.', required=True)
#     parser.add_argument('--endpoint', help='The domain names that other services can use to access OSS')
#     parser.add_argument('--key', help='The name of the object.', required=True)
#     parser.add_argument('--file_path', help='The path of Upload file.', required=True)
#
#     async def main(self):
#
#         # 1. 凭证与配置
#         credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
#         cfg = oss.config.load_default()
#         cfg.credentials_provider = credentials_provider
#         cfg.region = args.region
#         if args.endpoint:
#             cfg.endpoint = args.endpoint
#
#         # 2. 使用异步客户端
#         client = oss.AsyncClient(cfg)
#
#         try:
#             # 3. 异步上传文件（注意添加 await）
#             result = await client.put_object_from_file(
#                 oss.PutObjectRequest(
#                     bucket=args.bucket,
#                     key=args.key
#                 ),
#                 args.file_path
#             )
#
#             # 4. 打印结果
#             print(f'status code: {result.status_code}, '
#                   f'request id: {result.request_id}, '
#                   f'content md5: {result.content_md5}, '
#                   f'etag: {result.etag}, '
#                   f'hash crc64: {result.hash_crc64}, '
#                   f'version id: {result.version_id}, '
#                   f'server time: {result.headers.get("x-oss-server-time")}')
#         except Exception as e:
#             print(f"Upload failed: {e}")
#         finally:
#             # 异步客户端使用完毕后建议关闭底层连接池
#             await client.close()
#
#     if __name__ == "__main__":
#         asyncio.run(main())
