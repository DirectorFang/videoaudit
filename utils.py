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
import os
from dashscope.aigc.generation import AioGeneration
from http import HTTPStatus
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type



class AlibabaLLMAsync:
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")

    async def chat(self, prompt: str, model="qwen-plus",
                   system_content="You are a helpful assistant.",
                   result_format="message", response_format=None):
        """
        使用异步 SDK 调用通义千问
        """
        response = await AioGeneration.call(
            api_key=self.api_key,
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            result_format=result_format,
            enable_thinking=False,
            response_format=response_format
        )

        if (
                response.status_code == 200
                and response.output
                and response.output.choices
                and len(response.output.choices) > 0
                and response.output.choices[0].message
        ):
            return response.output.choices[0].message.content
        else:
            raise Exception(f"{model} async chat failed {response}")


class AlibabaLLM:
    # 若使用新加坡地域的模型，请取消以下注释
    # dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def rerank(self, query: str, documents: list[str], top_n: int = None, model="gte-rerank-v2"):
        resp = dashscope.TextReRank.call(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=False
        )
        if resp.status_code == HTTPStatus.OK:
            # return resp.output.results
            result = [
                {"index": result.index, "score": result.relevance_score}
                for result in resp.output.results
            ]
            # print(result)
            return result
        else:
            raise Exception(f"{model} rerank模型调用失败 {resp}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def embedding(self, input_text: str, model="text-embedding-v4", dimension=1024):
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=input_text,
            dimension=dimension
        )

        if resp.status_code == HTTPStatus.OK:
            # print(resp.output)

            return resp.output['embeddings'][0]['embedding']
        raise Exception(f"{model} embedding模型调用失败 {resp}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def embedding_with_batch(self, input_text: list[str], model="text-embedding-v4", dimension=1024):
        # 返回的结果下标与传入时相同 不会乱序
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=input_text,
            dimension=dimension
        )

        if resp.status_code == HTTPStatus.OK:
            # print(resp.output)
            result = [
                result['embedding']
                for result in resp.output['embeddings']
            ]
            return result
        raise Exception(f"{model} embedding模型调用失败 {resp}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat(self, prompt, model="qwen-plus",
             system_content="You are a helpful assistant.", result_format="message", response_format=None):
        messages = [
            {"role": "system", "content":
                system_content},
            {"role": "user", "content": prompt},
        ]
        response = Generation.call(
            api_key=self.api_key,
            model=model,
            messages=messages,
            result_format=result_format,
            enable_thinking=False,
            response_format=response_format
        )
        if (
                response.status_code == 200
                and response.output
                and response.output.choices
                and len(response.output.choices) > 0
                and response.output.choices[0].message
        ):
            return response.output.choices[0].message.content
        else:
            raise Exception(f"{model} chat模型调用失败 {response}")

