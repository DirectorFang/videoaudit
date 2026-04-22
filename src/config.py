from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field
from pathlib import Path


# =======================
# env 文件路径
# =======================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_file = PROJECT_ROOT / ".env"
env_file_encoding = "utf-8"


# =======================
# 阿里云百炼（DashScope）
# =======================
class DashScopeSettings(BaseSettings):
    api_key: str = Field(default="")

    model_config = ConfigDict(
        env_prefix="DASHSCOPE_",
        env_file=env_file,
        env_file_encoding=env_file_encoding,
        extra="ignore"
    )


# =======================
# OSS 配置
# =======================
class OSSSettings(BaseSettings):
    access_key_id: str = Field(default="")
    access_key_secret: str = Field(default="")
    bucket_name: str = Field(default="")

    model_config = ConfigDict(
        env_prefix="OSS_",
        env_file=env_file,
        env_file_encoding=env_file_encoding,
        extra="ignore"
    )


# =======================
# Doris 数据库
# =======================
class DorisSettings(BaseSettings):
    user: str = Field(default="root")
    password: str = Field(default="")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=9030)
    http_port: int = Field(default=8030)

    model_config = ConfigDict(
        env_prefix="DORIS_",
        env_file=env_file,
        env_file_encoding=env_file_encoding,
        extra="ignore"
    )


# =======================
# 总配置入口（优化版）
# =======================
class Settings:
    def __init__(self):
        self.dashscope = DashScopeSettings()
        self.oss = OSSSettings()
        self.doris = DorisSettings()

        # 👉 可选：启动时提示哪些是默认值（很有用）
        self._warn_if_default()

    def _warn_if_default(self):
        if not self.dashscope.api_key:
            print("[WARN] DASHSCOPE_API_KEY 未配置")

        if not self.oss.access_key_id:
            print("[WARN] OSS 未配置")

        if self.doris.host == "127.0.0.1":
            print("[WARN] Doris 使用默认本地配置")


settings = Settings()


if __name__ == '__main__':
    # ⚠️ 不建议打印敏感信息，这里只做调试示例
    print(settings.dashscope.model_dump())
    print(settings.oss.model_dump())
    print(settings.doris.model_dump())
    print(settings.dashscope.api_key)
