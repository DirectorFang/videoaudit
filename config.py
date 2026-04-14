from pydantic_settings import BaseSettings
from pydantic import ConfigDict


# =======================
# 阿里云百炼（DashScope）
# =======================
env_file = ".env"
env_file_encoding ="utf-8"
class DashScopeSettings(BaseSettings):
    api_key: str

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
    access_key_id: str
    access_key_secret: str
    bucket_name: str

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
    user: str
    password: str
    host: str
    port: int
    http_port: int

    model_config = ConfigDict(
        env_prefix="DORIS_",
        env_file=env_file,
        env_file_encoding=env_file_encoding,
        extra="ignore"
    )


# =======================
# 总配置入口（推荐写法）
# =======================
class Settings:
    def __init__(self):
        self.dashscope = DashScopeSettings()
        self.oss = OSSSettings()
        self.doris = DorisSettings()


settings = Settings()


if __name__ == '__main__':
    # ⚠️ 不建议打印敏感信息，这里只做调试示例
    print(settings.dashscope.model_dump())
    print(settings.oss.model_dump())
    print(settings.doris.model_dump())
    print(settings.dashscope.api_key)