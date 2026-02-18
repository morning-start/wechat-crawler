from pydantic import Field

from .common import BaseResponse, WeChatAPIError


class TokenError(WeChatAPIError):
    """Token获取失败异常"""

    pass


class TokenResponse(BaseResponse):
    """Token获取响应"""

    redirect_url: str = Field(description="重定向URL（包含token）")
