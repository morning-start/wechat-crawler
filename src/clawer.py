import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from re import M
from typing import Literal

import pandas as pd
import requests
from fake_useragent import UserAgent
from loguru import logger
from pydantic import BaseModel, field_serializer, field_validator
from tqdm import tqdm

from .tools import load_json, sanitize_filename, save_article_content, save_json
from .wechat_api import ArticleListItem, SearchBizError, TokenError, WeChatAPI


class TimeRange(BaseModel):
    """文章时间范围，用于缓存管理"""

    begin: datetime
    end: datetime

    @field_serializer("begin", "end")
    def serialize_datetime(self, dt: datetime) -> str:
        """将 datetime 对象序列化为 YYYY-MM-DD 格式字符串"""
        return dt.strftime("%Y-%m-%d")


class WeChatCrawler(WeChatAPI):
    def __init__(self, cookies: dict[str, str]) -> None:
        super().__init__(cookies)
        try:
            self.token = self._get_token()
            logger.info(f"获取token成功: {self.token}")
        except TokenError as e:
            logger.error(f"❌ 获取token失败: {str(e)}")
            raise

    @classmethod
    def from_cookies_file(cls, file_path: str) -> "WeChatCrawler":
        data = load_json(file_path)
        cookies = data["请求 Cookie"]
        return cls(cookies)

    def load_or_fetch_bizs(
        self, gzh_names: list[str], cache_file: Path = Path("temp/fakeids.json")
    ) -> dict[str, str]:
        """
        加载或获取公众号fakeid映射（带缓存优化）

        Args:
            gzh_names: 公众号名称列表
            cache_file: 缓存文件路径

        Returns:
            公众号名称到fakeid的映射
        """
        required_names = set(gzh_names)
        bizs = {}

        if cache_file.exists():
            logger.info(f"从缓存加载fakeids: {cache_file}")
            cached_bizs = load_json(cache_file)
            required_cached_bizs = {
                name: cached_bizs[name]
                for name in required_names & set(cached_bizs.keys())
            }
            bizs.update(required_cached_bizs)
        need_names = required_names - set(bizs.keys())
        if need_names:
            logger.info(f"从网络获取fakeids: {need_names}")
            for name in need_names:
                try:
                    result = self.search_fakeid(name)
                    if result.list:
                        nickname = result.list[0].nickname
                        fakeid = result.list[0].fakeid
                        bizs[nickname] = fakeid
                        logger.info(f"成功获取公众号: {nickname} -> {fakeid}")
                    else:
                        logger.warning(f"公众号搜索结果为空: {name}")
                except SearchBizError as e:
                    logger.error(f"搜索公众号失败: {name}, 错误: {str(e)}")
            save_json(bizs, cache_file)

        return bizs

    def fetch_article_list(
        self, fakeid: str, begin: int, count: int
    ) -> list[ArticleListItem]:
        """
        获取文章列表

        Args:
            fakeid: 公众号fakeid
            begin: 列表起始位置
            count: 返回数量

        Returns:
            过滤后的文章列表
        """
        articles = self.search_article_list(fakeid, begin, count)
        valid_articles = [
            article
            for article in articles.app_msg_list
            if self.is_valid_article_link(article.link)
        ]
        return valid_articles

    def fetch_articles(
        self,
        fakeid: str,
        max_count: int | None = None,
        time_range: TimeRange | None = None,
    ) -> list[ArticleListItem]:
        """
        加载或获取文章链接列表（带缓存优化）

        Args:
            nickname: 公众号名称
            fakeid: 公众号fakeid
            max_count: 最大获取数量限制
            time_range: 时间范围限制

        Returns:
            文章链接列表
        """
        EACH_COUNT = 5
        # article 中有时间属性，获取所有在时间范围的文章信息，
        all_articles: list[ArticleListItem] = []
        begin = 0
        while True:
            # 获取到的文章都是倒叙排列，就是由近到远的顺序，越在后面的越早
            articles = self.fetch_article_list(fakeid, begin, EACH_COUNT)
            all_articles += articles
            begin += EACH_COUNT
            # 如果articles为空list，说明超出范围，停止获取
            if not articles:
                logger.info(f"获取到的文章为空，结束获取")
                break
            # 如果时间范围限制存在，超过start_date，停止获取
            if time_range and articles[-1].create_time < time_range.begin.timestamp():
                break
            # 如果最大数量限制存在，超过最大数量，停止获取
            if max_count and len(all_articles) >= max_count:
                break

        return all_articles

    # 剩余的时间范围 meta_file,start_time,end_time
    def _get_remaining_time_range(
        self, meta_file: Path, time_range: TimeRange
    ) -> tuple[TimeRange, TimeRange]:
        """
        获取剩余的时间范围

        Args:
            meta_file: 元数据文件路径
            time_range: 时间范围

        Returns:
            剩余的时间范围（开始日期，结束日期）
        """
        if not meta_file.exists():
            return time_range, time_range
        meta_info = TimeRange(**load_json(meta_file))

        # 情况1: 完全没有重叠
        if meta_info.end < time_range.begin or time_range.end < meta_info.begin:
            remaining_range = TimeRange(begin=time_range.begin, end=time_range.end)
            meta_info.begin = time_range.begin
            meta_info.end = time_range.end
            return remaining_range, meta_info
        # 情况2: 缓存在请求范围内，需要扩展（请求的开始日期在缓存内，但结束日期超出）
        elif meta_info.begin < time_range.begin < meta_info.end < time_range.end:
            remaining_range = TimeRange(begin=meta_info.end, end=time_range.end)
            meta_info.end = time_range.end
            return remaining_range, meta_info
        # 情况3: 缓存在请求范围内，需要扩展（请求的结束日期在缓存内，但开始日期超出）
        elif meta_info.begin < time_range.end < meta_info.end:
            remaining_range = TimeRange(begin=time_range.begin, end=meta_info.begin)
            meta_info.begin = time_range.begin
            return remaining_range, meta_info
        # 情况4: 完全在范围内（无需获取）
        else:
            return None, meta_info

    def fetch_articles_info(
        self,
        bizs: dict[str, str],
        time_range: TimeRange,
        save_dir: Path = Path("temp/articles_info/"),
    ) -> pd.DataFrame:
        """
        获取文章信息（不带缓存优化）

        Args:
            bizs: 公众号名称到fakeid的映射
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            文章信息DataFrame
        """
        # 创建保存目录
        save_dir.mkdir(parents=True, exist_ok=True)
        for nickname, fakeid in bizs.items():
            safe_nickname = sanitize_filename(nickname)
            save_path = save_dir / f"{safe_nickname}.csv"
            meta_path = save_dir / f"{safe_nickname}.json"
            remaining_range, new_meta_info = self._get_remaining_time_range(
                meta_path, time_range
            )
            if remaining_range is None:
                logger.info(f"公众号 {nickname} 已经获取到文章，跳过")
                continue
            articles = self.fetch_articles(fakeid, time_range=remaining_range)
            if not articles:
                logger.warning(f"公众号 {nickname} 没有获取到有效文章")
                continue

            df_articles = pd.DataFrame([article.model_dump() for article in articles])

            # 如果文件存在，则合并
            if save_path.exists():
                df_existing = pd.read_csv(save_path)
                df_articles = pd.concat([df_articles, df_existing], ignore_index=True)

            # 去重
            df_articles = df_articles.drop_duplicates(
                subset=["title"], keep="first", ignore_index=True
            )
            # 按照时间排序
            df_articles["create_time"] = pd.to_datetime(df_articles["create_time"])
            df_articles = df_articles.sort_values(
                by="create_time", ascending=False, ignore_index=True
            )

            # 保存到缓存文件
            df_articles.to_csv(save_path, index=False, encoding="utf-8-sig")
            # 保存元数据
            save_json(new_meta_info.model_dump(), meta_path)
        # 合并文件夹下所有csv文件，并且 nickname 列为对应公众号名称
        df = pd.concat(
            [pd.read_csv(f).assign(nickname=f.stem) for f in save_dir.glob("*.csv")],
            ignore_index=True,
        )
        return df


def download_article_content(task_data: dict) -> bool:
    """
    保存文章内容到Markdown文件

    Args:
        task_data: 包含以下键的字典：
            - url: 文章链接
            - title: 文章标题
            - save_dir: 保存目录
            - save_file: 保存格式（md 或 html）
            - max_retries: 最大重试次数
            - timeout: 请求超时时间（秒）
            - date_str: 日期字符串
            - account_name: 公众号名称
            - digest: 文章摘要
            - min_file_size_kb: 最小文件大小（KB）

    Returns:
        是否成功保存
    """
    url = task_data["url"]
    title = task_data["title"]
    save_dir = task_data["save_dir"]
    save_file = task_data.get("save_file", "md")
    max_retries = task_data.get("max_retries", 3)
    timeout = task_data.get("timeout", 30)
    date_str = task_data.get("date_str", "")
    account_name = task_data.get("account_name", "")
    digest = task_data.get("digest", "")
    min_file_size_kb = task_data.get("min_file_size_kb", 3)

    save_dir.mkdir(parents=True, exist_ok=True)

    safe_title = sanitize_filename(title)
    save_path = save_dir / f"{safe_title}.{save_file}"

    if save_path.exists():
        return True

    headers = {"User-Agent": UserAgent().random}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            result = save_article_content(
                response.text,
                save_path,
                save_file,
                title=title,
                date_str=date_str,
                link=url,
                account_name=account_name,
                digest=digest,
                min_file_size_kb=min_file_size_kb,
            )
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"获取文章内容失败（重试{max_retries}次后）: {title}, 错误: {e}"
                )
                return False
            time.sleep(1)
    return False


def save_all_article_content(
    df: pd.DataFrame,
    save_dir: Path = Path("temp/article_content/"),
    max_workers: int = 5,
    time_range: TimeRange = None,
    save_file: Literal["md", "html"] = "md",
    min_file_size_kb: int = 3,
):
    """
    保存所有文章内容到Markdown文件（并发下载）

    Args:
        df: 包含文章信息的DataFrame
        save_dir: 保存目录
        max_workers: 最大并发数
        time_range: 时间范围
        save_file: 保存格式（md 或 html）
        min_file_size_kb: 最小文件大小（KB）
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    # 转化为 datetime 类型
    df["create_time"] = pd.to_datetime(df["create_time"])
    if time_range:
        df = df[
            (df["create_time"] >= time_range.begin)
            & (df["create_time"] <= time_range.end)
        ]

    tasks = []
    for _, row in df.iterrows():
        safe_nickname = sanitize_filename(row["nickname"])
        task_data = {
            "url": row["link"],
            "title": row["title"],
            "save_dir": save_dir / safe_nickname,
            "save_file": save_file,
            "max_retries": 3,
            "timeout": 30,
            "date_str": row.get("create_time", ""),
            "account_name": row.get("nickname", ""),
            "digest": row.get("digest", ""),
            "min_file_size_kb": min_file_size_kb,
        }
        tasks.append((task_data, row["link"], row["title"]))

    success_count = 0
    fail_count = 0
    skip_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_article_content, task_data): (url, title)
            for task_data, url, title in tasks
        }

        with tqdm(total=len(futures), desc="下载文章", unit="篇") as pbar:
            for future in as_completed(futures):
                url, title = futures[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    logger.error(f"处理文章时发生异常: {title}, 错误: {e}")
                pbar.update(1)

    logger.info(
        f"文章下载完成: 成功 {success_count} 篇, 失败 {fail_count} 篇, "
        f"跳过 {skip_count} 篇, 总计 {len(tasks)} 篇"
    )
