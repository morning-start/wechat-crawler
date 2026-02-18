# %%
from src.clawer import TimeRange, WeChatCrawler, save_all_article_content

crawler = WeChatCrawler.from_cookies_file("cookies.json")
gzh_names = [
    "国企求职网",
]
# bizs通过 temp/bizs.json 读取，如果不存在则缓存
bizs = crawler.load_or_fetch_bizs(gzh_names)
# 2026年2月1日至2026年2月18日
time_range = TimeRange(begin="2026-02-01", end="2026-02-18")
article_df = crawler.fetch_articles_info(bizs, time_range)
save_all_article_content(article_df, time_range=time_range)
# # %%
# for i in range(0, 11, 5):
#     print(i)
