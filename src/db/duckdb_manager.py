# -*- encoding: utf-8 -*-
"""
@Time    :   2025-03-01 22:16:47
@desc    :   
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""


import duckdb
import pickle
from typing import List, Optional

from utils.tools import timer_decorator


class DuckDBCache:
    def __init__(self, db_path: str = ".cache/duck.db", table_name: str = "cache"):
        """
        初始化 DuckDB 缓存系统。

        :param db_path: 数据库文件路径，默认为内存模式（':memory:'）。
        :param table_name: 缓存表的名称，默认为 'cache'。
        """
        self.db_path = db_path
        self.table_name = table_name
        self.conn = duckdb.connect(db_path)
        self._initialize_table()

    @timer_decorator
    def _initialize_table(self):
        """
        初始化缓存表结构。
        表结构：key (字符串主键), value (二进制数据)。
        """
        self.conn.execute(
            f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key VARCHAR PRIMARY KEY,
                    value BLOB
                )
            """
        )

    @timer_decorator
    def cache_result(self, key: str, value: List[float]):
        """
        将结果缓存到数据库中。

        :param key: 缓存的键（字符串）。
        :param value: 缓存的值（浮点数列表）。
        """
        serialized_value = pickle.dumps(value)
        self.conn.execute(
            f"""
                INSERT OR REPLACE INTO {self.table_name} (key, value)
                VALUES (?, ?)
            """,
            (key, serialized_value),
        )

    @timer_decorator
    def get_cached_result(self, key: str) -> Optional[List[float]]:
        """
        根据键获取缓存的结果。

        :param key: 缓存的键（字符串）。
        :return: 缓存的值（浮点数列表），如果不存在则返回 None。
        """
        result = self.conn.execute(
            f"""
        SELECT value FROM {self.table_name}
        WHERE key = ?
        """,
            (key,),
        ).fetchone()
        if result:
            return pickle.loads(result[0])
        return None

    @timer_decorator
    def clear_cache(self):
        """
        清空缓存表中的所有数据。
        """
        self.conn.execute(f"DELETE FROM {self.table_name}")

    def close(self):
        """
        关闭数据库连接。
        """
        self.conn.close()

    def __enter__(self):
        """
        支持上下文管理协议（with 语句）。
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理退出时关闭数据库连接。
        """
        self.close()


# 示例使用
if __name__ == "__main__":
    # 使用内存模式
    with DuckDBCache() as cache:
        # 缓存数据
        cache.cache_result("example_key", [1.0] * 1024)
        # 获取缓存
        result = cache.get_cached_result("example_key")
        print(result)

    # 使用持久化模式
    with DuckDBCache(db_path="cache.db") as cache:
        cache.cache_result("persistent_key", [2.0] * 1024)
        result = cache.get_cached_result("persistent_key")
        print(result)
