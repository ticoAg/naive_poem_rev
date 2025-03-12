import time
import asyncio
from functools import wraps
from typing import Callable, Any

from openai import AsyncOpenAI


def timer_decorator(func: Callable) -> Callable:
    """
    计时装饰器，支持同步函数和异步函数。

    :param func: 被装饰的函数。
    :return: 装饰后的函数。
    """

    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        """
        同步函数的包装器。
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        cost = end_time = time.perf_counter() - start_time
        if cost > 0.01:
            print(f"Function {func.__name__} executed in {cost:.4f} seconds")
        return result

    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        """
        异步函数的包装器。
        """
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        cost = end_time = time.perf_counter() - start_time
        if cost > 0.01:
            print(f"Async function {func.__name__} executed in {cost:.4f} seconds")
        return result

    # 根据函数类型选择合适的包装器
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def chunked_list(lst: list[Any], _chunk_size: int = 32) -> Any:
    """
    将给定的列表按照指定的块大小分割成多个子列表。

    :param lst: 需要被分割的列表。
    :param _chunk_size: 每个子列表的大小。
    :return: 生成器，用于迭代地获取每个子列表。
    """
    for i in range(0, len(lst), _chunk_size):
        yield lst[i : i + _chunk_size]


if __name__ == "__main__":

    @timer_decorator
    def sync_example_function(n: int):
        """一个简单的同步函数示例"""
        total = 0
        for i in range(n):
            total += i
        return total

    @timer_decorator
    async def async_example_function(n: int):
        """一个简单的异步函数示例"""
        await asyncio.sleep(1)
        return n * 2

    # 测试同步函数
    result_sync = sync_example_function(1000000)
    print(f"Sync function result: {result_sync}")

    # 测试异步函数
    result_async = asyncio.run(async_example_function(5))
    print(f"Async function result: {result_async}")
