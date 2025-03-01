import time
import asyncio
from functools import wraps
from typing import Callable, Any


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


# 示例使用
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
