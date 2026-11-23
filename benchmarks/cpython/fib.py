"""CPython baseline for the fib workload. Prints elapsed milliseconds."""

import time


def fib(n: int) -> int:
    return n if n < 2 else fib(n - 1) + fib(n - 2)


start = time.perf_counter()
result = fib(25)
print(f"{result} {(time.perf_counter() - start) * 1000:.2f}")
