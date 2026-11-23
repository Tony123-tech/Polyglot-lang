"""CPython baseline for the strings workload. Prints elapsed milliseconds."""

import time

text = "abc" * 50
start = time.perf_counter()
hits = 0
for _ in range(20_000):
    hits = hits + text.replace("a", "b").count("b")
print(f"{hits} {(time.perf_counter() - start) * 1000:.2f}")
