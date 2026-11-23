"""CPython baseline for the loop_sum workload. Prints elapsed milliseconds."""

import time

start = time.perf_counter()
total = 0
for i in range(5_000_000):
    total = total + i
print(f"{total} {(time.perf_counter() - start) * 1000:.2f}")
