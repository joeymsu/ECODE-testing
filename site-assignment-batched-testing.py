import time
import numpy as np
import pandas as pd
# from downstream import dstream

# imports - not working for now
from downstream.dstream.steady_algo._steady_lookup_ingest_times_batched import steady_lookup_ingest_times_batched
from downstream.dstream.stretched_algo._stretched_lookup_ingest_times_batched import stretched_lookup_ingest_times_batched
from downstream.dstream.tilted_algo._tilted_lookup_ingest_times_batched import tilted_lookup_ingest_times_batched

records = []

def measure_execution_time(func, S: int, T: np.ndarray):
    t1 = (time.perf_counter(), time.process_time())
    result = func(S, T, parallel=False)  
    t2 = (time.perf_counter(), time.process_time())

    realTime = t2[0] - t1[0]  # no rounding
    cpuTime = t2[1] - t1[1]   # no rounding

    records.append({"Call To Function": f"{func.__name__}(S={S}, Tsize={len(T)})", "Selected Site(s)": result.shape, "Real Execution Time": realTime, "CPU Execution Time": cpuTime})

# batched lookup algorithms
algorithms = [("steady", dstream.steady_algo.lookup_ingest_times_batched), ("stretched", dstream.stretched_algo.lookup_ingest_times_batched), ("tilted", dstream.tilted_algo.lookup_ingest_times_batched),]

surface_sizes = [64, 256, 1024]

np.random.seed(0)

for algo_name, algo_func in algorithms:
    for S in surface_sizes:
        if S == 256:
            # S=256, test T in [256, 2**16] only
            T_16 = np.random.randint(S, 2**16, size=256, dtype=np.int64)
            measure_execution_time(algo_func, S, T_16)
        else:
            # S=64 or S=1024, test T in [S, 2**32]
            T_32 = np.random.randint(S, 2**32, size=256, dtype=np.int64)
            measure_execution_time(algo_func, S, T_32)

df = pd.DataFrame.from_records(records)
print(df)
