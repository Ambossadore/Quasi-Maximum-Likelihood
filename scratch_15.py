import numpy as np
from time import time
from functools import partial
from tqdm import tqdm, trange
import multiprocessing as mp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

N_processes = 32


def batch_simulation(N_batch, seed):
    sims = []
    np.random.seed(seed)
    for i in range(N_batch):
        a = np.random.weibull(a=100, size=(1000, 1000))
        sims.append(a)
    return np.stack(sims)


def batch_simulation_simple(seed):
    np.random.seed(seed)
    return np.random.weibull(a=100, size=(1000, 1000))[:10, :10]


B = np.ones(shape=(1000, 10, 10))


def batch_simulation_set(seed):
    global B
    np.random.seed(seed)
    B[seed] = np.random.weibull(a=100, size=(1000, 1000))[:10, :10]


def batch_simulation_simple_joblib(seed, pbar):
    np.random.seed(seed)
    pbar.update()
    return np.random.weibull(a=100, size=(1000, 1000))[:10, :10]


def batch_simulation_inpu(inpu):
    return inpu


def my_callback(_):
    global pbar
    pbar.update()


def full_simulation_test(N_batch, N_sim):
    pool = mp.Pool(processes=N_processes)
    global pbar
    pbar = tqdm(total=N_sim)
    future_results = [pool.apply_async(batch_simulation, args=(N_batch, seed,), callback=my_callback) for seed in range(N_sim)]
    results = [f.get() for f in future_results]
    return results


def full_simulation_test_simple(N_sim):
    pool = mp.Pool(processes=N_processes)
    global pbar
    pbar = tqdm(total=N_sim)
    future_results = [pool.apply_async(batch_simulation_simple, args=(seed,), callback=my_callback) for seed in range(N_sim)]
    results = [f.get() for f in future_results]
    pbar.close()
    return results


def full_simulation(N_batch, N_sim):
    # multiprocessing magic
    pool = mp.Pool(processes=N_processes)
    future_results = [pool.apply_async(batch_simulation, (N_batch, seed, )) for seed in range(N_sim)]
    results = [f.get() for f in future_results]
    return results


def full_simulation_simple(N_sim):
    # multiprocessing magic
    pool = mp.Pool(processes=N_processes)
    future_results = [pool.apply_async(batch_simulation_simple, (seed, )) for seed in trange(N_sim)]
    results = [f.get() for f in future_results]
    return results


def full_simulation_simple_joblib(N_sim):
    # multiprocessing magic
    results = Parallel(n_jobs=N_processes)(delayed(batch_simulation_simple)(seed) for seed in trange(N_sim))
    return results


def full_simulation_set_joblib(N_sim):
    # multiprocessing magic
    global B
    results = Parallel(n_jobs=N_processes)(delayed(batch_simulation_simple)(seed) for seed in trange(N_sim))
    for i, result in enumerate(results):
        B[i] = result


def full_simulation_inpu():
    # multiprocessing magic
    pool = mp.Pool(processes=N_processes)
    future_results = [pool.apply_async(batch_simulation_inpu, (1, )) for _ in range(N_processes)]
    results = [f.get() for f in future_results]
    return results


def single_sim(seed):
    np.random.seed(seed)
    return np.random.weibull(a=100, size=(1000, 1000))


def full_sim():
    pool = mp.Pool(processes=N_processes)
    r = list(tqdm(pool.imap(single_sim, range(10)), total=10))
    return r



# if __name__ == '__main__':
#    with Pool(2) as p:
#       r = list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))


if __name__ == '__main__':
    pbar = 0

    tic = time()
    # for i in range(100):
    #     batch_simulation_set(seed=i)
    # results = batch_simulation_simple(seed=1000)
    # results = full_simulation_test(10)
    # results = full_simulation(1, 1000)
    # results = full_simulation_simple(10000)
    results = full_simulation_simple_joblib(10000)
    # full_simulation_set_joblib(1000)
    # N = 1
    # results = full_simulation_test_simple(int(1000 * N))
    # results = full_simulation_inpu()
    toc = time()
    print((toc - tic))
    # print(B)