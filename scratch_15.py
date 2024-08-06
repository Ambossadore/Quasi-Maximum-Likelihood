import numpy as np
from time import time
from functools import partial
import multiprocessing as mp

N_processes = 60


def batch_simulation(N_batch):
    sims = []
    for i in range(N_batch):
        a = np.random.weibull(a=100, size=(1000, 1000))
        sims.append(a)
    return np.stack(sims)


def full_simulation(N_batch):
    # define variable for results
    sim_results = []

    # multiprocessing magic
    pool = mp.Pool(processes=N_processes)
    simulation_routine = partial(batch_simulation, N_batch=N_batch)
    future_results = [pool.apply_async(simulation_routine) for _ in range(N_processes)]
    results = [f.get() for f in future_results]
    # for i in range(N_processes):
    #     sim_results = np.append(sim_results, results[i])

    return results


if __name__ == '__main__':
    tic = time()
    # batch_simulation(10)
    results = full_simulation(10)
    toc = time()
    print(toc - tic)
    print(len(results))
    print(results[0].shape)