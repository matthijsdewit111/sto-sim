# Names and UvAIDs:
# Matthijs de Wit, 10628258
# Menno Bruin, 11675225
#
# This file contains the code to produce the plots for assignment 2

import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simpy
from tqdm import tqdm


class Server:
    def __init__(self, env, n, shortest_job_first=False):
        self.env = env
        self.resources = simpy.Resource(env, capacity=n) if not shortest_job_first else simpy.PriorityResource(env, capacity=n)

    def execute_job(self, service_time):
        yield self.env.timeout(service_time)


def job_generator(env, server, arrival_rate, service_rate_distribution='M', shortest_job_first=False):
    for i in range(NUM_REQUESTS):
        j = job(env, i, server, service_rate_distribution, shortest_job_first)
        env.process(j)

        # wait for next arrival event
        arrival_delta = random.expovariate(arrival_rate)
        yield env.timeout(arrival_delta)


def job(env, id, server, srd, shortest_job_first):
    arrival_time = env.now
    # print(f"{id} arrives at {env.now:.1f}s")

    if srd == 'M':
        t = random.expovariate(1 / SERVICE_TIME)  # mean = ST
    elif srd == 'D':
        t = SERVICE_TIME
    elif srd == 'E':
        d = random.random()
        if d < 0.75:  # 0.75 * 0.5ST + 0.25 * 2.5ST = 1ST (combined mean = ST)
            t = random.expovariate(2/SERVICE_TIME)  # mean = 0.5 * ST
        else:
            t = random.expovariate(2/(5 * SERVICE_TIME))  # mean = 2.5 * ST
    else:
        print('invalid service rate distribution')

    if shortest_job_first:
        request = server.resources.request(priority=t)
    else:
        request = server.resources.request()

    with request as r:
        yield r
        start_time = env.now
        # print(f"{id} starts running at {env.now:.1f}s")

        yield env.process(server.execute_job(service_time=t))
        finish_time = env.now
        # print(f"{id} finished at {env.now:.1f}s")

    waiting_time = start_time - arrival_time
    waiting_times.append(waiting_time)


NUM_REQUESTS = 1000
ARRIVAL_RATE = 1  # mean job arrival interval, once every [x] seconds
SERVICE_TIME = 0.9  # mean job service time, comleted after [x] seconds
repetitions = 1000

# Part 2 and 3 of assignment
sjf = False
print(f"sjf: {sjf}")

for n in [1, 2, 4]:
    print(f"\r------------ n = {n} -------------")
    average_waiting_times = []
    arrival_rate = ARRIVAL_RATE * n  # scale with number of servers, to keep system load (rho) equal

    for i in tqdm(range(repetitions)):
        waiting_times = []

        env = simpy.Environment()
        server = Server(env, n, shortest_job_first=sjf)
        env.process(job_generator(env, server, arrival_rate, shortest_job_first=sjf))
        env.run()

        # print(waiting_times)

        average_waiting_times.append(np.mean(waiting_times))

    sns.distplot(np.array(average_waiting_times), label=f'n={n}')
    mean = np.mean(average_waiting_times)
    var = np.var(average_waiting_times)
    print(f"Mean waiting time: {mean:.2f}")
    print(f"Variance waiting times: {var:.2f}")
    rec_N = 4 * (1.96**2 * var) / (0.01 * mean ** 2)
    print(f"recommended N: {rec_N}")

plt.xlabel('Average Waiting Time')
plt.ylabel('# Measurements')
plt.legend()
plt.xlim(0, 20)
plt.savefig(f"M-M-{'SJF' if sjf else 'FIFO'}.png")
# plt.show()
plt.cla()


sjf = True
print(f"sjf: {sjf}")

for n in [1, 2, 4]:
    print(f"\r------------ n = {n} -------------")
    average_waiting_times = []
    arrival_rate = ARRIVAL_RATE * n  # scale with number of servers, to keep system load (rho) equal

    for i in tqdm(range(repetitions)):
        waiting_times = []

        env = simpy.Environment()
        server = Server(env, n, shortest_job_first=sjf)
        env.process(job_generator(env, server, arrival_rate, shortest_job_first=sjf))
        env.run()

        # print(waiting_times)

        average_waiting_times.append(np.mean(waiting_times))

    sns.distplot(np.array(average_waiting_times), label=f'n={n}')
    mean = np.mean(average_waiting_times)
    var = np.var(average_waiting_times)
    print(f"Mean waiting time: {mean:.2f}")
    print(f"Variance waiting times: {var:.2f}")
    rec_N = 4 * (1.96**2 * var) / (0.01 * mean ** 2)
    print(f"recommended N: {rec_N}")

plt.xlabel('Average Waiting Time')
plt.ylabel('# Measurements')
plt.legend()
plt.xlim(0, 8)
plt.savefig(f"M-M-{'SJF' if sjf else 'FIFO'}.png")
# plt.show()
plt.cla()

# Part 4 of assignment
srd = 'D'
print(f"srd: {srd}")

for n in [1, 2, 4]:
    print(f"\r------------ n = {n} -------------")
    arrival_rate = ARRIVAL_RATE * n  # scale with number of servers, to keep system load (rho) equal
    average_waiting_times = []

    for i in tqdm(range(repetitions)):
        waiting_times = []

        env = simpy.Environment()
        server = Server(env, n)
        env.process(job_generator(env, server, arrival_rate, service_rate_distribution=srd))
        env.run()

        average_waiting_times.append(np.mean(waiting_times))

    sns.distplot(np.array(average_waiting_times), label=f'n={n}')
    mean = np.mean(average_waiting_times)
    var = np.var(average_waiting_times)
    print(f"Mean waiting time: {mean:.2f}")
    print(f"Variance waiting times: {var:.2f}")
    rec_N = 4 * (1.96**2 * var) / (0.01 * mean ** 2)
    print(f"recommended N: {rec_N}")

plt.xlabel('Average Waiting Time')
plt.ylabel('# Measurements')
plt.legend()
plt.xlim(0, 10)
plt.savefig(f"M-{srd}-FIFO.png")
# plt.show()
plt.cla()


srd = 'E'
print(f"srd: {srd}")

for n in [1, 2, 4]:
    print(f"\r------------ n = {n} -------------")
    arrival_rate = ARRIVAL_RATE * n  # scale with number of servers, to keep system load (rho) equal
    average_waiting_times = []

    for i in tqdm(range(repetitions)):
        waiting_times = []

        env = simpy.Environment()
        server = Server(env, n)
        env.process(job_generator(env, server, arrival_rate, service_rate_distribution=srd))
        env.run()

        average_waiting_times.append(np.mean(waiting_times))

    sns.distplot(np.array(average_waiting_times), label=f'n={n}')
    mean = np.mean(average_waiting_times)
    var = np.var(average_waiting_times)
    print(f"Mean waiting time: {mean:.2f}")
    print(f"Variance waiting times: {var:.2f}")
    rec_N = 4 * (1.96**2 * var) / (0.01 * mean ** 2)
    print(f"recommended N: {rec_N}")

plt.xlabel('Average Waiting Time')
plt.ylabel('# Measurements')
plt.legend()
plt.xlim(0, 40)
plt.savefig(f"M-{srd}-FIFO.png")
# plt.show()
plt.cla()
