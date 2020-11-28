import simpy
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Server:
    def __init__(self, env, n, shortest_job_first=False):
        self.env = env
        self.resources = simpy.Resource(env, capacity=n) if not shortest_job_first else simpy.PriorityResource(env, capacity=n)

    def execute_job(self, service_time):
        yield self.env.timeout(service_time)


def job_generator(env, server, arrival_rate, shortest_job_first=False):
    for i in range(NUM_REQUESTS):
        j = job(env, i, server, shortest_job_first)
        env.process(j)

        yield env.timeout(arrival_rate)


def job(env, id, server, shortest_job_first):
    # print(f"{id} arrives at {env.now:.1f}s")

    t = random.expovariate(1 / SERVICE_TIME)

    if shortest_job_first:
        request = server.resources.request(priority=t)
    else:
        request = server.resources.request()

    with request as r:
        yield r
        # print(f"{id} starts running at {env.now:.1f}s")

        yield env.process(server.execute_job(service_time=t))
        # print(f"{id} finished at {env.now:.1f}s")


NUM_REQUESTS = 100
ARRIVAL_RATE = 5  # new request every [x] seconds
SERVICE_TIME = 5
MAX_SERVICE_TIME = 200
SJF = True  # shortest job first

average_waiting_times = []

for n in [1, 2, 4]:
    arrival_rate = ARRIVAL_RATE / n
    print(f"\r------------ n = {n} -------------", end='')
    for i in tqdm(range(500)):
        env = simpy.Environment()
        server = Server(env, n, shortest_job_first=SJF)
        env.process(job_generator(env, server, arrival_rate, shortest_job_first=SJF))
        env.run()

        average_waiting_time = env.now / NUM_REQUESTS - arrival_rate
        average_waiting_times.append(average_waiting_time)

    sns.distplot(np.array(average_waiting_times), label=f'n={n}')
    average_waiting_times = []

plt.xlabel('Average Waiting Time')
plt.ylabel('# Measurements')
plt.legend()
plt.ylim(0, 5)
plt.xlim(0, 5/2)
plt.show()
