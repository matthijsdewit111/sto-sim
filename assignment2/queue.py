import simpy
import random


class Server:
    def __init__(self, env, n):
        self.env = env
        self.resources = simpy.Resource(env, capacity=n)
        # self.resources = simpy.PriorityResource(env, capacity=n)

    def execute_job(self, id, service_time):
        yield self.env.timeout(service_time)


def job_generator(env, server):
    for i in range(NUM_REQUESTS):
        j = job(env, i, server)
        env.process(j)

        # wait for next arrival event
        yield env.timeout(ARRIVAL_RATE)


def job(env, id, server):
    print(f"{id} arrives at {env.now}")

    # request allocation of resource for job
    # with server.resources.request(priority=10-id) as request:
    with server.resources.request() as request:
        yield request
        print(f"{id} starts running at {env.now}")

        # run job
        yield env.process(server.execute_job(id, 10-id))
        print(f"{id} finished at {env.now}")


NUM_REQUESTS = 10
ARRIVAL_RATE = 5  # new request every [x] seconds
SERVICE_TIME = 6  # [x] second to complete job

for n in [1]:
    print(f"------------ n = {n} -------------")
    env = simpy.Environment()
    server = Server(env, n)
    env.process(job_generator(env, server))
    env.run()
