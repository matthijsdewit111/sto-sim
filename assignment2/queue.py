import simpy
import random


class Server:

    def __init__(self, env, shortest_first=False):
        self.env = env
        self.store = simpy.Store(env) if not shortest_first else simpy.PriorityStore(env)

    def execute_request(self, value):
        t = random.expovariate(1.0 / ARRIVAL_RATE)
        yield self.env.timeout(t)
        self.store.put(value)

    def put(self, value):
        self.env.process(self.execute_request(value))

    def get(self):
        return self.store.get()


def transmitter(env, server):
    for i in range(NUM_REQUESTS):
        yield env.timeout(5)
        server.put(env.now)


def receiver(env, server):
    for i in range(NUM_REQUESTS):
        transmit_time = yield server.get()
        print(f'Request handling time: {env.now - transmit_time:.2f}s')


env = simpy.Environment()

NUM_REQUESTS = 10
ARRIVAL_RATE = 5  # new request every [x] seconds

for n in [1, 2, 4]:
    print(f"------------ n = {n} -------------")
    server = Server(env)
    env.process(transmitter(env, server))
    env.process(receiver(env, server))
    env.run()
