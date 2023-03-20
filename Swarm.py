import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import time
import random
from NeuroEvolution import NeuroEvolution

class Topolgy:
    def __init__(self, n_particles, sleep=0, lifetime=np.inf, social=0):
        self.adjacency_matrix = np.ones(shape=(n_particles, n_particles))
        self.sleep = sleep
        self.lifetime = lifetime
        self.social = social

    def return_global_best(self, swarm, particle_idx, social=False):
        if (social):
            global_best = copy.deepcopy(
                swarm.position[np.argmax(swarm.pbest_cost[np.where(swarm.topology.adjacency_matrix[0] == 1)])])
        else:
            neigh = pd.DataFrame(columns=['index', 'distance'])

            for i in range(swarm.n_particles):
                if (i != particle_idx):
                    if (swarm.awake[i]):
                        # if (True):
                        neigh.loc[len(neigh)] = [int(i),
                                                 self.hamming_distance(swarm.position[particle_idx], swarm.position[i])]
                    else:
                        neigh.loc[len(neigh)] = [int(i), -np.inf]
                        # print(i, 'sleeping')

            best_neigh_cost = -np.inf
            best_idx = None
            for idx in neigh.sort_values(by=['distance']).iloc[:int(swarm.k[particle_idx])]['index']:
                if (best_neigh_cost < swarm.pbest_cost[int(idx)]):
                    best_neigh_cost = swarm.pbest_cost[int(idx)]
                    best_idx = int(idx)
            global_best = swarm.pbest_pos[best_idx]
            # print(best_idx)
            # print(global_best)
            # print(particle_idx)
            # print()
            # print(neigh.sort_values(by=['distance']).iloc[:int(swarm.k[particle_idx])]['index'])
        return global_best

    def draw_adjacency_matrix(self, node_order=None, partitions=[], colors=[]):
        # Plot adjacency matrix in toned-down black and white
        fig = plt.figure(figsize=(5, 5))  # in inches
        plt.imshow(self.adjacency_matrix,
                      cmap="Greys",
                      interpolation="none")

        # The rest is just if you have sorted nodes by a partition and want to
        # highlight the module boundaries
        assert len(partitions) == len(colors)
        ax = plt.gca()
        for partition, color in zip(partitions, colors):
            current_idx = 0
            for module in partition:
                ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                               len(module),  # Width
                                               len(module),  # Height
                                               facecolor="none",
                                               edgecolor=color,
                                               linewidth="1"))
                current_idx += len(module)
        plt.show()

    def hamming_distance(self, ind1, ind2):
        ind1 = np.array(ind1)
        ind2 = np.array(ind2)
        return (np.sum(np.abs(ind1 - ind2)))


class Star(Topolgy):
    def __init__(self, n_particles, sleep=0, lifetime=np.inf, social=0):
        self.adjacency_matrix = np.ones(shape=(n_particles, n_particles))
        self.sleep = np.zeros(shape=(n_particles))
        self.sleep[:] = sleep
        self.lifetime = lifetime
        self.social = social


class Swarm:
    def __init__(self, topology, n_particles, dimensions, options, bounds, velocity_clamp, bounded=False):
        self.log = pd.DataFrame(columns=['ind', 'time', 'fitness'])
        self.topology = topology
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.options = options
        self.bounds = bounds
        self.velocity_clamp = velocity_clamp
        self.bounded = bounded

        self.creation_time = time.time()
        self.position = np.zeros(shape=(self.n_particles, self.dimensions))
        #population = np.random.rand(self.n_particles, self.dimensions)
        population = np.empty((self.n_particles, self.dimensions))
        #print(population)
        for i in range(self.dimensions):
            #print(population[:, i])

            #print(np.random.uniform(low=bounds[i][0], high=bounds[i][1], size=(self.n_particles)))
            population[:, i] = np.random.uniform(low=bounds[i][0], high=bounds[i][1], size=(self.n_particles))
        population = population.astype(float)
        self.position = population
        self.velocity = np.zeros(shape=(self.n_particles, self.dimensions))
        self.velocity_clamp = np.zeros(shape=(self.dimensions, 2))
        self.awake = np.ones(shape=(self.n_particles))
        self.alive = np.ones(shape=(self.n_particles))
        self.alive[:] = self.topology.lifetime
        self.pbest_pos = np.zeros(shape=(self.n_particles, self.dimensions))
        self.best_pos = np.zeros(shape=(self.dimensions))
        self.pbest_cost = np.zeros(shape=(self.n_particles))
        self.k = np.zeros(shape=(self.n_particles))
        # self.k[:] = self.options['k']
        self.best_cost = -np.inf
        self.current_cost = -np.inf


        for i in range(dimensions):
            range_size = np.abs(bounds[i][1] - bounds[i][0])
            self.velocity_clamp[i][0] = - range_size * velocity_clamp
            self.velocity_clamp[i][1] = range_size * velocity_clamp

            for j in range(n_particles):

                # low = random.uniform(bounds[0], bounds[1])
                # high = random.uniform(low, bounds[1])
                # self.position[i] = np.random.uniform(low=low, high=bounds[1], size=(dimensions,))
                # population = Evolution.create_population(population_size, ind_size)

                self.velocity[j][i] = np.random.uniform(low=self.velocity_clamp[i][0], high=self.velocity_clamp[i][1])

                self.pbest_pos[j] = self.position[j]
                #self.k[i] = np.sum(self.topology.adjacency_matrix[i])

    def evaluate(self, position):
        delay = 0
        time.sleep(position[1] / 10)
        # print(position)
        # print(position - 0.5)
        # print(-np.sum(np.abs(position - 0.5)))
        # print()
        return -np.sum(np.abs(position - 0.5))

    def optimize(self, model, steps=np.inf, no_change=10, verbose=0):
        computation_df = pd.DataFrame(columns=['selected', 'time'])

        for i in range(self.n_particles):
            self.pbest_cost[i] = -np.inf
            # fitness = self.evaluate(self.position[i])
            ind = self.position[i]
            # print(len(self.log))
            # print(self.log['ind'].unique())
            if (len(self.log) > 1 and str(ind) in self.log['ind'].unique()):
                fitness = self.log[self.log['ind'] == str(ind)]['fitness'].iloc[0]
            else:
                # print(ind.shape)
                start = time.time()
                fitness = NeuroEvolution.evaluate(ind, [model])[0]

                computation_df.loc[len(computation_df)] = [np.sum(ind), time.time() - start]

                row = [str(ind), time.time() - self.creation_time, fitness]
                self.log.loc[len(self.log)] = row

                if (self.pbest_cost[i] is None or self.pbest_cost[i] < fitness):
                    self.pbest_cost[i] = copy.deepcopy(fitness)
                    self.pbest_pos[i] = copy.deepcopy(self.position[i])

                if (self.best_cost is None or self.best_cost < fitness):
                    self.best_cost = copy.deepcopy(fitness)
                    self.best_pos = copy.deepcopy(self.position[i])
                    time_found = time.time() - self.creation_time
                    # print('new best', self.best_cost)
                    # print('new best', self.best_pos)


        counter = 0
        no_change_counter = 0
        time_found = None
        while (counter != steps and no_change_counter < no_change):
            no_change_counter += 1
            awake_average = 0
            asleep_average = 0
            awake_count = 0
            asleep_count = 0
            for i in range(self.n_particles):
                if (random.uniform(0, 1) > self.topology.sleep[i]):
                    self.awake[i] = 1
                    ind = self.position[i]
                    # print('awake', counter, np.sum(ind))
                    awake_count += 1
                    awake_average += np.sum(ind)
                else:
                    self.awake[i] = 0
                    # print('asleep', counter, np.sum(ind))
                    asleep_count += 1
                    asleep_average += np.sum(ind)

                if (self.alive[i] and self.awake[i]):
                    #global_best = self.topology.return_global_best(self, i, self.topology.social)
                    global_best = copy.deepcopy(
                    self.pbest_pos[np.argmax(self.pbest_cost[np.where(self.topology.adjacency_matrix[i] == 1)])])
                    #print(global_best)
                    for j in range(self.dimensions):
                        r1 = random.uniform(0, 1)
                        r2 = random.uniform(0, 1)

                        #print(counter, i, np.where(self.topology.adjacency_matrix[i] == 1))
                        #if (global_best[j] != 0 and self.position[i][j] == 0):
                        #    print('i', i)
                        #    print('j', j)
                            #print('velocity', self.velocity[i][j])
                            #print('r1, r1', r1, r2)
                       #     print('current', self.position[i][j] )
                       #     print('pbest', self.pbest_pos[i][j] )
                       #     print('global_best', global_best[j] )
                       #     print('1', self.options['w'] * self.velocity[i][j])
                       #     print('2', self.options['c1'] * r1 * (
                       #                 self.pbest_pos[i][j] - self.position[i][j]))
                       #     print('3', self.options['c2'] * r2 * (
                       #                                       global_best[j] - self.position[i][j]))
                       #     print()

                        self.velocity[i][j] = self.options['w'] * self.velocity[i][j] + self.options['c1'] * r1 * (
                                    self.pbest_pos[i][j] - self.position[i][j]) + self.options['c2'] * r2 * (
                                                          global_best[j] - self.position[i][j])

                        #print('velocity after', self.velocity[i][j])


                        if (self.velocity[i][j] < self.velocity_clamp[j][0]):
                            self.velocity[i][j] = self.velocity_clamp[j][0]
                        if (self.velocity[i][j] > self.velocity_clamp[j][1]):
                            self.velocity[i][j] = self.velocity_clamp[j][1]



                        #print('spped', i, j, self.velocity[i][j])

                        #print('pbest_pos', i, j, self.pbest_pos[i][j])

                        #print('global_best', j, global_best[j])

                        #print()



                        self.position[i][j] = self.position[i][j] + self.velocity[i][j]

                        if (self.bounded and self.position[i][j] < self.bounds[j][0]):
                            self.position[i][j] = self.bounds[j][0]
                        if (self.bounded and self.position[i][j] > self.bounds[j][1]):
                            self.position[i][j] = self.bounds[j][1]

                        #if (global_best[j] != 0 and self.position[i][j] == 0):

                        #    print('current after', self.position[i][j])
                        #    print()

                    # fitness = self.evaluate(self.position[i])
                    self.alive[i] -= 1
                    ind = self.position[i]
                    if (len(self.log) > 1 and str(ind) in self.log['ind'].unique()):
                        fitness = self.log[self.log['ind'] == str(ind)]['fitness'].iloc[0]
                    else:
                        # print(ind.shape)
                        fitness = NeuroEvolution.evaluate(ind, [model])[0]
                        row = [str(ind), time.time() - self.creation_time, fitness]
                        self.log.loc[len(self.log)] = row

                        # fitness = Evolution.evaluate(ind, task, target_dataset, baseline_individual)[0]

                        if (self.pbest_cost[i] < fitness):
                            self.pbest_cost[i] = copy.deepcopy(fitness)
                            self.pbest_pos[i] = copy.deepcopy(self.position[i])
                            self.alive[i] = self.topology.lifetime
                            # print('personal', self.best_cost)
                            # print('personal', self.best_pos)
                            # print(counter)

                        if (self.best_cost is None or self.best_cost < fitness):
                            self.best_cost = copy.deepcopy(fitness)
                            self.best_pos = copy.deepcopy(self.position[i])
                            no_change_counter = 0
                            # print('new best', self.best_cost)
                            # print('new best', self.best_pos)
                            # print(counter)
                            time_found = time.time() - self.creation_time

                # print('sleep', self.awake[i], self.alive[i])
                if (self.alive[i] == 0):
                    print('dead', i)

            counter += 1

            if (verbose):
                print("Best Particle = ", np.round(self.best_cost, 4), ", Step = ", counter)
            # print(self.best_cost, self.best_pos)
        return self.best_cost, self.best_pos, time_found


