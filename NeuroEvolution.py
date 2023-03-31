import numpy as np
import pandas as pd
import deap
from deap import tools
from deap import base, creator
import time
from random import randrange
import copy
import random
import Activation
import NeuralNetwork
import Representation
import tensorflow as tf
import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
creator.create("FitnessMin", base.Fitness, weights=(1.0,1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)


class NeuroEvolution:

    @staticmethod
    def isMontonic(f):
        valueRange = np.linspace(-1, 1, 10)
        values = []
        for i in valueRange:
            values.append(f.s2(float(i)))
        return np.all(np.diff(values) >= 0)

    @staticmethod
    def treeCrossover(ind1, ind2):
        kmap = {2: [1, 3, 5, 6], 3: [2, 4, 7, 8], 4: [3, 5, 6], 5: [4, 7, 8]}
        crossPoint = random.choice([2, 3, 4, 5])
        changedIndexes = kmap[crossPoint]

        oldInd1 = copy.copy(ind1)

        for idx in changedIndexes:
            ind1[idx] = ind2[idx]
            ind2[idx] = oldInd1[idx]

        return ind1, ind2

    @staticmethod
    def HUX(ind1, ind2, fixed=True):
        # index variable
        idx = 0

        # Result list
        res = []

        # With iteration
        for i in ind1:
            if i != ind2[idx]:
                res.append(idx)
            idx = idx + 1
        if (len(res) > 1):
            numberOfSwapped = randrange(1, len(res))
            if (fixed):
                numberOfSwapped = len(res) // 2
            indx = random.sample(res, numberOfSwapped)

            oldInd1 = copy.copy(ind1)

            for i in indx:
                ind1[i] = ind2[i]

            numberOfSwapped = randrange(1, len(res))
            if (fixed):
                numberOfSwapped = len(res) // 2
            indx = random.sample(res, numberOfSwapped)

            for i in indx:
                ind2[i] = oldInd1[i]
        return ind1, ind2

    @staticmethod
    def nan_in_ind(individual):
        return np.isnan(NeuroEvolution.behavior(individual)).any()

    @staticmethod
    def inf_in_ind(individual):
        for num in NeuroEvolution.behavior(individual):
            if (math.isinf(num)):
                return True
        return False

    @staticmethod
    def constant_ind(individual):
        return len(set(NeuroEvolution.behavior(individual))) == 1

    @staticmethod
    def evaluate(individual, models, reps=1, test_time=None, verbose = 1):
        fitness = 0
        training_time = 0
        inference_time = 0
        epoch_time = 0


        #print(individual)
        for model in models:
            for _ in range(reps):
                #model1.model.save_weights('before_evolution.h5')
                if (isinstance(model, NeuralNetwork.FC)):
                    evaluation_model = type(model).__call__(model.dataset, model.layers, model.neurons)
                elif (isinstance(model, NeuralNetwork.VGG)):
                    evaluation_model = type(model).__call__(model.dataset,  model.blocks)

                evaluation_model.set_config(int(np.round(individual[0])), int(np.ceil(individual[1])),
                                            individual[2], individual[3], individual[4], individual[5], individual[6],
                                            individual[7], individual[8], individual[9], individual[10], individual[11],
                                            individual[12], individual[13], int(np.ceil(individual[14])),
                                            individual[15], individual[16], int(np.round(individual[17])),
                                            individual[18], individual[19], individual[20], individual[21],
                                            model.metric, model.epochs, model.iterations, model.patience, verbose=model.verbose,
                                            batch_range=model.batch_range, lr_range=model.lr_range, sleep=model.sleep, save_best=model.save_best, cut_threshold=model.cut_threshold)
                if (verbose):
                    print()
                    print('loss_noise:', evaluation_model.loss_noise, ', activation_noise:',
                          evaluation_model.activation_noise, ', input_noise:',
                          evaluation_model.input_noise, ', label_smoothing:', evaluation_model.label_smoothing,
                          ', weight_noise:', evaluation_model.weight_noise, ', gradient_dropout:',
                          evaluation_model.gradient_dropout,
                          ', gradient_noise:', evaluation_model.gradient_noise, ', batch_size:',
                          evaluation_model.batch_size, ', dropout:', evaluation_model.dropout, ', drop_connect:', evaluation_model.drop_connect,
                    ', batch_schedule:', evaluation_model.batch_schedule, ', drnn:', evaluation_model.drnn, ', weight_std:', evaluation_model.weight_std,
                          ', flip:', evaluation_model.random_flip, ', rotation:', evaluation_model.random_rotation, ', zoom:', evaluation_model.random_zoom,
                          ', translation:', evaluation_model.random_translation, ', contrast:', evaluation_model.random_contrast,
                          ', shuffle:', evaluation_model.shuffle, ', lr:', evaluation_model.lr, ', optimizer:', evaluation_model.optimizer,
                          ', lr_schedule:', evaluation_model.lr_schedule, ', batch_increase:', evaluation_model.batch_increase,
                          ', lr_increase:', evaluation_model.lr_increase)


                evaluation_model.create_model()
                #evaluation_model.max_batch = model.max_batch

                start = time.time()
                evaluation_model.fit()
                training_time += evaluation_model.training_time
                inference_time += evaluation_model.inference_time
                epoch_time += evaluation_model.epoch_time

                if (test_time):
                    fitness += evaluation_model.test_score
                    if (verbose):
                        print(evaluation_model.test_score)

                else:
                    fitness += evaluation_model.val_score



                if (verbose):

                    print(evaluation_model.val_score, evaluation_model.test_score)
                #print()


                    if (pd.isna(evaluation_model.val_loss)):
                        return -1 * np.inf,

                del evaluation_model



        return fitness/reps,

    @staticmethod
    def createPopulationMaxout(populationSize, k, include=False):
        pop = []
        np.random.seed()
        random.seed()

        for i in range(populationSize):
            ind = []
            for _ in range(k):
                ind.append([np.round(random.uniform(-1, 1), 2)])
            ind = np.array(ind)
            ind = ind.flatten()
            pop.append(deap.creator.Individual(ind))
        return list(pop)

    @staticmethod
    def createPopulation(populationSize, include=False):
        pop = []
        np.random.seed()
        random.seed()
        relu = deap.creator.Individual([5, 7, 9, 1, 1, 7, 7, 9, 7])
        if (include):
            include = deap.creator.Individual(include)
        for i in range(populationSize):
            pop.append(deap.creator.Individual(
                [random.randint(1, 6), random.randint(7, 31), random.randint(7, 31), random.randint(1, 6),
                 random.randint(1, 6),
                 random.randint(7, 31), random.randint(7, 31), random.randint(7, 31), random.randint(7, 31)]))
        if (include):
            del pop[-1]
            pop.append(include)
        return list(pop)

    @staticmethod
    def mutateOperation(ind, numberOfFlipped=3, pruning=0):
        kmap = {2: [1, 3, 5, 6], 3: [2, 4, 7, 8], 4: [3, 5, 6], 5: [4, 7, 8]}
        crossPoint = random.choice([2, 3, 4, 5])
        changedIndexes = kmap[crossPoint]
        #print('before', ind)
        if random.random() < pruning:
            # print('pruning')
            for idx in changedIndexes:
                if (idx in [3, 4]):
                    ind[idx] = 1
                else:
                    ind[idx] = 7
        else:
            #print('mutate')
            for _ in range(numberOfFlipped):
                flipedOperation = random.randint(0, len(ind) - 1)
                if (flipedOperation in [0, 3, 4]):
                    newOperation = random.randint(1, 6)
                    while (newOperation == ind[flipedOperation]):
                        newOperation = random.randint(1, 6)
                    ind[flipedOperation] = newOperation
                else:
                    newOperation = random.randint(7, 31)
                    while (newOperation == ind[flipedOperation]):
                        newOperation = random.randint(7, 31)
                    ind[flipedOperation] = newOperation
        #print('after', ind)
        #print()
        return ind

    @staticmethod
    def mutateMaxoutOperation(ind, numberOfFlipped=1, pruning=0.1):
        for _ in range(numberOfFlipped):
            flipedBit = random.randint(0, len(ind) - 1)
            ind[flipedBit] = np.round(random.uniform(-1, 1), 2)
        return ind

    @staticmethod
    def createToolbox(model, representation='pangaea', alg='CHC', epochs=10, optimizationTask='step', metric='accuracy'):
        toolbox = base.Toolbox()
        if (alg == 'CHC'):
            toolbox.register("mate", NeuroEvolution.HUX)
        elif (alg == 'GA'):
            toolbox.register("mate", NeuroEvolution.treeCrossover)
        if (representation == 'maxout'):
            toolbox.register("mutate", NeuroEvolution.mutateMaxoutOperation)
        else:
            toolbox.register("mutate", NeuroEvolution.mutateOperation)
        toolbox.register("select", tools.selTournament, tournsize=1)
        toolbox.register("evaluate", NeuroEvolution.evaluate, models=model, representation=representation, epochs=epochs, optimizationTask=optimizationTask, metric=metric)
        return toolbox

    @staticmethod
    def behavior(ind, x_min=-1.0, x_max=1.0, num_points=11):
        range = np.linspace(x_min, x_max, num_points)
        values = []
        for i in range:
            values.append(Representation.Function(ind).calculate(float(i)).numpy())
        return values

    @staticmethod
    def compare_func(ind1, ind2, x_min=-1.0, x_max=1.0, num_points=10):
        #range = np.linspace(x_min, x_max, num_points)
        behavior1 = NeuroEvolution.behavior(ind1, x_min, x_max, num_points)
        behavior2 = NeuroEvolution.behavior(ind2, x_min, x_max, num_points)

        if (behavior1 == behavior2):
            return True

        #if (behavior1[0] > behavior2[0]):
        #    const = behavior1[0] - behavior2[1]
        #    change_values = []
        #    for i in range:
        #        change_values.append(Representation.Function(ind1).s2(float(i)).numpy() - const)
        #else:
        #    const = behavior2[0] - behavior1[0]
        #    change_values = []
        #    for i in range:
        #        change_values.append(Representation.Function(ind1).s2(float(i)).numpy() + const)

        #if (behavior2 == change_values):
        #    return True

        return False

    @staticmethod
    def CHC(model, population=False, populationSize=40, population_history=None, d=4,
            divergence=3, epochs=10, representation='pangaea',
            maxGenerations=np.inf, maxNochange=np.inf, timeout=np.inf, optimizationTask='step',
             metric='accuracy',
            stop=np.inf, verbose=0, include=False):
        start = time.time()
        end = time.time()
        toolbox = NeuroEvolution.createToolbox(model, representation, 'CHC', epochs,
                                               optimizationTask, metric)

        generationCounter = 0
        evaulationCounter = 0
        best = -1 * np.inf
        noChange = 0

        # if (not d):
        #    d = d0
        logDF = pd.DataFrame(
            columns=(
                'generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations', 'best_solution', 'd'))

        if (not population):
            population = NeuroEvolution.createPopulation(populationSize, include)

        # for ind in population:
        #    print(ind)
        #    print(Representation.Maxout(np.array(ind)).s2(1))

        # calculate fitness tuple for each individual in the population:
        # fitnessValues = list(map(toolbox.evaluate, population))
        evaluatedIndividuals = [ind for ind in population if ind.fitness.valid]
        bestInd = toolbox.clone(population[0])
        updated = False
        for individual in evaluatedIndividuals:
            if (best < individual.fitness.values[0]):
                noChange = 0
                best = individual.fitness.values[0]
                bestInd = toolbox.clone(individual)
                bestTime = time.time()
                updated = True

            # print(time.time()-start)
        if (time.time() - start) > timeout:
            if (updated):
                print('log1', bestInd)
                row = [generationCounter, (bestTime - start), best, -1, evaulationCounter,
                       bestInd, d]
                logDF.loc[len(logDF)] = row
                updated = False
            return logDF, population

        freshIndividuals = [ind for ind in population if not ind.fitness.valid]
        for individual in freshIndividuals:
            # print(earlyTermination)
            individual.fitness.values = toolbox.evaluate(individual)
            evaulationCounter += 1
            if (best < individual.fitness.values[0]):
                noChange = 0
                best = individual.fitness.values[0]
                bestTime = time.time()
                bestInd = toolbox.clone(individual)
                updated = True
            # print(time.time()-start)
            if (time.time() - start) > timeout:
                if (updated):
                    row = [generationCounter, (bestTime - start), best, -1, evaulationCounter,
                           bestInd, d]
                    logDF.loc[len(logDF)] = row
                    updated = False
                return logDF, population

        # extract fitness values from all individuals in population:
        fitnessValues = [individual.fitness.values[0] for individual in population]
        # initialize statistics accumulators:
        maxFitnessValues = []
        meanFitnessValues = []

        d0 = len(population[0]) // 2
        if (not population_history):
            population_history = []
            for ind in population:
                population_history.append(ind)


        # main evolutionary loop:
        # stop if max fitness value reached the known max value
        # OR if number of generations exceeded the preset value:
        while best < stop and generationCounter < maxGenerations and noChange < maxNochange and (
                end - start) < timeout:
            # update counter:
            generationCounter = generationCounter + 1

            population[:] = tools.selNSGA2(population, populationSize)

            for ind in population:
                print(ind, ind.fitness.values)
            print()

            # apply the selection operator, to select the next generation's individuals:
            # offspring = toolbox.select(population, len(population))
            # clone the selected individuals:
            offspring = list(map(toolbox.clone, population))
            random.shuffle(offspring)

            newOffspring = []

            newOffspringCounter = 0

            # apply the crossover operator to pairs of offspring:
            numberOfPaired = 0
            numberOfMutation = 0
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if NeuroEvolution.hammingDistance(child1, child2) > d and d > 0:
                    # print('Before')
                    # print(child1)
                    # print(child2)
                    toolbox.mate(child1, child2)
                    numberOfPaired += 1
                    newOffspringCounter += 2
                    addChild = True
                    for ind in population_history:
                        if (NeuroEvolution.hammingDistance(ind, child1) == 0):
                            newOffspringCounter -= 1
                            addChild = False
                            break
                    if (addChild):
                        #population_history.append(child1)
                        newOffspring.append(child1)
                    addChild = True
                    for ind in population_history:
                        if (NeuroEvolution.hammingDistance(ind, child2) == 0):
                            newOffspringCounter -= 1
                            addChild = False
                            break
                    if (addChild):
                        #population_history.append(child2)
                        newOffspring.append(child2)
                    # print('history length', len(population_history))
                    # print('After')
                    # print(child1)
                    # print(child2)
                    # print()
                    del child1.fitness.values
                    del child2.fitness.values
            # print('this is d', d)
            if (d == 0):
                d = d0
                newOffspring = []
                bestIndividual = tools.selNSGA2(population, 1)[0]
                while (numberOfMutation < len(population)):
                    mutant = toolbox.clone(bestIndividual)
                    toolbox.mutate(mutant, divergence)
                    found = False
                    for ind2 in population_history:
                        if (NeuroEvolution.compare_func(ind2, mutant)):
                            found = True
                    if (not found):
                    #population_history.append(mutant)
                        newOffspring.append(mutant)
                        numberOfMutation += 1
                    del mutant.fitness.values

            # if (newOffspringCounter == 0 and d > 0):
            #    d -= 1
            noChange += 1
            # calculate fitness for the individuals with no previous calculated fitness value:
            freshIndividuals = [ind for ind in newOffspring if not ind.fitness.valid]
            # freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
            for individual in freshIndividuals:
                #print('fresh', individual)
                found = False
                for ind2 in population_history:
                    if (NeuroEvolution.compare_func(ind2, individual)):
                        found = True
                        ind3 = copy.copy(ind2)

                if (not found):
                    individual.fitness.values = toolbox.evaluate(individual)
                    population_history.append(individual)

                else:
                    individual.fitness.values = -1,
                    #print(ind3)
                    #print(NeuroEvolution.behavior(ind3))
                    #print((individual))

                    #print(NeuroEvolution.behavior(individual))
                    print('duplicate')

                evaulationCounter += 1
                if (best < individual.fitness.values[0]):
                    noChange = 0
                    best = individual.fitness.values[0]
                    bestTime = time.time()
                    bestInd = toolbox.clone(individual)
                    updated = True
                # print(time.time()-start)
                if (time.time() - start) > timeout:
                    if (updated):
                        row = [generationCounter, (bestTime - start), best, -1, evaulationCounter,
                               bestInd, d]
                        logDF.loc[len(logDF)] = row
                    # row = [generationCounter, (end - start), np.round(best, 4), -1, evaulationCounter,
                    #       individual, d]
                    # logDF.loc[len(logDF)] = row
                    return logDF, population

            # evaulationCounter = evaulationCounter + len(freshIndividuals)

            if (numberOfMutation == 0):
                oldPopulation = copy.copy(population)
                population[:] = tools.selNSGA2(population + newOffspring, populationSize)
                differentPopulation = False
                for index in range(0, len(population)):
                    if (NeuroEvolution.hammingDistance(oldPopulation[index], population[index]) != 0):
                        differentPopulation = True
                print(differentPopulation)
                if (not differentPopulation):
                    d -= 1
            else:
                bestIndividual = tools.selNSGA2(population, 1)
                population[:] = tools.selNSGA2(bestIndividual + newOffspring, populationSize)

            # collect fitnessValues into a list, update statistics and print:
            fitnessValues = [ind.fitness.values[0] for ind in population]

            maxFitness = max(fitnessValues)
            # if (best >= maxFitness):
            #    noChange += 1
            meanFitness = sum(fitnessValues) / len(population)
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)

            end = time.time()

            # find and print best individual:
            best_index = fitnessValues.index(max(fitnessValues))
            if (verbose):
                print("Best Individual = ", np.round(maxFitness, 2), ", Gen = ", generationCounter, '\r', end='')
            # print()
            print(np.round(maxFitness, 2), 'number of paired:', numberOfPaired, 'number of mutations:',
                  numberOfMutation, ' d:', d, ' no change:', noChange)
            # print('new', newOffspringCounter)
            print()
            end = time.time()
            if (updated):
                row = [generationCounter, (bestTime - start), best, -1, evaulationCounter,
                       bestInd, d]
                logDF.loc[len(logDF)] = row
                updated = False

            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            #
            # xdata = []
            # for ind in population:
            #     xdata.append(ind.fitness.values[0])
            #
            # ydata = []
            # for ind in population:
            #     ydata.append(ind.fitness.values[1])
            #
            # zdata = []
            # for ind in population:
            #     zdata.append(ind.fitness.values[2])
            #
            # ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Reds');
            #
            # plt.show()

        end = time.time()
        return logDF, population_history, population
    @staticmethod
    def GA(model, earyTermination, population=False, representation='maxout', populationSize=40, crossOverP=0, mutationP=1, zeroP=0.5, maxGenerations=np.inf,
           maxNochange=np.inf, epochs=10, optimizationTask='activation', evaluation='validation',
           timeout=np.inf, stop=np.inf, verbose=0, include=None, metric='loss'):

        start = time.time()
        end = time.time()

        indSize = 9
        toolbox = NeuroEvolution.createToolbox(model, earyTermination, representation, metric, 'GA', epochs, optimizationTask, evaluation)
        if (not population):
            population = NeuroEvolution.createPopulation(populationSize, include)

        generationCounter = 0
        # calculate fitness tuple for each individual in the population:
        fitnessValues = list(map(toolbox.evaluate, population))
        for individual, fitnessValue in zip(population, fitnessValues):
            individual.fitness.values = fitnessValue

        # extract fitness values from all individuals in population:
        fitnessValues = [individual.fitness.values[0] for individual in population]

        # initialize statistics accumulators:
        maxFitnessValues = []
        meanFitnessValues = []

        best = -1 * np.inf
        noChange = 0
        evaulationCounter = populationSize

        populationHistory = []
        for ind in population:
            populationHistory.append(ind)

        logDF = pd.DataFrame(
            columns=('generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations', 'best_solution'))

        # main evolutionary loop:
        # stop if max fitness value reached the known max value
        # OR if number of generations exceeded the preset value:
        while best < stop and generationCounter < maxGenerations and noChange < maxNochange and (
                end - start) < timeout:
            generationCounter = generationCounter + 1

            for ind in population:
                print(ind, ind.fitness.values)
            print()

            # apply the selection operator, to select the next generation's individuals:
            offspring = toolbox.select(population, len(population))
            # clone the selected individuals:
            offspring = list(map(toolbox.clone, offspring))

            # apply the crossover operator to pairs of offspring:

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossOverP:
                    toolbox.mate(child1, child2)
                    populationHistory.append(child1)
                    populationHistory.append(child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutationP:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # calculate fitness for the individuals with no previous calculated fitness value:
            freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
            freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
            for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
                individual.fitness.values = fitnessValue

            evaulationCounter = evaulationCounter + len(freshIndividuals)

            population[:] = tools.selBest(population + offspring, populationSize)

            # collect fitnessValues into a list, update statistics and print:
            fitnessValues = [ind.fitness.values[0] for ind in population]

            maxFitness = max(fitnessValues)
            if (best >= maxFitness):
                noChange += 1
            if (best < maxFitness):
                best = maxFitness
                noChange = 0
            meanFitness = sum(fitnessValues) / len(population)
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)

            end = time.time()

            # find and print best individual:
            best_index = fitnessValues.index(max(fitnessValues))
            if (verbose):
                print("Best Individual = ", np.round(maxFitness, 2), ", Gen = ", generationCounter, '\r', end='')
            # print()
            # print(np.round(100*maxFitness, 2), 'number of paired:', numberOfPaired, 'number of mutations:', numberOfMutation, ' d:', d)
            # print()
            end = time.time()
            row = [generationCounter, (end - start), maxFitness, meanFitness, evaulationCounter,
                   population[best_index]]
            logDF.loc[len(logDF)] = row

        end = time.time()
        return logDF, population

    @staticmethod
    def hammingDistance(ind1, ind2):
        ind1 = np.array(ind1)
        ind2 = np.array(ind2)
        return (len(ind1) - (np.sum(np.equal(ind1, ind2))))

    @staticmethod
    def SAGA(model, representation='maxout', k=5, populationSize=40, reductionRate=0.5, pateince=2, step=10, d=False, divergence=3, epochs=10,
             targetFitness=1,
             verbose=0, qualOnly=False, timeout=np.inf, include=False,
             noChange=np.inf, evaluation='validation'):

        start = time.time()
        logDF = pd.DataFrame(columns=('generation', 'time', 'best_fitness',
                                      'best_solution', 'surrogate_level', 'epochs'))
        partialModel = copy.copy(model)
        sampleSize = 100
        # partialModel.setTrainingSample(sampleSize)
        v_epochs = 2
        task = 'feature_selection'
        if (representation == 'maxout'):
            population = NeuroEvolution.createPopulationMaxout(populationSize, k, include)
        else:
            population = NeuroEvolution.createPopulation(populationSize, include)

        bestTrueFitnessValue = 0 # np.inf
        sagaActivationFunction = [1] * 9
        qual = False

        numberOfEvaluations = 0
        generationCounter = 0
        maxAllowedSize = int(partialModel.x_train.shape[0])

        d = len(population[0]) // 2
        surrogateLevel = 0

        pateince0 = pateince

        while (bestTrueFitnessValue < targetFitness and pateince > 0 and v_epochs <= epochs): #sampleSize < maxAllowedSize):
            # print('patience:', pateince)
            if (verbose):
                print('Current epochs:', v_epochs)
                # print('Current Approx Sample Size:', sampleSize)
                print('Current Population Size:', populationSize)
            pateince -= 1
            log, population = NeuroEvolution.CHC(model,
                                                 population,
                                                 d=d,
                                                 divergence=divergence,
                                                 epochs=v_epochs,
                                                 populationSize=populationSize,
                                                 maxNochange=step,
                                                 verbose=verbose)
            generationCounter = generationCounter + int(log.iloc[-1]['generation'])
            activationFunctionIndividual = log.iloc[-1]['best_solution']

            approxBestInGeneration = np.round(
                NeuroEvolution.evaluate(activationFunctionIndividual, model, v_epochs), 2)[0]
            end = time.time()

            # Check if the original value improved
            if (sagaActivationFunction != activationFunctionIndividual):
                pateince = pateince0
                bestTrueFitnessValue = approxBestInGeneration
                sagaActivationFunction = activationFunctionIndividual
                sagaIndividual = tools.selBest(population, 1)
                row = [generationCounter, (end - start), bestTrueFitnessValue,
                       sagaActivationFunction, surrogateLevel, v_epochs]
                logDF.loc[len(logDF)] = row
                if (verbose):
                    print('The best individual is saved', bestTrueFitnessValue)
                    print(row)

            v_epochs = v_epochs * 2
            populationSize = int(populationSize * reductionRate)
            surrogateLevel += 1
            d = indSize // 2
            # partialModel.setTrainingSample(sampleSize)
            newInd = NeuroEvolution.createPopulation(populationSize, indSize)

            population[:] = tools.selBest(sagaIndividual + newInd, populationSize)

        return logDF, population

    @staticmethod
    def SAGA_V0(model, schedule='instance', representation='maxout', k=5, populationSize=40, reductionRate=0.5, pateince=0, tolerance=0, maxNoChange=np.inf, d=3, divergence=3,
             targetFitness=1, initilizationMax=2, optimizationTask='step', epochs='10', sampleSize='512',
             verbose=0, qualOnly=False, timeout=np.inf, include=False,
             noChange=np.inf, evaluation='validation', metric='loss', starting_epoch=1):

        start = time.time()
        logDF = pd.DataFrame(columns=('generation', 'time', 'best_fitness',
                                   'best_solution', 'surrogate_level', 'epochs'))
        epoch_level = 1
        earlyTermination = pd.DataFrame(columns=('epochs', 'min_fitness'))
        while epoch_level < epochs:
                row = [epoch_level, -1 * np.inf]
                earlyTermination.loc[len(earlyTermination)] = row
                epoch_level = epoch_level * 2
        row = [epochs, -1 * np.inf]
        earlyTermination.loc[len(earlyTermination)] = row
        #print(earlyTermination)

        maxAllowedSize = int(model[0].dataset.X_train.shape[0])
        if (schedule == 'instance'):
            sampleSize = model[0].dataset.X_train.shape[0] // 4
            v_epochs = epochs
        else:
            sampleSize = model[0].dataset.X_train.shape[0]
            v_epochs = int(epochs/32)+1
            v_epochs = starting_epoch
            #maxAllowedSize = np.inf



        #v_epochs = 1
        task = 'feature_selection'
        if (representation == 'maxout'):
            population = NeuroEvolution.createPopulationMaxout(populationSize, k, include)
        else:
            population = NeuroEvolution.createPopulation(populationSize, include)

        for individual in population:
            individual.fitness.values = NeuroEvolution.evaluate(individual, models=model, earlyTermination=earlyTermination, representation=representation, epochs=v_epochs, sampleSize=sampleSize, optimizationTask=optimizationTask, evaluation=evaluation, metric=metric)


        #sagaIndividual = deap.creator.Individual(
        #        [random.randint(1, 6), random.randint(7, 30), random.randint(7, 30), random.randint(1, 6),
        #         random.randint(1, 6),
        #         random.randint(7, 30), random.randint(7, 30), random.randint(7, 30), random.randint(7, 30)])
        sagaIndividual = tools.selBest(population, 1)[0]
        bestTrueFitnessValue = -1 * np.inf
        #sagaActivationFunction = [1] * 9
        currentFunction = [1] * 9
        qual = False
        #callbacks = [EarlyStopping(monitor='val_cindex', patience=100, restore_best_weights=True, mode='max')]

        if (include):
            bestTrueFitnessValue = NeuroEvolution.evaluate(include, models=model, earlyTermination=earlyTermination, representation=representation, epochs=epochs, sampleSize=maxAllowedSize, optimizationTask=optimizationTask, evaluation=evaluation, metric=metric)[0]
            sagaIndividual = deap.creator.Individual(include)
            currentFunction = include
            sagaIndividual.fitness.values = bestTrueFitnessValue,
            print('Initial individual fitness is:', bestTrueFitnessValue)
            #print(sagaIndividual, bestTrueFitnessValue)
            #print(type(sagaIndividual), sagaIndividual, sagaIndividual.fitness.values)

        numberOfEvaluations = 0
        generationCounter = 0

        #d = indSize // 2
        surrogateLevel = 0

        pateinceCounter = np.inf
        improvedInLevel = False
        maxNoChangeCounter = 0
        tolernaceCounter = tolerance

        initilizationCounter = 0

        minPopulationSize = 4

        if (verbose):
            print('Current surrogate epochs:', v_epochs)
            print('Current surrogate sample size:', sampleSize)
            print('Current population size:', populationSize)
        end = time.time()
        while (bestTrueFitnessValue < targetFitness and pateinceCounter >= 0 and v_epochs <= epochs and (
                end - start) < timeout and  sampleSize <= maxAllowedSize):
            print('patience:', pateinceCounter, ' improved:', improvedInLevel, ' maxnochange:', maxNoChangeCounter, ' initilizationCounter:', initilizationCounter,  'tolerance:', tolernaceCounter)
            #print('saga ind out:', sagaIndividual)
            #print(earlyTermination)
            log, population = NeuroEvolution.CHC(model,
                                                 earlyTermination,
                                                 representation,
                                                 k,
                                                 population,
                                                 d=d,
                                                 divergence=divergence,
                                                 epochs=v_epochs,
                                                 sampleSize=sampleSize,
                                                 populationSize=populationSize,
                                                 maxGenerations=1,
                                                 optimizationTask=optimizationTask,
                                                 evaluation=evaluation,
                                                 metric=metric,
                                                 timeout=np.inf,
                                                 verbose=0)
            generationCounter = generationCounter + int(log.iloc[-1]['generation'])
            activationFunctionIndividual = log.iloc[-1]['best_solution']
            d = log.iloc[-1]['d']

            worstInd = tools.selWorst(population, 1)[0]
            #if (earlyTermination[earlyTermination['epochs'] == v_epochs].iloc[0][
            #    'min_fitness'] < worstInd.fitness.values):
            #    earlyTermination.loc[earlyTermination['epochs'] >= v_epochs, 'min_fitness'] = worstInd.fitness.values
            #    print(earlyTermination)
            #    print()
            #print('this is SAGA d', d)


            end = time.time()
            print(end-start)
            # Check if the original value improved
            if (currentFunction != activationFunctionIndividual):
                currentFunction = activationFunctionIndividual
                trueBestInGeneration = NeuroEvolution.evaluate(activationFunctionIndividual, models=model, earlyTermination=earlyTermination, representation=representation, epochs=epochs, sampleSize=maxAllowedSize, optimizationTask=optimizationTask, evaluation=evaluation, metric=metric)[0]
                if (trueBestInGeneration > bestTrueFitnessValue):
                    improvedInLevel = True
                    maxNoChangeCounter = 0
                    initilizationCounter = 0
                    bestTrueFitnessValue = trueBestInGeneration
                    pateinceCounter = pateince
                    sagaIndividual = activationFunctionIndividual
                    #print('saga ind:', sagaIndividual)
                    if (verbose):
                        end = time.time()
                        row = [generationCounter, (end - start), bestTrueFitnessValue,
                               sagaIndividual, surrogateLevel, v_epochs]
                        logDF.loc[len(logDF)] = row
                        print('The best individual is saved', bestTrueFitnessValue)
                        print(row)
                #elif (trueBestInGeneration < log.iloc[-1]['best_fitness']):
                #    tolernaceCounter -= 1
                #    if (verbose):
                #        print('False Optimum:', trueBestInGeneration)
                #        print(currentFunction)
                elif (trueBestInGeneration <= bestTrueFitnessValue):
                    tolernaceCounter -= 1
                    if (verbose):
                        print('False Optimum:', trueBestInGeneration)
                        print(currentFunction)

            else:
                maxNoChangeCounter+=1

            if (d == 0):
                initilizationCounter += 1

            if (maxNoChangeCounter > maxNoChange or tolernaceCounter < 0 or initilizationCounter > initilizationMax):
                #print(type(sagaIndividual), sagaIndividual)
                if (not improvedInLevel):
                    pateinceCounter -= 1
                else:
                    pateinceCounter = pateince
                maxNoChangeCounter = 0
                initilizationCounter = 0
                tolernaceCounter = tolerance
                currentFunction = sagaIndividual

                if (schedule == 'instance'):
                    sampleSize = sampleSize * 2
                else:
                    v_epochs = v_epochs * 2
                if (bestTrueFitnessValue > targetFitness or pateinceCounter < 0  or v_epochs > epochs or (
                        time.time() - start) > timeout  or sampleSize > maxAllowedSize):
                    # print(bestTrueFitnessValue)
                    # print(pateinceCounter)
                    break

                if (improvedInLevel):
                    populationSize = int(populationSize * reductionRate)
                if (populationSize < minPopulationSize):
                    populationSize = minPopulationSize
                improvedInLevel = False
                surrogateLevel += 1
                d = len(population[0]) // 2
                cycle = 1
                # partialModel.setTrainingSample(sampleSize)
                if (representation == 'maxout'):
                    newInd = NeuroEvolution.createPopulationMaxout(populationSize, k)
                else:
                    newInd = NeuroEvolution.createPopulation(populationSize)

                #newInd = NeuroEvolution.createPopulation(populationSize, indSize)
                newInd.append(sagaIndividual)
                for individual in newInd:
                    individual.fitness.values = NeuroEvolution.evaluate(individual, models=model, earlyTermination=earlyTermination, representation=representation, epochs=v_epochs, sampleSize=sampleSize,
                                                                        optimizationTask=optimizationTask,
                                                                        evaluation=evaluation, metric=metric)
                #for ind in newInd:
                #    print(ind, ind.fitness.values)

                population[:] = tools.selBest(newInd, populationSize)

                if (verbose):
                    print('Current surrogate epochs:', v_epochs)
                    print('Current surrogate sample size:', sampleSize)
                    print('Current population size:', populationSize)
        end = time.time()

        print('Surrogate phase is over!')
        newInd = []
        numberOfMutation = 0
        toolbox = NeuroEvolution.createToolbox(model, earlyTermination, representation, metric, 'CHC', epochs, sampleSize, optimizationTask, evaluation)
        while (numberOfMutation < populationSize-1):
            mutant = toolbox.clone(sagaIndividual)
            numberOfMutation += 1
            s = toolbox.mutate(mutant)
            newInd.append(mutant)
        for individual in newInd:
            individual.fitness.values = NeuroEvolution.evaluate(individual, models=model,
                                                                earlyTermination=earlyTermination,
                                                                representation=representation, epochs=epochs, sampleSize=sampleSize,
                                                                optimizationTask=optimizationTask,
                                                                evaluation=evaluation, metric=metric)
            if((time.time() - start) > timeout):
                return logDF, population

        sagaIndividual.fitness.values = bestTrueFitnessValue,
        newInd.append(sagaIndividual)
        qualTime = time.time() - start
        log, population = NeuroEvolution.CHC(model,
                                             earlyTermination,
                                             representation,
                                             k,
                                             newInd,
                                             d=d,
                                             divergence=divergence,
                                             epochs=epochs,
                                             sampleSize=maxAllowedSize,
                                             populationSize=populationSize,
                                             maxNochange=noChange,
                                             optimizationTask=optimizationTask,
                                             evaluation=evaluation,
                                             metric=metric,
                                             timeout=timeout-qualTime,
                                             verbose=0)
        updated = False
        if (len(log)> 0):
            for index, row in log.iterrows():
                if (row['best_fitness'] > logDF.iloc[-1]['best_fitness']):
                    log.at[index, 'time'] = log.loc[index, 'time'] + qualTime
                    log.at[index, 'number_of_evaluations'] = log.loc[index, 'number_of_evaluations'] + numberOfEvaluations
                    log.at[index, 'generation'] = log.loc[index, 'generation'] + logDF.iloc[-1]['generation']
                    log['best_fitness_original'] = log['best_fitness']
                    updated = True
                else:
                    log.drop(index, inplace=True)
        if (updated):
                logDF = pd.concat((logDF, log))
        return logDF, population


    @staticmethod
    def SAGA_V1(model, populationSize=40, reductionRate=0.5, pateince=0, maxNoChange=np.inf, d=False, divergence=3, epochs=10,
             targetFitness=1, initilizationMax=2, optimizationTask='step',
             verbose=0, qualOnly=False, timeout=np.inf, include=False,
             maxGen=10, noChange=np.inf, evaluation='validation'):

        start = time.time()
        logDF = pd.DataFrame(columns=('generation', 'time', 'best_fitness',
                                      'best_solution', 'surrogate_level', 'epochs'))

        epoch_level = 1
        earlyTermination = pd.DataFrame(columns=('epochs', 'min_fitness'))
        while epoch_level < epochs:
            row = [epoch_level, -1 * np.inf]
            earlyTermination.loc[len(earlyTermination)] = row
            epoch_level = epoch_level * 2

        minPopulationSize = 4
        partialModel = copy.copy(model)
        sampleSize = 100
        # partialModel.setTrainingSample(sampleSize)
        v_epochs = 1
        task = 'feature_selection'
        indSize = 9
        population = NeuroEvolution.createPopulation(populationSize, include)
        for individual in population:
            individual.fitness.values = NeuroEvolution.evaluate(individual, models=model, earlyTermination=earlyTermination, representation='tree', epochs=v_epochs, optimizationTask=optimizationTask, evaluation=evaluation, metric='loss')

        worstInd = tools.selWorst(population, 1)[0]
        earlyTermination.loc[earlyTermination['epochs'] == v_epochs, 'min_fitness'] = worstInd.fitness.values

        bestTrueFitnessValue = -1 * np.inf
        trueFitnessValue = -1 * np.inf

        sagaActivationFunction = [1] * 9
        currentFunction = [1] * 9
        qual = False
        #callbacks = [EarlyStopping(monitor='val_cindex', patience=100, restore_best_weights=True, mode='max')]

        if (include):
            bestTrueFitnessValue = np.round(
                NeuroEvolution.evaluate(include, models=model, earlyTermination=earlyTermination, epochs=epochs, optimizationTask=optimizationTask, evaluation=evaluation), 4)[0]
            sagaIndividual = deap.creator.Individual(include)
            currentFunction = include
            sagaIndividual.fitness.values = bestTrueFitnessValue,
            print('Initial individual fitness is:', bestTrueFitnessValue)
            #print(sagaIndividual, bestTrueFitnessValue)
            #print(type(sagaIndividual), sagaIndividual, sagaIndividual.fitness.values)

        numberOfEvaluations = 0
        generationCounter = 0
        #maxAllowedSize = int(partialModel.X_train.shape[0])

        d = indSize // 2
        surrogateLevel = 0

        pateinceCounter = pateince
        improvedInLevel = True
        maxNoChangeCounter = 0

        initilizationCounter = 0

        if (verbose):
            print('Current epochs:', v_epochs)
            # print('Current Approx Sample Size:', sampleSize)
            print('Current Population Size:', populationSize)
        end = time.time()

        while (bestTrueFitnessValue < targetFitness and pateinceCounter >= 0 and v_epochs < epochs and (
                end - start) < timeout):  # sampleSize < maxAllowedSize):
            gen = 0
            while True:
                #print('before', population)
                log, population = NeuroEvolution.CHC(model,
                                                     earlyTermination,
                                                     'tree',
                                                     5,
                                                     population,
                                                     d=d,
                                                     divergence=divergence,
                                                     epochs=v_epochs,
                                                     populationSize=populationSize,
                                                     maxNochange=2,
                                                     optimizationTask=optimizationTask,
                                                     evaluation=evaluation,
                                                     metric='loss',
                                                     timeout=np.inf,
                                                     verbose=0)
                generationCounter = generationCounter + int(log.iloc[-1]['generation'])
                activationFunctionIndividual = log.iloc[-1]['best_solution']
                trueFitnessValue = log.iloc[-1]['best_fitness']
                #print(population)



                d = log.iloc[-1]['d']
                if d == 0:
                    d = 1
            #print('this is SAGA d', d)
                #print(earlyTermination[earlyTermination['epochs'] == v_epochs].iloc[0][
                #    'min_fitness'])
                #print(tools.selWorst(population, 1)[0])

                worstInd = tools.selWorst(population, 1)[0]
                if (earlyTermination[earlyTermination['epochs'] == v_epochs].iloc[0][
                    'min_fitness'] < worstInd.fitness.values):
                    earlyTermination.loc[earlyTermination['epochs'] >= v_epochs, 'min_fitness'] = worstInd.fitness.values
                    print(earlyTermination)
                    print()
                    gen += 1


                    end = time.time()

                    # Check if the original value improved
                if (trueFitnessValue > bestTrueFitnessValue):
                        bestTrueFitnessValue = trueFitnessValue
                        sagaActivationFunction = activationFunctionIndividual
                        sagaIndividual = tools.selBest(population, 1)[0]
                        row = [generationCounter, (end - start), bestTrueFitnessValue,
                                   sagaActivationFunction, surrogateLevel, v_epochs]
                        logDF.loc[len(logDF)] = row
                        if (verbose):
                                print('The best individual is saved', trueFitnessValue)
                                print(row)
                    #else:
                    #    break

                v_epochs = v_epochs * 2
                populationSize = int(populationSize * reductionRate)
                if (populationSize < minPopulationSize):
                    populationSize = minPopulationSize
                if (bestTrueFitnessValue > targetFitness or pateinceCounter < 0 or v_epochs > epochs or (
                            time.time() - start) > timeout):  # sampleSize < maxAllowedSize):
                    break
                '''
                newInd = NeuroEvolution.createPopulation(populationSize)
                for individual in newInd:
                    individual.fitness.values = NeuroEvolution.evaluate(individual, models=model, earlyTermination=None,
                                                                        representation='tree', epochs=v_epochs,
                                                                        optimizationTask=optimizationTask,
                                                                        evaluation=evaluation, metric='loss')
                newInd[:] = tools.selBest(newInd, int(populationSize/2))

                    #print('new: ', individual, individual.fitness.values)
                    #print('new', individual.fitness.values)
                population[:] = tools.selBest(population, int(populationSize / 2))

                for individual in population:
                    individual.fitness.values = NeuroEvolution.evaluate(individual, models=model, earlyTermination=None,
                                                                        representation='tree', epochs=v_epochs,
                                                                        optimizationTask=optimizationTask,
                                                                        evaluation=evaluation, metric='loss')

                    #print('old: ', individual, individual.fitness.values)

                    #print('old', individual.fitness.values)

                #for ind in newInd:
                #    print(ind, ind.fitness.values)

                population[:] = tools.selBest(newInd+population, populationSize)
                #print('result: ', population)
                '''
                if (verbose):
                        print('Current epochs:', v_epochs)
                        # print('Current Approx Sample Size:', sampleSize)
                        print('Current Population Size:', populationSize)

                end = time.time()

        timePassed = time.time() -start
        log, population = NeuroEvolution.CHC(model,
                                             earlyTermination,
                                             'tree',
                                             5,
                                             None,
                                             d=d,
                                             divergence=divergence,
                                             epochs=epochs,
                                             populationSize=10,
                                             maxGenerations=np.inf,
                                             optimizationTask=optimizationTask,
                                             evaluation=evaluation,
                                             metric='loss',
                                             timeout=timeout-timePassed,
                                             verbose=0)
        updated = False
        if (len(log) > 0):
            for index, row in log.iterrows():
                if (row['best_fitness'] > logDF.iloc[-1]['best_fitness']):
                    log.at[index, 'time'] = log.loc[index, 'time'] + timePassed
                    log.at[index, 'number_of_evaluations'] = log.loc[
                                                                 index, 'number_of_evaluations'] + numberOfEvaluations
                    log.at[index, 'generation'] = log.loc[index, 'generation'] + logDF.iloc[-1]['generation']
                    log['best_fitness_original'] = log['best_fitness']
                    updated = True
                else:
                    log.drop(index, inplace=True)
        if (updated):
            logDF = pd.concat((logDF, log))

        return logDF, population

    def SAGA_V2(model, populationSize=40, reductionRate=0.5, pateince=0, tolerance=0, maxNoChange=np.inf, d=False, divergence=3, epochs=10,
             targetFitness=1, initilizationMax=2, optimizationTask='step',
             verbose=0, qualOnly=False, timeout=np.inf, include=False,
             noChange=np.inf, evaluation='validation'):

        start = time.time()
        logDF = pd.DataFrame(columns=('generation', 'time', 'best_fitness',
                                      'best_solution', 'surrogate_level', 'epochs'))
        partialModel = copy.copy(model)
        sampleSize = 100
        # partialModel.setTrainingSample(sampleSize)
        v_epochs = 1

        epoch_level = 1
        earlyTermination = pd.DataFrame(columns=('epochs', 'min_fitness'))
        while epoch_level<epochs:
            row = [epoch_level, -1 * np.inf]
            earlyTermination.loc[len(earlyTermination)] = row
            epoch_level = epoch_level * 2

        task = 'feature_selection'
        indSize = 9
        population = NeuroEvolution.createPopulation(populationSize, indSize, include)
        for individual in population:
            individual.fitness.values = NeuroEvolution.evaluate(individual, models=model, earlyTermination=earlyTermination, epochs=v_epochs, optimizationTask=optimizationTask, evaluation=evaluation)
        sagaIndividual = tools.selBest(population, 1)[0]
        worstInd = tools.selWorst(population, 1)[0]
        earlyTermination.loc[earlyTermination['epochs'] == v_epochs, 'min_fitness'] = worstInd.fitness.values

        bestTrueFitnessValue = -1 * np.inf
        sagaActivationFunction = [1] * 9
        currentFunction = [1] * 9
        qual = False
        #callbacks = [EarlyStopping(monitor='val_cindex', patience=100, restore_best_weights=True, mode='max')]

        if (include):
            bestTrueFitnessValue = np.round(
                NeuroEvolution.evaluate(include, models=model, epochs=epochs, optimizationTask=optimizationTask, evaluation=evaluation), 4)[0]
            sagaIndividual = deap.creator.Individual(include)
            currentFunction = include
            sagaIndividual.fitness.values = bestTrueFitnessValue,
            print('Initial individual fitness is:', bestTrueFitnessValue)
            #print(sagaIndividual, bestTrueFitnessValue)
            #print(type(sagaIndividual), sagaIndividual, sagaIndividual.fitness.values)

        numberOfEvaluations = 0
        generationCounter = 0
        #maxAllowedSize = int(partialModel.X_train.shape[0])

        d = indSize // 2
        surrogateLevel = 0

        pateinceCounter = pateince
        improvedInLevel = False
        maxNoChangeCounter = 0
        tolernaceCounter = tolerance

        initilizationCounter = 0

        if (verbose):
            print('Current epochs:', v_epochs)
            # print('Current Approx Sample Size:', sampleSize)
            print('Current Population Size:', populationSize)
        end = time.time()
        while (bestTrueFitnessValue < targetFitness and pateinceCounter >= 0 and v_epochs < epochs and (
                end - start) < timeout):  #sampleSize < maxAllowedSize):
            print('patience:', pateinceCounter, ' improved:', improvedInLevel, ' maxnochange:', maxNoChangeCounter, ' initilizationCounter:', initilizationCounter,  'tolerance:', tolernaceCounter)
            log, population = NeuroEvolution.CHC(model,
                                                 earlyTermination,
                                                 population,
                                                 d=d,
                                                 divergence=divergence,
                                                 epochs=v_epochs,
                                                 populationSize=populationSize,
                                                 maxGenerations=1,
                                                 optimizationTask=optimizationTask,
                                                 evaluation=evaluation,
                                                 verbose=0)
            generationCounter = generationCounter + int(log.iloc[-1]['generation'])
            activationFunctionIndividual = log.iloc[-1]['best_solution']
            d = log.iloc[-1]['d']
            #print('this is SAGA d', d)

            worstInd = tools.selWorst(population, 1)[0]
            if (earlyTermination[earlyTermination['epochs'] == v_epochs].iloc[0]['min_fitness'] < worstInd.fitness.values):
                earlyTermination.loc[earlyTermination['epochs'] == v_epochs, 'min_fitness'] = worstInd.fitness.values
            print(earlyTermination)
            print()

            end = time.time()

            # Check if the original value improved
            if (currentFunction != activationFunctionIndividual):
                currentFunction = activationFunctionIndividual
                trueBestInGeneration = NeuroEvolution.evaluate(activationFunctionIndividual, models=model, earlyTermination=earlyTermination, epochs=epochs, optimizationTask=optimizationTask, evaluation=evaluation, metric='loss')[0]
                if (trueBestInGeneration > bestTrueFitnessValue):
                    improvedInLevel = True
                    maxNoChangeCounter = 0
                    initilizationCounter = 0
                    bestTrueFitnessValue = trueBestInGeneration
                    pateinceCounter = pateince
                    sagaActivationFunction = activationFunctionIndividual
                    sagaIndividual = tools.selBest(population, 1)[0]

                    if (verbose):
                        end = time.time()
                        row = [generationCounter, (end - start), bestTrueFitnessValue,
                               sagaActivationFunction, surrogateLevel, v_epochs]
                        logDF.loc[len(logDF)] = row
                        print('The best individual is saved', bestTrueFitnessValue)
                        print(row)
                elif (trueBestInGeneration <= bestTrueFitnessValue):
                    tolernaceCounter -= 1
                    if (verbose):
                        print('False Optimum:', trueBestInGeneration)
                        print(currentFunction)

            else:
                maxNoChangeCounter+=1

            if (d == 0):
                initilizationCounter += 1

            if (maxNoChangeCounter > maxNoChange or tolernaceCounter < 0 or initilizationCounter > initilizationMax):
                #print(type(sagaIndividual), sagaIndividual)
                if (not improvedInLevel):
                    pateinceCounter -= 1
                else:
                    pateinceCounter = pateince
                maxNoChangeCounter = 0
                initilizationCounter = 0
                tolernaceCounter = tolerance
                currentFunction = sagaIndividual

                v_epochs = v_epochs * 2
                if (bestTrueFitnessValue > targetFitness or pateinceCounter < 0  or v_epochs > epochs or (
                        time.time() - start) > timeout):  # sampleSize < maxAllowedSize):
                    #print(bestTrueFitnessValue)
                    #print(pateinceCounter)
                    break

                improvedInLevel = False
                populationSize = int(populationSize * reductionRate)
                surrogateLevel += 1
                d = indSize // 2
                cycle = 1
                # partialModel.setTrainingSample(sampleSize)

                if (verbose):
                    print('Current epochs:', v_epochs)
                    # print('Current Approx Sample Size:', sampleSize)
                    print('Current Population Size:', populationSize)



        end = time.time()

        return logDF, population


    @staticmethod
    def SAGA_V3(model, populationSize=40, reductionRate=0.5, pateince=0, tolerance=0, maxNoChange=np.inf, d=False, divergence=3, epochs=10,
             targetFitness=1, initilizationMax=2, optimizationTask='step',
             verbose=0, qualOnly=False, timeout=np.inf, include=False,
             noChange=np.inf, evaluation='validation'):

        start = time.time()
        logDF = pd.DataFrame(columns=('generation', 'time', 'best_fitness',
                                      'best_solution', 'surrogate_level', 'epochs'))
        epoch_level = 1
        earlyTermination = pd.DataFrame(columns=('epochs', 'min_fitness'))
        while epoch_level<epochs:
            row = [epoch_level, -1 * np.inf]
            earlyTermination.loc[len(earlyTermination)] = row
            epoch_level = epoch_level * 2

        partialModel = copy.copy(model)
        sampleSize = 100
        # partialModel.setTrainingSample(sampleSize)
        v_epochs = 1
        task = 'feature_selection'
        indSize = 9
        population = NeuroEvolution.createPopulation(populationSize, indSize, include)
        for individual in population:
            individual.fitness.values = NeuroEvolution.evaluate(individual, models=model, earlyTermination=earlyTermination,epochs=v_epochs, optimizationTask=optimizationTask, evaluation=evaluation)

        sagaIndividual = tools.selBest(population, 1)[0]
        worstInd = tools.selWorst(population, 1)[0]
        earlyTermination.loc[earlyTermination['epochs'] == v_epochs, 'min_fitness'] = worstInd.fitness.values

        bestTrueFitnessValue = -1 * np.inf
        sagaActivationFunction = [1] * 9
        currentFunction = [1] * 9
        qual = False
        #callbacks = [EarlyStopping(monitor='val_cindex', patience=100, restore_best_weights=True, mode='max')]

        if (include):
            bestTrueFitnessValue = np.round(
                NeuroEvolution.evaluate(include, models=model, earlyTermination=earlyTermination, epochs=epochs, optimizationTask=optimizationTask, evaluation=evaluation), 4)[0]
            sagaIndividual = deap.creator.Individual(include)
            currentFunction = include
            sagaIndividual.fitness.values = bestTrueFitnessValue,
            print('Initial individual fitness is:', bestTrueFitnessValue)
            #print(sagaIndividual, bestTrueFitnessValue)
            #print(type(sagaIndividual), sagaIndividual, sagaIndividual.fitness.values)

        numberOfEvaluations = 0
        generationCounter = 0
        #maxAllowedSize = int(partialModel.X_train.shape[0])

        d = indSize // 2
        surrogateLevel = 0

        pateinceCounter = pateince
        improvedInLevel = False
        maxNoChangeCounter = 0
        tolernaceCounter = tolerance

        initilizationCounter = 0

        if (verbose):
            print('Current epochs:', v_epochs)
            # print('Current Approx Sample Size:', sampleSize)
            print('Current Population Size:', populationSize)
        end = time.time()
        while (bestTrueFitnessValue < targetFitness and pateinceCounter >= 0 and v_epochs < epochs and (
                end - start) < timeout):  #sampleSize < maxAllowedSize):
            print('patience:', pateinceCounter, ' improved:', improvedInLevel, ' maxnochange:', maxNoChangeCounter, ' initilizationCounter:', initilizationCounter,  'tolerance:', tolernaceCounter)
            log, population = NeuroEvolution.CHC(model,
                                                 earlyTermination,
                                                 population,
                                                 d=d,
                                                 divergence=divergence,
                                                 epochs=v_epochs,
                                                 populationSize=populationSize,
                                                 maxGenerations=1,
                                                 optimizationTask=optimizationTask,
                                                 evaluation=evaluation,
                                                 verbose=0)
            generationCounter = generationCounter + int(log.iloc[-1]['generation'])
            activationFunctionIndividual = log.iloc[-1]['best_solution']
            d = log.iloc[-1]['d']

            worstInd = tools.selWorst(population, 1)[0]
            earlyTermination.loc[earlyTermination['epochs'] == v_epochs, 'min_fitness'] = worstInd.fitness.values
            print(earlyTermination)
            print()
            #print('this is SAGA d', d)


            end = time.time()

            # Check if the original value improved
            if (currentFunction != activationFunctionIndividual):
                currentFunction = activationFunctionIndividual
                trueBestInGeneration = NeuroEvolution.evaluate(activationFunctionIndividual, models=model, earlyTermination=earlyTermination, epochs=epochs, optimizationTask=optimizationTask, evaluation=evaluation, metric='loss')[0]
                if (trueBestInGeneration > bestTrueFitnessValue):
                    improvedInLevel = True
                    maxNoChangeCounter = 0
                    initilizationCounter = 0
                    bestTrueFitnessValue = trueBestInGeneration
                    pateinceCounter = pateince
                    sagaActivationFunction = activationFunctionIndividual
                    sagaIndividual = tools.selBest(population, 1)[0]
                    if (verbose):
                        end = time.time()
                        row = [generationCounter, (end - start), bestTrueFitnessValue,
                               sagaActivationFunction, surrogateLevel, v_epochs]
                        logDF.loc[len(logDF)] = row
                        print('The best individual is saved', bestTrueFitnessValue)
                        print(row)
                elif (trueBestInGeneration <= bestTrueFitnessValue):
                    tolernaceCounter -= 1
                    if (verbose):
                        print('False Optimum:', trueBestInGeneration)
                        print(currentFunction)

            else:
                maxNoChangeCounter+=1

            if (d == 0):
                initilizationCounter += 1

            if (maxNoChangeCounter > maxNoChange or tolernaceCounter < 0 or initilizationCounter > initilizationMax):
                #print(type(sagaIndividual), sagaIndividual)
                if (not improvedInLevel):
                    pateinceCounter -= 1
                else:
                    pateinceCounter = pateince
                maxNoChangeCounter = 0
                initilizationCounter = 0
                tolernaceCounter = tolerance
                currentFunction = sagaIndividual



                v_epochs = v_epochs * 2
                if (bestTrueFitnessValue > targetFitness or pateinceCounter < 0  or v_epochs > epochs or (
                        time.time() - start) > timeout):  # sampleSize < maxAllowedSize):
                    # print(bestTrueFitnessValue)
                    # print(pateinceCounter)
                    break

                improvedInLevel = False
                populationSize = int(populationSize * reductionRate)
                surrogateLevel += 1
                d = indSize // 2
                cycle = 1
                # partialModel.setTrainingSample(sampleSize)
                newInd = NeuroEvolution.createPopulation(populationSize, indSize)
                newInd.append(sagaIndividual)
                for individual in newInd:
                    individual.fitness.values = NeuroEvolution.evaluate(individual, models=model, earlyTermination=earlyTermination, epochs=v_epochs,
                                                                        optimizationTask=optimizationTask,
                                                                        evaluation=evaluation)
                #for ind in newInd:
                #    print(ind, ind.fitness.values)

                population[:] = tools.selBest(newInd, populationSize)

                if (verbose):
                    print('Current epochs:', v_epochs)
                    # print('Current Approx Sample Size:', sampleSize)
                    print('Current Population Size:', populationSize)
        end = time.time()

        return logDF, population



    @staticmethod
    def SAGA_V4(model, earlyTermination=None, representation='maxout', k=5, populationSize=40, reductionRate=0.5, pateince=0, tolerance=0, maxNoChange=np.inf, d=3, divergence=3, epochs=10,
             targetFitness=1, initilizationMax=2, optimizationTask='step',
             verbose=0, qualOnly=False, timeout=np.inf, include=False,
             noChange=np.inf, evaluation='validation', metric='loss'):

        start = time.time()
        logDF = pd.DataFrame(columns=('generation', 'time', 'best_fitness',
                                   'best_solution', 'surrogate_level', 'epochs'))
        if (earlyTermination):
            epoch_level = 1
            earlyTermination = pd.DataFrame(columns=('epochs', 'min_fitness'))
            while epoch_level < epochs:
                row = [epoch_level, -1 * np.inf]
                earlyTermination.loc[len(earlyTermination)] = row
                epoch_level = epoch_level * 2

        if (representation == 'maxout'):
            population = NeuroEvolution.createPopulationMaxout(populationSize, k, include)
        else:
            population = NeuroEvolution.createPopulation(populationSize, include)

        for individual in population:
            individual.fitness.values = NeuroEvolution.evaluate(individual, models=model, earlyTermination=earlyTermination, representation=representation, epochs=v_epochs, optimizationTask=optimizationTask, evaluation=evaluation, metric=metric)


        if (include):
            bestTrueFitnessValue = np.round(
                NeuroEvolution.evaluate(include, models=model, earlyTermination=earlyTermination, representation=representation, epochs=epochs, optimizationTask=optimizationTask, evaluation=evaluation, metric=metric), 4)[0]
            sagaIndividual = deap.creator.Individual(include)
            currentFunction = include
            sagaIndividual.fitness.values = bestTrueFitnessValue,
            print('Initial individual fitness is:', bestTrueFitnessValue)

        numberOfEvaluations = 0
        generationCounter = 0
        maxNoChangeCounter = 0

        minPopulationSize = 4

        if (verbose):
            print('Current epochs:', v_epochs)
            # print('Current Approx Sample Size:', sampleSize)
            print('Current Population Size:', populationSize)
        end = time.time()
        while (bestTrueFitnessValue < targetFitness and pateinceCounter >= 0 and v_epochs < epochs and (
                end - start) < timeout):  #sampleSize < maxAllowedSize):
            print('patience:', pateinceCounter, ' improved:', improvedInLevel, ' maxnochange:', maxNoChangeCounter, ' initilizationCounter:', initilizationCounter,  'tolerance:', tolernaceCounter)
            #print('saga ind out:', sagaIndividual)
            log, population = NeuroEvolution.CHC(model,
                                                 earlyTermination,
                                                 representation,
                                                 k,
                                                 population,
                                                 d=d,
                                                 divergence=divergence,
                                                 epochs=v_epochs,
                                                 populationSize=populationSize,
                                                 maxGenerations=1,
                                                 optimizationTask=optimizationTask,
                                                 evaluation=evaluation,
                                                 metric=metric,
                                                 timeout=np.inf,
                                                 verbose=0)
            generationCounter = generationCounter + int(log.iloc[-1]['generation'])
            #print(log)
            activationFunctionIndividual = log.iloc[-1]['best_solution']
            d = log.iloc[-1]['d']

            worstInd = tools.selWorst(population, 1)[0]
            #if (earlyTermination[earlyTermination['epochs'] == v_epochs].iloc[0][
            #    'min_fitness'] < worstInd.fitness.values):
            #    earlyTermination.loc[earlyTermination['epochs'] >= v_epochs, 'min_fitness'] = worstInd.fitness.values
            #    print(earlyTermination)
            #    print()
            #print('this is SAGA d', d)


            end = time.time()
            print(end-start)

            # Check if the original value improved
            if (currentFunction != activationFunctionIndividual):
                currentFunction = activationFunctionIndividual
                #print('testing:', activationFunctionIndividual, log.iloc[-1]['best_fitness'])
                trueBestInGeneration = NeuroEvolution.evaluate(activationFunctionIndividual, models=model, earlyTermination=earlyTermination, representation=representation, epochs=epochs, optimizationTask=optimizationTask, evaluation=evaluation, metric=metric)[0]
                if (trueBestInGeneration > bestTrueFitnessValue and trueBestInGeneration > log.iloc[-1]['best_fitness']):
                    improvedInLevel = True
                    maxNoChangeCounter = 0
                    initilizationCounter = 0
                    bestTrueFitnessValue = trueBestInGeneration
                    pateinceCounter = pateince
                    sagaIndividual = activationFunctionIndividual
                    #print('saga ind:', sagaIndividual)
                    if (verbose):
                        end = time.time()
                        row = [generationCounter, (end - start), bestTrueFitnessValue,
                               sagaIndividual, surrogateLevel, v_epochs]
                        logDF.loc[len(logDF)] = row
                        print('The best individual is saved', bestTrueFitnessValue)
                        print(row)
                elif (trueBestInGeneration < log.iloc[-1]['best_fitness']):
                    tolernaceCounter -= 1
                    if (verbose):
                        print('False Optimum:', trueBestInGeneration)
                        print(currentFunction)
                elif (trueBestInGeneration <= bestTrueFitnessValue):
                    tolernaceCounter -= 1
                    if (verbose):
                        print('False Optimum:', trueBestInGeneration)
                        print(currentFunction)

            else:
                maxNoChangeCounter+=1

            if (d == 0):
                initilizationCounter += 1

            if (maxNoChangeCounter > maxNoChange or tolernaceCounter < 0 or initilizationCounter > initilizationMax):
                #print(type(sagaIndividual), sagaIndividual)
                if (not improvedInLevel):
                    pateinceCounter -= 1
                else:
                    pateinceCounter = pateince
                maxNoChangeCounter = 0
                initilizationCounter = 0
                tolernaceCounter = tolerance
                currentFunction = sagaIndividual

                v_epochs = v_epochs * 2
                if (bestTrueFitnessValue > targetFitness or pateinceCounter < 0  or v_epochs > epochs or (
                        time.time() - start) > timeout):  # sampleSize < maxAllowedSize):
                    # print(bestTrueFitnessValue)
                    # print(pateinceCounter)
                    break

                if (improvedInLevel):
                    populationSize = int(populationSize * reductionRate)
                if (populationSize < minPopulationSize):
                    populationSize = minPopulationSize
                improvedInLevel = False
                surrogateLevel += 1
                d = len(population[0]) // 2
                cycle = 1
                # partialModel.setTrainingSample(sampleSize)
                if (representation == 'maxout'):
                    newInd = NeuroEvolution.createPopulationMaxout(populationSize, k)
                else:
                    newInd = NeuroEvolution.createPopulation(populationSize)

                #newInd = NeuroEvolution.createPopulation(populationSize, indSize)
                newInd.append(sagaIndividual)
                for individual in newInd:
                    individual.fitness.values = NeuroEvolution.evaluate(individual, models=model, earlyTermination=earlyTermination, representation=representation, epochs=v_epochs,
                                                                        optimizationTask=optimizationTask,
                                                                        evaluation=evaluation, metric=metric)
                #for ind in newInd:
                #    print(ind, ind.fitness.values)

                population[:] = tools.selBest(newInd, populationSize)

                if (verbose):
                    print('Current epochs:', v_epochs)
                    # print('Current Approx Sample Size:', sampleSize)
                    print('Current Population Size:', populationSize)
        end = time.time()

        print('Surrogate phase is over!')
        newInd = []
        numberOfMutation = 0
        toolbox = NeuroEvolution.createToolbox(model, earlyTermination, representation, metric, 'CHC', epochs, optimizationTask, evaluation)
        while (numberOfMutation < minPopulationSize-1):
            mutant = toolbox.clone(sagaIndividual)
            numberOfMutation += 1
            s = toolbox.mutate(mutant)
            newInd.append(mutant)
        for individual in newInd:
            individual.fitness.values = NeuroEvolution.evaluate(individual, models=model,
                                                                earlyTermination=earlyTermination,
                                                                representation=representation, epochs=epochs,
                                                                optimizationTask=optimizationTask,
                                                                evaluation=evaluation, metric=metric)
            if((time.time() - start) > timeout):
                return logDF, population

        sagaIndividual.fitness.values = bestTrueFitnessValue,
        newInd.append(sagaIndividual)
        qualTime = time.time() - start
        log, population = NeuroEvolution.CHC(model,
                                             earlyTermination,
                                             representation,
                                             k,
                                             newInd,
                                             d=d,
                                             divergence=divergence,
                                             epochs=epochs,
                                             populationSize=minPopulationSize,
                                             maxNochange=np.inf,
                                             optimizationTask=optimizationTask,
                                             evaluation=evaluation,
                                             metric=metric,
                                             timeout=timeout-qualTime,
                                             verbose=0)
        updated = False
        if (len(log)> 0):
            for index, row in log.iterrows():
                if (row['best_fitness'] > logDF.iloc[-1]['best_fitness']):
                    log.at[index, 'time'] = log.loc[index, 'time'] + qualTime
                    log.at[index, 'number_of_evaluations'] = log.loc[index, 'number_of_evaluations'] + numberOfEvaluations
                    log.at[index, 'generation'] = log.loc[index, 'generation'] + logDF.iloc[-1]['generation']
                    log['best_fitness_original'] = log['best_fitness']
                    updated = True
                else:
                    log.drop(index, inplace=True)
        if (updated):
                logDF = pd.concat((logDF, log))
        return logDF, population

    @staticmethod
    def SAGA_V5(model, schedule='instance', representation='maxout', k=5, populationSize=40, reductionRate=0.5,
                pateince=0, tolerance=0, maxNoChange=np.inf, d=3, divergence=3,
                targetFitness=1, initilizationMax=2, optimizationTask='step', epochs='10', sampleSize='512',
                verbose=0, qualOnly=False, timeout=np.inf, include=False,
                noChange=np.inf, evaluation='validation', metric='loss'):

        start = time.time()
        logDF = pd.DataFrame(columns=('generation', 'time', 'best_fitness',
                                      'best_solution', 'surrogate_level', 'epochs'))
        epoch_level = 1
        earlyTermination = pd.DataFrame(columns=('epochs', 'min_fitness'))
        while epoch_level < epochs:
            row = [epoch_level, -1 * np.inf]
            earlyTermination.loc[len(earlyTermination)] = row
            epoch_level = epoch_level * 2
        row = [epochs, -1 * np.inf]
        earlyTermination.loc[len(earlyTermination)] = row
        # print(earlyTermination)

        maxAllowedSize = int(model[0].X_train.shape[0])
        if (schedule == 'instance'):
            sampleSize = 256
            v_epochs = epochs
        else:
            sampleSize = model[0].X_train.shape[0]
            v_epochs = int(epochs / 32) + 1
            v_epochs = 1
            # maxAllowedSize = np.inf

        # v_epochs = 1
        task = 'feature_selection'
        if (representation == 'maxout'):
            population = NeuroEvolution.createPopulationMaxout(populationSize, k, include)
        else:
            population = NeuroEvolution.createPopulation(populationSize, include)

        for individual in population:
            individual.fitness.values = NeuroEvolution.evaluate(individual, models=model,
                                                                earlyTermination=earlyTermination,
                                                                representation=representation, epochs=v_epochs,
                                                                sampleSize=sampleSize,
                                                                optimizationTask=optimizationTask,
                                                                evaluation=evaluation, metric=metric)

        # sagaIndividual = deap.creator.Individual(
        #        [random.randint(1, 6), random.randint(7, 30), random.randint(7, 30), random.randint(1, 6),
        #         random.randint(1, 6),
        #         random.randint(7, 30), random.randint(7, 30), random.randint(7, 30), random.randint(7, 30)])
        sagaIndividual = tools.selBest(population, 1)[0]
        bestTrueFitnessValue = -1 * np.inf
        # sagaActivationFunction = [1] * 9
        currentFunction = [1] * 9
        qual = False
        # callbacks = [EarlyStopping(monitor='val_cindex', patience=100, restore_best_weights=True, mode='max')]

        if (include):
            bestTrueFitnessValue = np.round(
                NeuroEvolution.evaluate(include, models=model, schedule=schedule, earlyTermination=earlyTermination,
                                        representation=representation, epochs=epochs, sampleSize=sampleSize,
                                        optimizationTask=optimizationTask, evaluation=evaluation, metric=metric), 4)[0]
            sagaIndividual = deap.creator.Individual(include)
            currentFunction = include
            sagaIndividual.fitness.values = bestTrueFitnessValue,
            print('Initial individual fitness is:', bestTrueFitnessValue)
            # print(sagaIndividual, bestTrueFitnessValue)
            # print(type(sagaIndividual), sagaIndividual, sagaIndividual.fitness.values)

        numberOfEvaluations = 0
        generationCounter = 0

        # d = indSize // 2
        surrogateLevel = 0

        pateinceCounter = np.inf
        improvedInLevel = False
        maxNoChangeCounter = 0
        tolernaceCounter = tolerance

        initilizationCounter = 0

        minPopulationSize = 4

        if (verbose):
            print('Current surrogate epochs:', v_epochs)
            print('Current surrogate sample size:', sampleSize)
            print('Current population size:', populationSize)
        end = time.time()
        while (bestTrueFitnessValue < targetFitness and pateinceCounter >= 0 and v_epochs <= epochs and (
                end - start) < timeout and sampleSize <= maxAllowedSize):
            print('patience:', pateinceCounter, ' improved:', improvedInLevel, ' maxnochange:', maxNoChangeCounter,
                  ' initilizationCounter:', initilizationCounter, 'tolerance:', tolernaceCounter)
            # print('saga ind out:', sagaIndividual)
            # print(earlyTermination)
            log, population = NeuroEvolution.CHC(model,
                                                 earlyTermination,
                                                 representation,
                                                 k,
                                                 population,
                                                 d=d,
                                                 divergence=divergence,
                                                 epochs=v_epochs,
                                                 sampleSize=sampleSize,
                                                 populationSize=populationSize,
                                                 maxGenerations=1,
                                                 optimizationTask=optimizationTask,
                                                 evaluation=evaluation,
                                                 metric=metric,
                                                 timeout=np.inf,
                                                 verbose=0)
            generationCounter = generationCounter + int(log.iloc[-1]['generation'])
            activationFunctionIndividual = log.iloc[-1]['best_solution']
            d = log.iloc[-1]['d']

            worstInd = tools.selWorst(population, 1)[0]
            # if (earlyTermination[earlyTermination['epochs'] == v_epochs].iloc[0][
            #    'min_fitness'] < worstInd.fitness.values):
            #    earlyTermination.loc[earlyTermination['epochs'] >= v_epochs, 'min_fitness'] = worstInd.fitness.values
            #    print(earlyTermination)
            #    print()
            # print('this is SAGA d', d)

            end = time.time()
            print(end - start)
            # Check if the original value improved
            if (currentFunction != activationFunctionIndividual):
                currentFunction = activationFunctionIndividual
                trueBestInGeneration = NeuroEvolution.evaluate(activationFunctionIndividual, models=model, earlyTermination=earlyTermination,
                                        representation=representation, epochs=v_epochs * 2, sampleSize=maxAllowedSize,
                                        optimizationTask=optimizationTask, evaluation=evaluation, metric=metric)[0]
                if (trueBestInGeneration > bestTrueFitnessValue):
                    improvedInLevel = True
                    maxNoChangeCounter = 0
                    initilizationCounter = 0
                    bestTrueFitnessValue = trueBestInGeneration
                    pateinceCounter = pateince
                    sagaIndividual = activationFunctionIndividual
                    # print('saga ind:', sagaIndividual)
                    if (verbose):
                        end = time.time()
                        #row = [generationCounter, (end - start), bestTrueFitnessValue,
                        #       sagaIndividual, surrogateLevel, v_epochs*2]
                        #logDF.loc[len(logDF)] = row
                        #print('The best individual is saved', bestTrueFitnessValue)
                        #print(row)
                # elif (trueBestInGeneration < log.iloc[-1]['best_fitness']):
                #    tolernaceCounter -= 1
                #    if (verbose):
                #        print('False Optimum:', trueBestInGeneration)
                #        print(currentFunction)
                elif (trueBestInGeneration <= bestTrueFitnessValue):
                    tolernaceCounter -= 1
                    if (verbose):
                        print('False Optimum:', trueBestInGeneration)
                        print(currentFunction)

            else:
                maxNoChangeCounter += 1

            if (d == 0):
                initilizationCounter += 1

            if (maxNoChangeCounter > maxNoChange or tolernaceCounter < 0 or initilizationCounter > initilizationMax):
                # print(type(sagaIndividual), sagaIndividual)
                if (not improvedInLevel):
                    pateinceCounter -= 1
                else:
                    pateinceCounter = pateince
                maxNoChangeCounter = 0
                initilizationCounter = 0
                tolernaceCounter = tolerance
                currentFunction = sagaIndividual

                if (schedule == 'instance'):
                    sampleSize = sampleSize * 2
                else:
                    v_epochs = v_epochs * 2
                if (bestTrueFitnessValue > targetFitness or pateinceCounter < 0 or v_epochs > epochs or (
                        time.time() - start) > timeout or sampleSize > maxAllowedSize):
                    # print(bestTrueFitnessValue)
                    # print(pateinceCounter)
                    break

                if (improvedInLevel):
                    populationSize = int(populationSize * reductionRate)
                if (populationSize < minPopulationSize):
                    populationSize = minPopulationSize
                improvedInLevel = False
                surrogateLevel += 1
                d = len(population[0]) // 2
                cycle = 1
                # partialModel.setTrainingSample(sampleSize)
                if (representation == 'maxout'):
                    newInd = NeuroEvolution.createPopulationMaxout(populationSize, k)
                else:
                    newInd = NeuroEvolution.createPopulation(populationSize)

                # newInd = NeuroEvolution.createPopulation(populationSize, indSize)
                for individual in newInd:
                    individual.fitness.values = NeuroEvolution.evaluate(individual, models=model,
                                                                        earlyTermination=earlyTermination,
                                                                        representation=representation, epochs=v_epochs,
                                                                        sampleSize=sampleSize,
                                                                        optimizationTask=optimizationTask,
                                                                        evaluation=evaluation, metric=metric)
                sagaIndividual.fitness.values = bestTrueFitnessValue,
                newInd.append(sagaIndividual)
                # for ind in newInd:
                #    print(ind, ind.fitness.values)

                population[:] = tools.selBest(newInd, populationSize)

                if (verbose):
                    print('Current surrogate epochs:', v_epochs)
                    print('Current surrogate sample size:', sampleSize)
                    print('Current population size:', populationSize)
        end = time.time()

        print('Surrogate phase is over!')
        newInd = []
        numberOfMutation = 0
        toolbox = NeuroEvolution.createToolbox(model, earlyTermination, representation, metric, 'CHC', epochs,
                                               sampleSize, optimizationTask, evaluation)
        while (numberOfMutation < minPopulationSize - 1):
            mutant = toolbox.clone(sagaIndividual)
            numberOfMutation += 1
            s = toolbox.mutate(mutant)
            newInd.append(mutant)

        newInd.append(sagaIndividual)
        for individual in newInd:
            individual.fitness.values = NeuroEvolution.evaluate(individual, models=model,
                                                                earlyTermination=earlyTermination,
                                                                representation=representation, epochs=epochs,
                                                                sampleSize=sampleSize,
                                                                optimizationTask=optimizationTask,
                                                                evaluation=evaluation, metric=metric)
            if ((time.time() - start) > timeout):
                return logDF, population

        qualTime = time.time() - start
        log, population = NeuroEvolution.CHC(model,
                                             earlyTermination,
                                             representation,
                                             k,
                                             newInd,
                                             d=d,
                                             divergence=divergence,
                                             epochs=epochs,
                                             sampleSize=maxAllowedSize,
                                             populationSize=minPopulationSize,
                                             maxNochange=np.inf,
                                             optimizationTask=optimizationTask,
                                             evaluation=evaluation,
                                             metric=metric,
                                             timeout=timeout - qualTime,
                                             verbose=0)
        updated = False
        if (len(log) > 0):
            for index, row in log.iterrows():
                if (row['best_fitness'] > logDF.iloc[-1]['best_fitness']):
                    log.at[index, 'time'] = log.loc[index, 'time'] + qualTime
                    log.at[index, 'number_of_evaluations'] = log.loc[
                                                                 index, 'number_of_evaluations'] + numberOfEvaluations
                    log.at[index, 'generation'] = log.loc[index, 'generation'] + logDF.iloc[-1]['generation']
                    log['best_fitness_original'] = log['best_fitness']
                    updated = True
                else:
                    log.drop(index, inplace=True)
        if (updated):
            logDF = pd.concat((logDF, log))
        return logDF, population


# import numpy as np
# import pandas as pd
# import deap
# from deap import tools
# from deap import base, creator
# import time
# from random import randrange
# import copy
# import random
# import Representation
# # import MNIST
# from Model.model import SurvModel
#
# creator.create("FitnessMin", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)
#
# class NeuroEvolution:
#
#     @staticmethod
#     def treeCrossover(ind1, ind2):
#         kmap = {2:[1, 3, 5, 6], 3:[2, 4, 7, 8], 4:[3, 5, 6], 5:[4, 7, 8]}
#         crossPoint = random.choice([2, 3, 4, 5])
#         changedIndexes = kmap[crossPoint]
#
#         oldInd1 = copy.copy(ind1)
#
#         for idx in changedIndexes:
#             ind1[idx] = ind2[idx]
#             ind2[idx] = oldInd1[idx]
#
#         return ind1, ind2
#
#     @staticmethod
#     def HUX(ind1, ind2, fixed=True):
#         # index variable
#         idx = 0
#
#         # Result list
#         res = []
#
#         # With iteration
#         for i in ind1:
#             if i != ind2[idx]:
#                 res.append(idx)
#             idx = idx + 1
#         if (len(res) > 1):
#             numberOfSwapped = randrange(1, len(res))
#             if (fixed):
#                 numberOfSwapped = len(res) // 2
#             indx = random.sample(res, numberOfSwapped)
#
#             oldInd1 = copy.copy(ind1)
#
#             for i in indx:
#                 ind1[i] = ind2[i]
#
#             numberOfSwapped = randrange(1, len(res))
#             if (fixed):
#                 numberOfSwapped = len(res) // 2
#             indx = random.sample(res, numberOfSwapped)
#
#             for i in indx:
#                 ind2[i] = oldInd1[i]
#         return ind1, ind2
#
#     @staticmethod
#     def evaluate(individual, model, epochs=10, evaluation='validation'):
#             model.f = Representation.Function(individual)
#             model.createModel()
#             batchSize = 128
#             loss = "categorical_crossentropy"
#             optimizer = "adam"
#             metric = "accuracy"
#             verbose = 0
#             model.setConfig(optimizer, batchSize, epochs, verbose)
#             model.fitModel()
#             np.random.seed()
#             random.seed()
#             if (evaluation == 'test'):
#                 score = model.model.evaluate(model.X_test, model.y_test, verbose=0)
#                 return score
#             #print(model.history.history)
#             key = 'val_cindex'  # 'val_loss'
#             if (pd.isna(model.history.history[key][-1])):
#                 return np.inf,
#             else:
#                 return model.history.history[key][-1],
#
#     @staticmethod
#     def createPopulation(populationSize, indSize, includeRelu=False):
#         pop = []
#         np.random.seed()
#         random.seed()
#         # relu = deap.creator.Individual([5, 7, 9, 1, 1, 7, 7, 9, 7])
#         relu = deap.creator.Individual([4, 18, 18, 1, 1, 26, 7, 8, 8])
#         for i in range(populationSize):
#              pop.append(deap.creator.Individual([random.randint(1, 6), random.randint(7, 30), random.randint(7, 30), random.randint(1, 6), random.randint(1, 6),
#               random.randint(7, 30), random.randint(7, 30), random.randint(7, 30), random.randint(7, 30)]))
#         if (includeRelu):
#             del pop[-1]
#             pop.append(relu)
#         return list(pop)
#
#     @staticmethod
#     def mutateOperation(ind, numberOfFlipped=1, pruning=0.1):
#         kmap = {2: [1, 3, 5, 6], 3: [2, 4, 7, 8], 4: [3, 5, 6], 5: [4, 7, 8]}
#         crossPoint = random.choice([2, 3, 4, 5])
#         changedIndexes = kmap[crossPoint]
#         #print('before', ind)
#         if random.random() < pruning:
#             #print('pruning')
#             for idx in changedIndexes:
#                 if (idx in [3, 4]):
#                     ind[idx] = 1
#                 else:
#                     ind[idx] = 7
#         else:
#             #print('mutate')
#             for _ in range(numberOfFlipped):
#                 flipedOperation = random.randint(0, len(ind)-1)
#                 if (flipedOperation in [0, 3, 4]):
#                     ind[flipedOperation] = random.randint(1, 6)
#                 else:
#                     ind[flipedOperation] = random.randint(7, 30)
#         #print('after', ind)
#         #print()
#         return ind
#
#     @staticmethod
#     def createToolbox(indSize, model, alg='CHC', epochs=10):
#         toolbox = base.Toolbox()
#         if (alg == 'CHC'):
#             toolbox.register("mate", NeuroEvolution.HUX)
#         elif (alg == 'GA'):
#             toolbox.register("mate", NeuroEvolution.treeCrossover)
#         toolbox.register("mutate", NeuroEvolution.mutateOperation)
#         toolbox.register("select", tools.selTournament, tournsize=3)
#         toolbox.register("evaluate", NeuroEvolution.evaluate, model=model, epochs=epochs)
#         return toolbox
#
#     def CHC(model, population=False, populationSize=40, d=False, divergence=0.35, epochs=10,
#             maxGenerations=np.inf, maxNochange=np.inf, timeout=np.inf,
#             stop=1, verbose=0, include=False):
#         start = time.time()
#         end = time.time()
#
#         indSize = 9
#         toolbox = NeuroEvolution.createToolbox(indSize, model, 'CHC', epochs)
#         if (not population):
#             population = NeuroEvolution.createPopulation(populationSize, indSize, include)
#
#         generationCounter = 0
#         # calculate fitness tuple for each individual in the population:
#         fitnessValues = list(map(toolbox.evaluate, population))
#         for individual, fitnessValue in zip(population, fitnessValues):
#             individual.fitness.values = fitnessValue
#
#         # extract fitness values from all individuals in population:
#         fitnessValues = [individual.fitness.values[0] for individual in population]
#
#         # initialize statistics accumulators:
#         maxFitnessValues = []
#         meanFitnessValues = []
#
#         best = 0 # -1 * np.inf
#         noChange = 0
#         evaulationCounter = populationSize
#
#         d0 = len(population[0]) // 2
#         if (not d):
#             d = d0
#
#         populationHistory = []
#         for ind in population:
#             populationHistory.append(ind)
#
#         logDF = pd.DataFrame(
#             columns=(
#             'generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations', 'best_solution', 'd'))
#
#         # main evolutionary loop:
#         # stop if max fitness value reached the known max value
#         # OR if number of generations exceeded the preset value:
#         while best < stop and generationCounter < maxGenerations and noChange < maxNochange and (
#                 end - start) < timeout:
#             # update counter:
#             generationCounter = generationCounter + 1
#
#             for ind in population:
#                 print(ind, ind.fitness.values)
#             print()
#
#             # apply the selection operator, to select the next generation's individuals:
#             offspring = toolbox.select(population, len(population))
#             # clone the selected individuals:
#             offspring = list(map(toolbox.clone, offspring))
#             random.shuffle(offspring)
#
#             newOffspring = []
#
#             newOffspringCounter = 0
#
#             # apply the crossover operator to pairs of offspring:
#             numberOfPaired = 0
#             numberOfMutation = 0
#             for child1, child2 in zip(offspring[::2], offspring[1::2]):
#                 if NeuroEvolution.hammingDistance(child1, child2) > d:
#                     # print('Before')
#                     # print(child1)
#                     # print(child2)
#                     toolbox.mate(child1, child2)
#                     numberOfPaired += 1
#                     newOffspringCounter += 2
#                     addChild = True
#                     for ind in populationHistory:
#                         if (NeuroEvolution.hammingDistance(ind, child1) == 0):
#                             newOffspringCounter -= 1
#                             addChild = False
#                             break
#                     if (addChild):
#                         populationHistory.append(child1)
#                         newOffspring.append(child1)
#                     addChild = True
#                     for ind in populationHistory:
#                         if (NeuroEvolution.hammingDistance(ind, child2) == 0):
#                             newOffspringCounter -= 1
#                             addChild = False
#                             break
#                     if (addChild):
#                         populationHistory.append(child2)
#                         newOffspring.append(child2)
#                     # print('history length', len(populationHistory))
#                     # print('After')
#                     # print(child1)
#                     # print(child2)
#                     # print()
#                     del child1.fitness.values
#                     del child2.fitness.values
#
#             if (d == 0):
#                 d = d0
#                 newOffspring = []
#                 bestInd = tools.selBest(population, 1)[0]
#                 while (numberOfMutation < len(population)):
#                     mutant = toolbox.clone(bestInd)
#                     numberOfMutation += 1
#                     toolbox.mutate(mutant, divergence)
#                     newOffspring.append(mutant)
#                     del mutant.fitness.values
#
#             # if (newOffspringCounter == 0 and d > 0):
#             #    d -= 1
#
#             # calculate fitness for the individuals with no previous calculated fitness value:
#             freshIndividuals = [ind for ind in newOffspring if not ind.fitness.valid]
#             freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
#             for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
#                 individual.fitness.values = fitnessValue
#
#             evaulationCounter = evaulationCounter + len(freshIndividuals)
#
#             if (numberOfMutation == 0):
#                 oldPopulation = copy.copy(population)
#                 population[:] = tools.selBest(population + newOffspring, populationSize)
#                 differentPopulation = False
#                 for index in range(0, len(population)):
#                     if (NeuroEvolution.hammingDistance(oldPopulation[index], population[index]) != 0):
#                         differentPopulation = True
#                 print(differentPopulation)
#                 if (not differentPopulation):
#                     d -= 1
#             else:
#                 bestInd = tools.selBest(population, 1)
#                 population[:] = tools.selBest(bestInd + newOffspring, populationSize)
#
#             # collect fitnessValues into a list, update statistics and print:
#             fitnessValues = [ind.fitness.values[0] for ind in population]
#
#             maxFitness = max(fitnessValues)
#             if (best >= maxFitness):
#                 noChange += 1
#             if (best < maxFitness):
#                 best = maxFitness
#                 noChange = 0
#             meanFitness = sum(fitnessValues) / len(population)
#             maxFitnessValues.append(maxFitness)
#             meanFitnessValues.append(meanFitness)
#
#             end = time.time()
#
#             # find and print best individual:
#             best_index = fitnessValues.index(max(fitnessValues))
#             if (verbose):
#                 print("Best Individual = %", np.round(maxFitness, 2), ", Gen = ", generationCounter, '\r', end='')
#             # print()
#             print(np.round(maxFitness, 2), 'number of paired:', numberOfPaired, 'number of mutations:',
#                   numberOfMutation, ' d:', d)
#             # print('new', newOffspringCounter)
#             print()
#             end = time.time()
#             row = [generationCounter, (end - start), np.round(maxFitness, 2), meanFitness, evaulationCounter,
#                    population[best_index], d]
#             logDF.loc[len(logDF)] = row
#
#         end = time.time()
#         return logDF, population
#
#     # @staticmethod
#     # def CHC(model, population=False, populationSize=40, d=False, divergence=0.35, zeroP=0.5,
#     #         maxGenerations=np.inf, maxNochange=np.inf, timeout=np.inf,
#     #         stop=np.inf, verbose=0, includeRelu=False):
#     #     start = time.time()
#     #     end = time.time()
#     #
#     #     indSize = 9
#     #     toolbox = NeuroEvolution.createToolbox(indSize, model, 'CHC')
#     #     if (not population):
#     #         population = NeuroEvolution.createPopulation(populationSize, indSize, includeRelu)
#     #
#     #     generationCounter = 0
#     #     # calculate fitness tuple for each individual in the population:
#     #     fitnessValues = list(map(toolbox.evaluate, population))
#     #     for individual, fitnessValue in zip(population, fitnessValues):
#     #         individual.fitness.values = fitnessValue
#     #
#     #     # extract fitness values from all individuals in population:
#     #     fitnessValues = [individual.fitness.values[0] for individual in population]
#     #
#     #     # initialize statistics accumulators:
#     #     maxFitnessValues = []
#     #     meanFitnessValues = []
#     #
#     #     best = 0 #np.inf
#     #     noChange = 0
#     #     evaulationCounter = populationSize
#     #     d0 = len(population[0]) // 2
#     #     if (not d):
#     #         d = d0
#     #
#     #     populationHistory = []
#     #     for ind in population:
#     #         populationHistory.append(ind)
#     #
#     #
#     #     logDF = pd.DataFrame(
#     #         columns=('generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations', 'best_solution', 'd'))
#     #
#     #     # main evolutionary loop:
#     #     # stop if max fitness value reached the known max value
#     #     # OR if number of generations exceeded the preset value:
#     #     while best < stop and generationCounter < maxGenerations and noChange < maxNochange and (
#     #             end - start) < timeout:
#     #         # update counter:
#     #         generationCounter = generationCounter + 1
#     #
#     #         for ind in population:
#     #             print(ind, ind.fitness.values)
#     #         print()
#     #
#     #         # apply the selection operator, to select the next generation's individuals:
#     #         offspring = toolbox.select(population, len(population))
#     #         # clone the selected individuals:
#     #         offspring = list(map(toolbox.clone, offspring))
#     #         random.shuffle(offspring)
#     #
#     #         newOffspring = []
#     #
#     #         newOffspringCounter = 0
#     #
#     #         # apply the crossover operator to pairs of offspring:
#     #         numberOfPaired = 0
#     #         numberOfMutation = 0
#     #         for child1, child2 in zip(offspring[::2], offspring[1::2]):
#     #             if NeuroEvolution.hammingDistance(child1, child2) > d:
#     #                 #print('Before')
#     #                 #print(child1)
#     #                 #print(child2)
#     #                 toolbox.mate(child1, child2)
#     #                 numberOfPaired += 1
#     #                 newOffspringCounter += 2
#     #                 addChild = True
#     #                 for ind in populationHistory:
#     #                     if(NeuroEvolution.hammingDistance(ind, child1) == 0):
#     #                         newOffspringCounter -= 1
#     #                         addChild = False
#     #                         break
#     #                 if (addChild):
#     #                     populationHistory.append(child1)
#     #                     newOffspring.append(child1)
#     #                 addChild = True
#     #                 for ind in populationHistory:
#     #                     if(NeuroEvolution.hammingDistance(ind, child2) == 0):
#     #                         newOffspringCounter -= 1
#     #                         addChild = False
#     #                         break
#     #                 if (addChild):
#     #                     populationHistory.append(child2)
#     #                     newOffspring.append(child2)
#     #                 #print('history length', len(populationHistory))
#     #                 #print('After')
#     #                 #print(child1)
#     #                 #print(child2)
#     #                 #print()
#     #                 del child1.fitness.values
#     #                 del child2.fitness.values
#     #
#     #         if (d == 0):
#     #             d = d0
#     #             newOffspring = []
#     #             bestInd = tools.selBest(population, 1)[0]
#     #             while(numberOfMutation < len(population)):
#     #                 mutant = toolbox.clone(bestInd)
#     #                 numberOfMutation += 1
#     #                 toolbox.mutate(mutant, divergence)
#     #                 newOffspring.append(mutant)
#     #                 del mutant.fitness.values
#     #
#     #         #if (newOffspringCounter == 0 and d > 0):
#     #         #    d -= 1
#     #
#     #         # calculate fitness for the individuals with no previous calculated fitness value:
#     #         freshIndividuals = [ind for ind in newOffspring if not ind.fitness.valid]
#     #         freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
#     #         for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
#     #             individual.fitness.values = fitnessValue
#     #
#     #         evaulationCounter = evaulationCounter + len(freshIndividuals)
#     #
#     #         if (numberOfMutation == 0):
#     #             oldPopulation = copy.copy(population)
#     #             population[:] = tools.selBest(population + newOffspring, populationSize)
#     #             differentPopulation = False
#     #             for index in range(0, len(population)):
#     #                 if (NeuroEvolution.hammingDistance(oldPopulation[index], population[index]) != 0):
#     #                         differentPopulation = True
#     #             print(differentPopulation)
#     #             if (not differentPopulation):
#     #                 d -= 1
#     #         else:
#     #             bestInd = tools.selBest(population, 1)
#     #             population[:] = tools.selBest(bestInd + newOffspring, populationSize)
#     #
#     #         # collect fitnessValues into a list, update statistics and print:
#     #         fitnessValues = [ind.fitness.values[0] for ind in population]
#     #
#     #         maxFitness = max(fitnessValues)
#     #         if (best >= maxFitness):
#     #             noChange+= 1
#     #         if (best < maxFitness):
#     #             best = maxFitness
#     #             noChange = 0
#     #         meanFitness = sum(fitnessValues) / len(population)
#     #         maxFitnessValues.append(maxFitness)
#     #         meanFitnessValues.append(meanFitness)
#     #
#     #         end = time.time()
#     #
#     #         # find and print best individual:
#     #         print('fitnessValues:', fitnessValues)
#     #         best_index = fitnessValues.index(max(fitnessValues))
#     #         if (verbose):
#     #             print("Best Individual = %", np.round(100 * maxFitness, 2), ", Gen = ", generationCounter, '\r', end='')
#     #         #print()
#     #         print(np.round(100*maxFitness, 2), 'number of paired:', numberOfPaired, 'number of mutations:', numberOfMutation, ' d:', d)
#     #         #print('new', newOffspringCounter)
#     #         print()
#     #         end = time.time()
#     #         row = [generationCounter, (end - start), np.round(100 * maxFitness, 2), meanFitness, evaulationCounter,
#     #                population[best_index], d]
#     #         logDF.loc[len(logDF)] = row
#     #
#     #     end = time.time()
#     #     return logDF, population
#
#     @staticmethod
#     def GA(model, population=False, populationSize=40, crossOverP=0.9, mutationP=0.1, zeroP=0.5,  maxGenerations=np.inf, maxNochange=np.inf,
#             timeout=np.inf, stop=np.inf, verbose=0):
#
#         start = time.time()
#         end = time.time()
#
#         indSize = 9
#         toolbox = NeuroEvolution.createToolbox(indSize, model, 'GA')
#         if (not population):
#                 population = NeuroEvolution.createPopulation(populationSize, indSize)
#
#         generationCounter = 0
#         # calculate fitness tuple for each individual in the population:
#         fitnessValues = list(map(toolbox.evaluate, population))
#         for individual, fitnessValue in zip(population, fitnessValues):
#             individual.fitness.values = fitnessValue
#
#         # extract fitness values from all individuals in population:
#         fitnessValues = [individual.fitness.values[0] for individual in population]
#
#         # initialize statistics accumulators:
#         maxFitnessValues = []
#         meanFitnessValues = []
#
#         best = np.inf
#         noChange = 0
#         evaulationCounter = populationSize
#
#         logDF = pd.DataFrame(
#             columns=('generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations', 'best_solution'))
#
#         # main evolutionary loop:
#         # stop if max fitness value reached the known max value
#         # OR if number of generations exceeded the preset value:
#         while best < stop and generationCounter < maxGenerations and noChange < maxNochange and (
#                 end - start) < timeout:
#             generationCounter = generationCounter + 1
#
#             for ind in population:
#                 print(ind, ind.fitness.values)
#             print()
#
#             # apply the selection operator, to select the next generation's individuals:
#             offspring = toolbox.select(population, len(population))
#             # clone the selected individuals:
#             offspring = list(map(toolbox.clone, offspring))
#
#             # apply the crossover operator to pairs of offspring:
#
#             for child1, child2 in zip(offspring[::2], offspring[1::2]):
#                 if random.random() < crossOverP:
#                     toolbox.mate(child1, child2)
#                     del child1.fitness.values
#                     del child2.fitness.values
#
#             for mutant in offspring:
#                 if random.random() < mutationP:
#                     toolbox.mutate(mutant)
#                     del mutant.fitness.values
#
#             # calculate fitness for the individuals with no previous calculated fitness value:
#             freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
#             freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
#             for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
#                 individual.fitness.values = fitnessValue
#
#             evaulationCounter = evaulationCounter + len(freshIndividuals)
#
#             population[:] = tools.selBest(population + offspring, populationSize)
#
#             # collect fitnessValues into a list, update statistics and print:
#             fitnessValues = [ind.fitness.values[0] for ind in population]
#
#             maxFitness = max(fitnessValues)
#             if (best >= maxFitness):
#                 noChange += 1
#             if (best < maxFitness):
#                 best = maxFitness
#                 noChange = 0
#             meanFitness = sum(fitnessValues) / len(population)
#             maxFitnessValues.append(maxFitness)
#             meanFitnessValues.append(meanFitness)
#
#             end = time.time()
#
#             # find and print best individual:
#             best_index = fitnessValues.index(max(fitnessValues))
#             if (verbose):
#                 print("Best   Individual = ", np.round(maxFitness, 2), ", Gen = ", generationCounter, '\r', end='')
#             # print()
#             # print(np.round(100*maxFitness, 2), 'number of paired:', numberOfPaired, 'number of mutations:', numberOfMutation, ' d:', d)
#             # print()
#             end = time.time()
#             row = [generationCounter, (end - start), np.round(100 * maxFitness, 2), meanFitness, evaulationCounter,
#                    population[best_index]]
#             logDF.loc[len(logDF)] = row
#
#         end = time.time()
#         return logDF, population
#
#     @staticmethod
#     def hammingDistance(ind1, ind2):
#         ind1 = np.array(ind1)
#         ind2 = np.array(ind2)
#         return (len(ind1)-(np.sum(np.equal(ind1, ind2))))
#
#
#
#


'''

import numpy as np
import pandas as pd
import deap
from deap import tools
from deap import base, creator
import time
from random import randrange
import copy
import random
import Representation
import MNIST
from tensorflow import keras


creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

class NeuroEvolution:

    @staticmethod
    def treeCrossover(ind1, ind2):
        kmap = {2:[1, 3, 5, 6], 3:[2, 4, 7, 8], 4:[3, 5, 6], 5:[4, 7, 8]}
        crossPoint = random.choice([2, 3, 4, 5])
        changedIndexes = kmap[crossPoint]

        oldInd1 = copy.copy(ind1)

        for idx in changedIndexes:
            ind1[idx] = ind2[idx]
            ind2[idx] = oldInd1[idx]

        return ind1, ind2

    @staticmethod
    def HUX(ind1, ind2, fixed=True):
        # index variable
        idx = 0

        # Result list
        res = []

        # With iteration
        for i in ind1:
            if i != ind2[idx]:
                res.append(idx)
            idx = idx + 1
        if (len(res) > 1):
            numberOfSwapped = randrange(1, len(res))
            if (fixed):
                numberOfSwapped = len(res) // 2
            indx = random.sample(res, numberOfSwapped)

            oldInd1 = copy.copy(ind1)

            for i in indx:
                ind1[i] = ind2[i]

            numberOfSwapped = randrange(1, len(res))
            if (fixed):
                numberOfSwapped = len(res) // 2
            indx = random.sample(res, numberOfSwapped)

            for i in indx:
                ind2[i] = oldInd1[i]
        return ind1, ind2

    @staticmethod
    def evaluate(individual, model, epochs=10, evaluation='validation'):
            model.f = Representation.Function(individual)
            model.createModel()
            batchSize = 128
            loss = "categorical_crossentropy"
            optimizer = "adam"
            metric = "accuracy"
            patience = 10
            verbose = 0
            model.setConfig(loss, optimizer, metric, batchSize, epochs, patience, verbose)
            model.fitModel()
            np.random.seed()
            random.seed()
            if (evaluation == 'test'):
                score = model.model.evaluate(model.X_test, model.y_test, verbose=0)
                return score
            #print(model.history.history)
            if (pd.isna(model.history.history['val_loss'][-1])):
                return -1 * np.inf,
            else:
                return -1 * np.min(model.history.history['val_loss'][-1]),

    @staticmethod
    def createPopulation(populationSize, indSize, include=False):
        pop = []
        np.random.seed()
        random.seed()
        relu = deap.creator.Individual([5, 7, 9, 1, 1, 7, 7, 9, 7])
        if (include):
            include = deap.creator.Individual(include)
        for i in range(populationSize):
             pop.append(deap.creator.Individual([random.randint(1, 6), random.randint(7, 30), random.randint(7, 30), random.randint(1, 6), random.randint(1, 6),
              random.randint(7, 30), random.randint(7, 30), random.randint(7, 30), random.randint(7, 30)]))
        if (include):
            del pop[-1]
            pop.append(include)
        return list(pop)

    @staticmethod
    def mutateOperation(ind, numberOfFlipped=1, pruning=0.1):
        kmap = {2: [1, 3, 5, 6], 3: [2, 4, 7, 8], 4: [3, 5, 6], 5: [4, 7, 8]}
        crossPoint = random.choice([2, 3, 4, 5])
        changedIndexes = kmap[crossPoint]
        #print('before', ind)
        if random.random() < pruning:
            #print('pruning')
            for idx in changedIndexes:
                if (idx in [3, 4]):
                    ind[idx] = 1
                else:
                    ind[idx] = 7
        else:
            #print('mutate')
            for _ in range(numberOfFlipped):
                flipedOperation = random.randint(0, len(ind)-1)
                if (flipedOperation in [0, 3, 4]):
                    ind[flipedOperation] = random.randint(1, 6)
                else:
                    ind[flipedOperation] = random.randint(7, 30)
        #print('after', ind)
        #print()
        return ind

    @staticmethod
    def createToolbox(indSize, model, alg='CHC', epochs=10):
        toolbox = base.Toolbox()
        if (alg == 'CHC'):
            toolbox.register("mate", NeuroEvolution.HUX)
        elif (alg == 'GA'):
            toolbox.register("mate", NeuroEvolution.treeCrossover)
        toolbox.register("mutate", NeuroEvolution.mutateOperation)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", NeuroEvolution.evaluate, model=model, epochs=epochs)
        return toolbox

    @staticmethod
    def CHC(model, population=False, populationSize=40, d=False, divergence=0.35, epochs=10,
            maxGenerations=np.inf, maxNochange=np.inf, timeout=np.inf,
            stop=np.inf, verbose=0, include=False):
        start = time.time()
        end = time.time()

        indSize = 9
        toolbox = NeuroEvolution.createToolbox(indSize, model, 'CHC', epochs)
        if (not population):
            population = NeuroEvolution.createPopulation(populationSize, indSize, include)

        generationCounter = 0
        # calculate fitness tuple for each individual in the population:
        fitnessValues = list(map(toolbox.evaluate, population))
        for individual, fitnessValue in zip(population, fitnessValues):
            individual.fitness.values = fitnessValue

        # extract fitness values from all individuals in population:
        fitnessValues = [individual.fitness.values[0] for individual in population]

        # initialize statistics accumulators:
        maxFitnessValues = []
        meanFitnessValues = []

        best = -1 * np.inf
        noChange = 0
        evaulationCounter = populationSize

        d0 = len(population[0]) // 2
        if (not d):
            d = d0

        populationHistory = []
        for ind in population:
            populationHistory.append(ind)

        logDF = pd.DataFrame(
            columns=('generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations', 'best_solution', 'd'))

        # main evolutionary loop:
        # stop if max fitness value reached the known max value
        # OR if number of generations exceeded the preset value:
        while best < stop and generationCounter < maxGenerations and noChange < maxNochange and (
                end - start) < timeout:
            # update counter:
            generationCounter = generationCounter + 1

            for ind in population:
                print(ind, ind.fitness.values)
            print()

            # apply the selection operator, to select the next generation's individuals:
            offspring = toolbox.select(population, len(population))
            # clone the selected individuals:
            offspring = list(map(toolbox.clone, offspring))
            random.shuffle(offspring)

            newOffspring = []

            newOffspringCounter = 0

            # apply the crossover operator to pairs of offspring:
            numberOfPaired = 0
            numberOfMutation = 0
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if NeuroEvolution.hammingDistance(child1, child2) > d:
                    #print('Before')
                    #print(child1)
                    #print(child2)
                    toolbox.mate(child1, child2)
                    numberOfPaired += 1
                    newOffspringCounter += 2
                    addChild = True
                    for ind in populationHistory:
                        if(NeuroEvolution.hammingDistance(ind, child1) == 0):
                            newOffspringCounter -= 1
                            addChild = False
                            break
                    if (addChild):
                        populationHistory.append(child1)
                        newOffspring.append(child1)
                    addChild = True
                    for ind in populationHistory:
                        if(NeuroEvolution.hammingDistance(ind, child2) == 0):
                            newOffspringCounter -= 1
                            addChild = False
                            break
                    if (addChild):
                        populationHistory.append(child2)
                        newOffspring.append(child2)
                    #print('history length', len(populationHistory))
                    #print('After')
                    #print(child1)
                    #print(child2)
                    #print()
                    del child1.fitness.values
                    del child2.fitness.values

            if (d == 0):
                d = d0
                newOffspring = []
                bestInd = tools.selBest(population, 1)[0]
                while(numberOfMutation < len(population)):
                    mutant = toolbox.clone(bestInd)
                    numberOfMutation += 1
                    toolbox.mutate(mutant, divergence)
                    newOffspring.append(mutant)
                    del mutant.fitness.values

            #if (newOffspringCounter == 0 and d > 0):
            #    d -= 1

            # calculate fitness for the individuals with no previous calculated fitness value:
            freshIndividuals = [ind for ind in newOffspring if not ind.fitness.valid]
            freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
            for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
                individual.fitness.values = fitnessValue

            evaulationCounter = evaulationCounter + len(freshIndividuals)

            if (numberOfMutation == 0):
                oldPopulation = copy.copy(population)
                population[:] = tools.selBest(population + newOffspring, populationSize)
                differentPopulation = False
                for index in range(0, len(population)):
                    if (NeuroEvolution.hammingDistance(oldPopulation[index], population[index]) != 0):
                            differentPopulation = True
                print(differentPopulation)
                if (not differentPopulation):
                    d -= 1
            else:
                bestInd = tools.selBest(population, 1)
                population[:] = tools.selBest(bestInd + newOffspring, populationSize)

            # collect fitnessValues into a list, update statistics and print:
            fitnessValues = [ind.fitness.values[0] for ind in population]

            maxFitness = max(fitnessValues)
            if (best >= maxFitness):
                noChange+= 1
            if (best < maxFitness):
                best = maxFitness
                noChange = 0
            meanFitness = sum(fitnessValues) / len(population)
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)

            end = time.time()

            # find and print best individual:
            best_index = fitnessValues.index(max(fitnessValues))
            if (verbose):
                print("Best Individual = %", np.round(maxFitness, 2), ", Gen = ", generationCounter, '\r', end='')
            #print()
            print(np.round(maxFitness, 2), 'number of paired:', numberOfPaired, 'number of mutations:', numberOfMutation, ' d:', d)
            #print('new', newOffspringCounter)
            print()
            end = time.time()
            row = [generationCounter, (end - start), np.round(maxFitness, 2), meanFitness, evaulationCounter,
                   population[best_index], d]
            logDF.loc[len(logDF)] = row

        end = time.time()
        return logDF, population

    @staticmethod
    def GA(model, population=False, populationSize=40, crossOverP=0.9, mutationP=0.1, zeroP=0.5,  maxGenerations=np.inf, maxNochange=np.inf,
            timeout=np.inf, stop=np.inf, verbose=0):

        start = time.time()
        end = time.time()

        indSize = 9
        toolbox = NeuroEvolution.createToolbox(indSize, model, 'GA')
        if (not population):
                population = NeuroEvolution.createPopulation(populationSize, indSize)

        generationCounter = 0
        # calculate fitness tuple for each individual in the population:
        fitnessValues = list(map(toolbox.evaluate, population))
        for individual, fitnessValue in zip(population, fitnessValues):
            individual.fitness.values = fitnessValue

        # extract fitness values from all individuals in population:
        fitnessValues = [individual.fitness.values[0] for individual in population]

        # initialize statistics accumulators:
        maxFitnessValues = []
        meanFitnessValues = []

        best = -1 * np.inf
        noChange = 0
        evaulationCounter = populationSize

        logDF = pd.DataFrame(
            columns=('generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations', 'best_solution'))

        # main evolutionary loop:
        # stop if max fitness value reached the known max value
        # OR if number of generations exceeded the preset value:
        while best < stop and generationCounter < maxGenerations and noChange < maxNochange and (
                end - start) < timeout:
            generationCounter = generationCounter + 1

            for ind in population:
                print(ind, ind.fitness.values)
            print()

            # apply the selection operator, to select the next generation's individuals:
            offspring = toolbox.select(population, len(population))
            # clone the selected individuals:
            offspring = list(map(toolbox.clone, offspring))

            # apply the crossover operator to pairs of offspring:

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossOverP:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutationP:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # calculate fitness for the individuals with no previous calculated fitness value:
            freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
            freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
            for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
                individual.fitness.values = fitnessValue

            evaulationCounter = evaulationCounter + len(freshIndividuals)

            population[:] = tools.selBest(population + offspring, populationSize)

            # collect fitnessValues into a list, update statistics and print:
            fitnessValues = [ind.fitness.values[0] for ind in population]

            maxFitness = max(fitnessValues)
            if (best >= maxFitness):
                noChange += 1
            if (best < maxFitness):
                best = maxFitness
                noChange = 0
            meanFitness = sum(fitnessValues) / len(population)
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)

            end = time.time()

            # find and print best individual:
            best_index = fitnessValues.index(max(fitnessValues))
            if (verbose):
                print("Best Individual = ", np.round(maxFitness, 2), ", Gen = ", generationCounter, '\r', end='')
            # print()
            # print(np.round(100*maxFitness, 2), 'number of paired:', numberOfPaired, 'number of mutations:', numberOfMutation, ' d:', d)
            # print()
            end = time.time()
            row = [generationCounter, (end - start), np.round(100 * maxFitness, 2), meanFitness, evaulationCounter,
                   population[best_index]]
            logDF.loc[len(logDF)] = row

        end = time.time()
        return logDF, population

    @staticmethod
    def hammingDistance(ind1, ind2):
        ind1 = np.array(ind1)
        ind2 = np.array(ind2)
        return (len(ind1)-(np.sum(np.equal(ind1, ind2))))


    @staticmethod
    def SAGA(model, populationSize=40, reductionRate=0.5, pateince=2, step=10, d=False, divergence=3, epochs=10, targetFitness=0,
             verbose=0, qualOnly=False, timeout=np.inf, include=False,
             noChange=np.inf, evaluation='validation'):

        start = time.time()
        logDF = pd.DataFrame(columns=('generation', 'time', 'best_fitness',
                                       'best_solution', 'surrogate_level', 'sample_size'))
        partialModel = copy.copy(model)
        sampleSize = 100
        partialModel.setTrainingSample(sampleSize)

        task = 'feature_selection'
        indSize = 9
        population = NeuroEvolution.createPopulation(populationSize, indSize, include)

        bestTrueFitnessValue = np.inf
        sagaActivationFunction = [1] * 9
        qual = False

        numberOfEvaluations = 0
        generationCounter = 0
        maxAllowedSize = int(partialModel.X_train.shape[0])

        d = indSize//2
        surrogateLevel = 0

        pateince0 = pateince

        while (bestTrueFitnessValue > targetFitness and pateince > 0 and sampleSize < maxAllowedSize):
            #print('patience:', pateince)
            if (verbose):
                print('Current Approx Sample Size:', sampleSize)
                print('Current Population Size:', populationSize)
            pateince-=1
            log, population = NeuroEvolution.CHC(partialModel,
                                                                      population,
                                                                      d=d,
                                                                      divergence=divergence,
                                                                      epochs=epochs,
                                                                      populationSize=populationSize,
                                                                      maxNochange=step,
                                                                      verbose=verbose)
            generationCounter = generationCounter + int(log.iloc[-1]['generation'])
            activationFunctionIndividual = log.iloc[-1]['best_solution']


            approxBestInGeneration = np.round(
                    NeuroEvolution.evaluate(activationFunctionIndividual, partialModel, epochs), 2)[0]
            end = time.time()

                # Check if the original value improved
            if (sagaActivationFunction != activationFunctionIndividual):
                    pateince = pateince0
                    bestTrueFitnessValue = -1 * approxBestInGeneration
                    sagaActivationFunction = activationFunctionIndividual
                    sagaIndividual = tools.selBest(population, 1)
                    row = [generationCounter, (end - start), bestTrueFitnessValue,
                           sagaActivationFunction, surrogateLevel, sampleSize]
                    logDF.loc[len(logDF)] = row
                    if (verbose):
                        print('The best individual is saved', bestTrueFitnessValue)
                        print(row)

            sampleSize = sampleSize * 2
            populationSize = int(populationSize * reductionRate)
            surrogateLevel+=1
            d = indSize // 2
            partialModel.setTrainingSample(sampleSize)
            newInd = NeuroEvolution.createPopulation(populationSize, indSize)


            population[:] = tools.selBest(sagaIndividual + newInd, populationSize)

        return logDF, population

'''
