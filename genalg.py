from pyeasyga import pyeasyga
import random
import json


POPULATION_SIZE = 200
GENERATIONS_NUMBER = 500
SELECTION_K = 0.2
MUTATE_K = 0.05
SELECTION_DELTA = 0.1

with open('8.txt') as f:
    max_weight, max_volume = [float(x) for x in next(f).split()]
    data = []
    for line in f:
        data.append([float(x) for x in line.split()])

def task1(data):

    ga = pyeasyga.GeneticAlgorithm(data)
    ga.population_size = POPULATION_SIZE

    def fitness(individual, data):
            weight, volume, price = 0, 0, 0
            for (selected, item) in zip(individual, data):
                if selected:
                    weight += item[0]
                    volume += item[1]
                    price += item[2]
            if weight > max_weight or volume > max_volume:
                price = 0
            return price

    ga.fitness_function = fitness
    ga.run()
    result = ga.best_individual()
    items = []
    for i in range(0, len(result[1])):
        if result[1][i] == 1:
            items.append(i+1)
    real_volume = 0
    real_weight = 0
    for i in range(0, len(items)):
        real_volume += data[items[i]-1][1]
        real_weight += data[items[i]-1][0]
    j = dict(value=result[0], weight=real_weight, volume=real_volume, items=items)
    return j

class Individual:
    def __init__(self):
        # Вектор значений для каждой вещи есть/нет
        self.values = list()
        # Суммарная стоимость (функция приспособленности)
        self.cost = int()
        # Для отбора в 20% перед скрещиванием. Не самая лучшая реализация.
        self.selected = False
        # Перед созданием потомков ставим всем существующим особям флаг, что они старые.
        self.old = False

class Task2:

    def clean_selected_in_population(self):
        for individual in self.population:
            individual.selected = False

    def calculate_individual_cost(self, individual):
        weight, volume, price = 0, 0, 0
        for (selected, item) in zip(individual.values, self.data):
            if selected:
                weight += item[0]
                volume += item[1]
                price += item[2]
        if weight > self.max_weight or volume > self.max_volume:
            price = 0
        return price

    def create_individual(self):
        individual = Individual()
        for i in range(0, len(self.data)):
            value = random.randint(0,1)
            individual.values.append(value)
        individual.cost = self.calculate_individual_cost(individual)
        return individual

    def create_population(self):
        for i in range(0, POPULATION_SIZE):
            self.population.append(self.create_individual())

    def selection(self):
        """Отбор особей для скрещивания. Выбираем только 20% самых лучших."""

        amount = int(len(self.population) * SELECTION_K)
        selected_population = list()
        self.population.sort(key=lambda x: x.cost, reverse=True)
        selected_population = self.population[0:amount]
        self.clean_selected_in_population()
        return selected_population

    def make_children(self, parent1, parent2):
        """Делаем результат скрещивания."""

        child1 = Individual()
        child2 = Individual()
        for index, (value1, value2) in enumerate(zip(parent1.values, parent2.values)):
            rnd = random.random()
            if rnd >= 0.5:
                child1.values.insert(index, value1)
                child2.values.insert(index, value2)
            else:
                child1.values.insert(index, value2)
                child2.values.insert(index, value1)
        child1.cost = self.calculate_individual_cost(child1)
        child2.cost = self.calculate_individual_cost(child2)
        return child1, child2

    def crossover(self):
        """Скрещивание. Однородное (каждый бит от случайного родителя)"""

        selection = self.selection()
        next_generation = list()
        for individual in self.population:
            individual.old = True
        for index in range(0, len(selection) - 1, 2):
            parent1 = selection[index]
            parent2 = selection[index + 1]
            children = self.make_children(parent1, parent2)
            next_generation.append(children[0])
            next_generation.append(children[1])
        self.population.extend(next_generation)

    def mutate(self):
        """Мутация. Случайное изменение 3х битов у 5% особей."""
        bit = 3
        individuals = [random.randint(0, len(self.population)-1) for i in range(int(len(self.population) * MUTATE_K))]
        bits = [random.randint(0, len(self.population[0].values)-1) for i in range(bit)]
        for i in individuals:
            for j in bits:
                self.population[i].values[j] = 1 - self.population[i].values[j]

    def select_new_generation(self):
        """Замена родителей."""

        for individual in self.population:
            individual.cost = self.calculate_individual_cost(individual)
            if individual.old is True:
                individual.cost *= 0

        self.population.sort(key=lambda x: x.cost, reverse=True)
        self.population = self.population[0:POPULATION_SIZE]
        self.create_result(self.population[0])

    def create_result(self, result):
        items = []
        for i in range(0, len(result.values)):
            if result.values[i] == 1:
                items.append(i+1)
        real_volume = 0
        real_weight = 0
        real_price = 0
        for i in range(0, len(items)):
            real_volume += self.data[items[i] - 1][1]
            real_weight += self.data[items[i] - 1][0]
            real_price += self.data[items[i] - 1][2]
        j = dict(value=real_price, weight=real_weight, volume=real_volume, items=items)
        return j

    def __init__(self, data):
        self.data = data
        self.population = list()
        self.max_weight = max_weight
        self.max_volume = max_volume

    def run(self):
        prev_best = 0
        for i in range(0, GENERATIONS_NUMBER):
            self.create_population()
            self.crossover()
            self.mutate()
            self.select_new_generation()
            if prev_best == 0:
                prev_best = self.calculate_individual_cost(self.population[0])
            else:
                delta = self.calculate_individual_cost(self.population[0]) / prev_best - 1
                if delta < self.SELECTION_DELTA:
                    break
                else:
                    prev_best = self.calculate_individual_cost(self.population[0])
            return self.create_result(self.population[0])



result1 = task1(data)
task2 = Task2(data)
result2 = task2.run()

full_result = {"1": result1, "2": result2}
print(json.dumps(full_result, indent=4, sort_keys=True))
