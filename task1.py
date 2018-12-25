from pyeasyga import pyeasyga

def run():
    with open('8.txt') as f:
        max_weight, max_volume = [float(x) for x in next(f).split()]
        data = []
        for line in f:
            data.append([float(x) for x in line.split()])
    ga = pyeasyga.GeneticAlgorithm(data)
    ga.population_size = 250

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
    print(j)
    print("a")
    return j

result1 = run()
print("a")
print(result1)
