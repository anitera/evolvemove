import numpy as np
import operator
import random
import copy
from tkinter import *
import time

class Point:
    def __init__(self, x=None, y=None):
        self.coord = None
        if (x is not None) and (y is not None):
            self.coord = (x,y)
        else:
            self.coord = (10,10) #int(random.random() * 200)


class Individ:
    def __init__(self, step, size):
        self.route = np.random.randint(-1, 2, size=(2, size)) * step  # step*(-1) / step*1 / step*0
        self.individ_size = size
        self.fitness = 0
        self.setfitness()

    def setfitness(self):  # start_point = (x,y)
        global start_point
        global end_point
        global gameField
        cumsum = np.cumsum(self.route, axis=1) + np.array(start_point).reshape((2, 1))
        index = self.checkblocks(cumsum, gameField)
        # print ('cumsum ', cumsum[:, index])
        distance = np.linalg.norm(cumsum[:, index] - np.array(end_point))
        self.fitness = distance  # check normalization

    def getfitness(self):
        return self.fitness

    def checkblocks(self, cumsum, our_field):
        global gameField
        for i in range(1, self.individ_size):
            if (any(cumsum[:, i] < 0) or (cumsum[0, i] >= gameField.field_size[0]) or (
                cumsum[1, i] >= gameField.field_size[1])):
                return i - 1
            elif gameField.getvalue(cumsum[0, i], cumsum[1, i]):
                return i - 1
        return self.individ_size - 1

    def getFinalRoute(self):
        global start_point
        global end_point
        global gameField
        cumsum = np.cumsum(self.route, axis=1) + np.array(start_point).reshape((2, 1))
        index = self.checkblocks(cumsum, gameField)
        return cumsum[:, :index + 1]

class Field:
    def __init__(self, size):
        self.matrix = np.zeros(size)
        self.matrix[0, :] = 1
        self.matrix[:, 0] = 1
        self.matrix[:, size[1] - 1] = 1
        self.matrix[size[0] - 1, :] = 1
        self.field_size = size

    def addSquareBlock(self, left_top, right_bottom):  # point = (x,y)
        for i in range(left_top[0], right_bottom[0]):
            for j in range(left_top[1], right_bottom[1]):
                self.matrix[i, j] = 1

    def addCircle(self, center, radius):  # center = (x,y)
        for i in range(min(0, center[0] - radius), max(self.field_size[0], center[0] + radius)):
            for j in range(min(0, center[1] - radius), max(self.field_size[1], center[1] + radius)):
                if (center[0] - i) ** 2 + (center[1] - j) ** 2 <= radius ** 2:
                    self.matrix[i, j] = 1
                    
                    
    def newstar(self, start, height): #two triangles
        #coef = width/height
        coef = 1
        #self.matrix[start[0], start[1]] = 1
        for i in range(1, int(0.2*height)+1):
            coef = 1
            for j in range(0, i):
                self.matrix[start[0]+i, start[1]+int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]-int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]] = 1
                
        #coef = width/height     

        for i in range(int(0.2*height+1), int(0.8*height-1)):
            
            for j in range(0, i):
                self.matrix[start[0]+i, start[1]+int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]-int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]] = 1
            #coef = coef - 0.01     
        for i in range(int(0.2*height)+1, int(height)):
            for j in range(int(0.8*height-i), 0, -1):
                self.matrix[start[0]+i, start[1]+int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]-int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]] = 1
                
        for i in range(int(0.8*height)-1, height+1):
            coef = 1
            for j in range(int(height-i-2), -1, -1):
                self.matrix[start[0]+i, start[1]+int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]-int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]] = 1
            #coef = coef + 0.01
        #self.matrix[start[0]+int(0.8*height+1), start[1]] = 1
        

    def snowflake(self, center, radius):
        for i in range(0, radius+1):
            self.matrix[center[0], center[1]+i] = 1
            self.matrix[center[0]+i, center[1]] = 1
            self.matrix[center[0]+i, center[1]+i] = 1
            self.matrix[center[0], center[1]] = 1
            self.matrix[center[0]-i, center[1]-i] = 1
            self.matrix[center[0]-i, center[1]] = 1
            self.matrix[center[0], center[1]-i] = 1
            self.matrix[center[0]-i, center[1]+i] = 1
            self.matrix[center[0]+i, center[1]-i] = 1
            if i == (radius-1):
                self.matrix[center[0]+i-1, center[1]-i-1] = 1
                self.matrix[center[0]+i+1, center[1]-i+1] = 1
                self.matrix[center[0]-i-1, center[1]+i-1] = 1
                self.matrix[center[0]-i+1, center[1]+i+1] = 1
                self.matrix[center[0]-i+1, center[1]-i-1] = 1
                self.matrix[center[0]-i-1, center[1]-i+1] = 1
                self.matrix[center[0]+i+1, center[1]+i-1] = 1
                self.matrix[center[0]+i-1, center[1]+i+1] = 1
                
            
        for i in range(min(0,center[0]-radius), max(self.field_size[0],center[0]+radius)):
            for j in range(min(0,center[1]-radius), max(self.field_size[1],center[1]+radius)):
                if (center[0]-i)**2+(center[1]-j)**2 - (radius-1)**2 == 1:
                    self.matrix[i, j] = 1
    
    def сhristmas_tree(self, start, height):
        for i in range(1, int(0.3*height)+1):
            coef = 0.6
            for j in range(0, i):
                self.matrix[start[0]+i, start[1]+int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]-int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]] = 1
        for i in range(int(0.3*height)-1, int(0.6*height)+1):
            coef = 0.8
            for j in range(0, i-int(0.3*height)+1):
                self.matrix[start[0]+i, start[1]+int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]-int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]] = 1
                
        for i in range(int(0.6*height)-1, int(0.9*height)+1):
            coef = 1
            for j in range(0, i-int(0.6*height)+1):
                self.matrix[start[0]+i, start[1]+int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]-int(j*coef)] = 1
                self.matrix[start[0]+i, start[1]] = 1
                
        for i in range(int(0.9*height)-1, height):
            coef = 1
            for j in range(start[1]-int(0.1*height), start[1]+int(0.1*height)+1):
                self.matrix[start[0]+i, int(j*coef)] = 1
                    


    # add diff shapes of blocks

    def getmatrix(self):
        return self.matrix

    def getvalue(self, x, y):
        return self.matrix[x, y]

class Population:
    def __init__(self, step, individSize, populationSize):  # , start_point, end_point, our_field):
        self.population = []
        # self.fitness = []
        for i in range(0, populationSize):
            newInd = Individ(step, individSize)  # , start_point, end_point, our_field)
            self.population.append(newInd)
            # self.fitness.append(newInd.fitness)

    def getFitness(self):
        fitness = [x.fitness for x in self.population]
        return fitness
        # return self.fitness

    def getBest(self, n):
        # sortedPopulation = [pop for _,pop in sorted(zip(self.fitness, self.population), key = lambda x: x[0])]
        sortedPopulation = sorted(self.population, key=operator.attrgetter("fitness"))
        if n == 1:
            return sortedPopulation[0]
        else:
            return sortedPopulation[:n]

    def add_individ(self, newIndivid):
        self.population.append(newIndivid)
        # self.fitness.append(newIndivid.fitness)


class GA:
    def __init__(self, user_mutationRate, user_crossoverProbability, user_elitism,
                 user_crossoverFunction, user_parentSelection):

        self.mutationRate = user_mutationRate
        # self.tournamentSize = 10
        self.crossoverProbability = user_crossoverProbability
        self.elitism = user_elitism

        if user_crossoverFunction == 1:
            self.crossover = self.crossover_1point
        elif user_crossoverFunction == 2:
            self.crossover = self.crossover_2point

        if user_parentSelection == 'wheel':
            self.chooseParents = self.chooseParents_wheel
        elif user_parentSelection == 'tournament':
            self.chooseParents = self.chooseParents_tournament

    def CreateFirstPopulation(self, user_step, user_individSize, user_populationSize):
        initialPop = Population(step=user_step, individSize=user_individSize, populationSize=user_populationSize)
        return initialPop

    def chooseParents_wheel(self, generation):
        reversed_fitness = 1.0 / np.array(generation.getFitness())
        select_prob = reversed_fitness / sum(reversed_fitness)
        selected = np.random.choice(generation.population, 2, replace=False)  # p=select_prob, replace=False)
        return list(selected)
    
    
    def chooseParents_tournament(self, generation):
        ''' Eficiency depends on tournament size '''
        parents = []
        
        for j in range(0, self.numOfParents):
            selection = []
            for i in range(0, self.tournamentSize):
                randomId = random.randint(0, len(generation.population)-1)#int(random.randrange(0, len(generation.population)))
                selection.append(generation.population[randomId])
           # print ('selection ', selection)
            parents.append(self.getFittestForTournament(selection))
            print('parent fitness ', parents[j].fitness)
            
        return list(parents)
    
    def chooseParents_elittournament(self, generation):
        ''' Generation could easily degenerate, but for small cases works faster '''
        parents = []
        
        for j in range(0, self.numOfParents):
            selection = self.getFittestForTournament(generation, self.tournamentSize)
            randomId = int(random.randrange(0, len(selection)))
            parents.append(selection[randomId])
            #print ('selection ', selection)
            print('parent fitness ', parents[j].fitness)
        return list(parents)
    
    def getFittestForTournament(self, selection, number = None):
        ''' Choose best individs from list '''
        if number:
            population = {}
            for i in range(0, len(selection.population)):
                population[selection.population[i].getfitness()] = selection.population[i]
            fittest = [population[key] for key in sorted(population)[:number]]
            print('length of fittest ', len(fittest))
            return fittest
                
        else:
            fittest = selection[0]
            for s in selection:
                if fittest.getfitness() >= s.getfitness():
                    fittest = s
            return fittest

    def get_breakes(self, n):
        break1, break2 = random.sample(range(0, n), 2)
        if break1 < break2:
            return (break1, break2)
        else:
            return (break2, break1)

    def crossover_2point(self, parents):
        children = copy.deepcopy(parents)
        pp = random.random()
        if pp <= self.crossoverProbability:
            startPos, endPos = self.get_breakes(parents[0].individ_size)
            children[0].route[:, startPos:endPos], children[1].route[:, startPos:endPos] = \
                copy.deepcopy(children[1].route[:, startPos:endPos]), copy.deepcopy(
                    children[0].route[:, startPos:endPos])
        return children

    def crossover_1point(self, parents):
        children = copy.deepcopy(parents)
        pp = random.random()
        if pp <= self.crossoverProbability:
            startPos = random.randint(0, parents[0].individ_size)
            children[0].route[:, startPos:], children[1].route[:, startPos:] = \
                copy.deepcopy(children[1].route[:, startPos:]), copy.deepcopy(children[0].route[:, startPos:])
        return children

    def mutate(self, children):
        if random.random() < self.mutationRate:
            indx = random.randint(0, children[0].individ_size - 1)
            children[0].route[0, indx], children[0].route[1, indx] = children[0].route[1, indx].copy(), \
                                                                     children[0].route[0, indx].copy()

        if random.random() < self.mutationRate:
            indx = random.randint(0, children[1].individ_size - 1)
            children[1].route[0, indx], children[1].route[1, indx] = children[1].route[1, indx].copy(), \
                                                                     children[1].route[0, indx].copy()

        return children

    def fittest(self, child_list, n):  # n - how much return
        child_list.sort(key=lambda x: x.fitness)
        return child_list[:n]

    def evolve(self, step, individSize, populationSize, generation,
               user_chooseFromAll):  # generation - previous population

        children_list = []
        for i in range(populationSize):
            # choose parents
            parents = self.chooseParents(generation)

            # make 2 children
            children = self.crossover(parents)

            # mutate
            children = self.mutate(children)

            children[0].setfitness()
            children[1].setfitness()

            children_list = children_list + children

        # Create new generation with elitism?
        if user_chooseFromAll:
            newPopulation = Population(step=step, individSize=individSize, populationSize=0)
            for child in children_list:
                generation.add_individ(child)
            best_individs = generation.getBest(populationSize)
            for individ in best_individs:
                newPopulation.add_individ(individ)
        else:
            newPopulation = Population(step=step, individSize=individSize, populationSize=0)
            if self.elitism:
                newPopulation.add_individ(generation.getBest(1))
                best_children = self.fittest(children_list, populationSize - 1)
            else:
                best_children = self.fittest(children_list, populationSize)
            for child in best_children:
                newPopulation.add_individ(child)

        return newPopulation

class GUI(Frame):
    def __init__(self, root):
        super().__init__()
        self.coord = None
        self.initUI(root)

    def initUI(self, root):
        self.master.title("Project")
        self.grid(row = 0, column = 1, rowspan = 20)
        self.canvas = Canvas(self,width=500, height=500,)
        self.canvas.create_rectangle(0, 0, 500, 500,
                                outline="#00fbaa", fill="#00fbaa")

        self.canvas.grid(row=0, column=0, sticky="we")

    def animation(self):
        self.canvas.create_rectangle(start_point[0], start_point[1], start_point[0]+5, start_point[1]+5, outline="#a50", fill="#a50")
        self.canvas.create_rectangle(end_point[0], end_point[1], end_point[0]+5, end_point[1]+5, outline="#a50", fill="#a50")

        # algorithm

        ga = GA(user_mutationRate=mutationRate, user_crossoverProbability=crossoverProbability, user_elitism=elitism,
                user_crossoverFunction=crossoverFunc, user_parentSelection=parentFunc)
        initialPop = ga.CreateFirstPopulation(user_step=step, user_individSize=individSize,
                                              user_populationSize=populationSize)
        print('min fitness ', min(initialPop.getFitness()))
        print('evolve function: ')
        dont_change = 0
        prev = -1
        while (not any(np.array(initialPop.getFitness()) == 0)) and (dont_change < 50):
            initialPop = ga.evolve(step=step, individSize=individSize, populationSize=populationSize,
                                   generation=initialPop,
                                   user_chooseFromAll=chooseFromAll)
            # print('all fitness ', initialPop.getFitness())
            print('min fitness ', min(initialPop.getFitness()))
            if prev == min(initialPop.getFitness()):
                dont_change += 1
            prev = min(initialPop.getFitness())
            print(dont_change)


        best_individ = initialPop.getBest(1)
        self.coord = best_individ.getFinalRoute()

        for i in range(self.coord.shape[1]):
            time.sleep(0.005)
            self.canvas.create_rectangle(self.coord[:, i][0], self.coord[:, i][1], self.coord[:, i][0]+3, self.coord[:, i][1]+3,
                                    outline="#f50", fill="#f50")
            self.canvas.update()


def anime():
    root = Tk()
    #### start algorithm
    #gameField = gf
    # gameField.addSquareBlock((41, 20), (100, 40))
    # gameField.addSquareBlock((0, 60), (59, 80))
    # gameField.addSquareBlock((40,60),(60,70))
    # gameField.addCircle((70,70),10)
    # ex = GUI(vector, root)
    root.geometry("900x600")
    e = Entry(root)
    l = Label(root, text = 'Start coordinate')
    e.focus_set()
    e.grid(row=0, column=0)
    l.grid(row=1, column=0)

    e1 = Entry(root)
    l1 = Label(root, text='End coordinate')
    e1.focus_set()
    e1.grid(row=2, column=0)
    l1.grid(row=3, column=0)

    e2 = Entry(root)
    l2 = Label(root, text = 'Crossover probability')
    e2.focus_set()
    e2.grid(row=4, column=0)
    l2.grid(row=5, column=0)

    e3 = Entry(root)
    l3 = Label(root, text='Mutation rate')
    e3.focus_set()
    e3.grid(row=6, column=0)
    l3.grid(row=7, column=0)

    e4 = Entry(root)
    l4 = Label(root, text='Step')
    e4.focus_set()
    e4.grid(row=8, column=0)
    l4.grid(row=9, column=0)

    e5 = Entry(root)
    l5 = Label(root, text='Elitism')
    e5.focus_set()
    e5.grid(row=10, column=0)
    l5.grid(row=11, column=0)

    e6 = Entry(root)
    l6 = Label(root, text='Size of individ')
    e6.focus_set()
    e6.grid(row=12, column=0)
    l6.grid(row=13, column=0)

    e7 = Entry(root)
    l7 = Label(root, text='Population size')
    e7.focus_set()
    e7.grid(row=14, column=0)
    l7.grid(row=15, column=0)

    e8 = Entry(root)
    l8 = Label(root, text='Choose from all?')
    e8.focus_set()
    e8.grid(row=16, column=0)
    l8.grid(row=17, column=0)

    e9 = Entry(root)
    l9 = Label(root, text='Crossover function')
    e9.focus_set()
    e9.grid(row=18, column=0)
    l9.grid(row=19, column=0)

    e10 = Entry(root)
    l10 = Label(root, text='Parent function')
    e10.focus_set()
    e10.grid(row=20, column=0)
    l10.grid(row=21, column=0)

    e11 = Entry(root)
    l11 = Label(root, text = 'Start coordinate')
    e11.focus_set()
    e11.grid(row=0, column=3)
    l11.grid(row=1, column=3)

    e12 = Entry(root)
    l12 = Label(root, text='End coordinate')
    e12.focus_set()
    e12.grid(row=2, column=3)
    l12.grid(row=3, column=3)

    e13 = Entry(root)
    l13 = Label(root, text = 'Crossover probability')
    e13.focus_set()
    e13.grid(row=4, column=3)
    l13.grid(row=5, column=3)

    e14 = Entry(root)
    l14 = Label(root, text='Mutation rate')
    e14.focus_set()
    e14.grid(row=6, column=3)
    l14.grid(row=7, column=3)

    e15 = Entry(root)
    l15 = Label(root, text='Step')
    e15.focus_set()
    e15.grid(row=8, column=3)
    l15.grid(row=9, column=3)

    e16 = Entry(root)
    l16 = Label(root, text='Elitism')
    e16.focus_set()
    e16.grid(row=10, column=3)
    l16.grid(row=11, column=3)

    e17 = Entry(root)
    l17 = Label(root, text='Size of individ')
    e17.focus_set()
    e17.grid(row=12, column=3)
    l17.grid(row=13, column=3)

    e18 = Entry(root)
    l18 = Label(root, text='Population size')
    e18.focus_set()
    e18.grid(row=14, column=3)
    l18.grid(row=15, column=3)

    e19 = Entry(root)
    l19 = Label(root, text='Choose from all?')
    e19.focus_set()
    e19.grid(row=16, column=3)
    l19.grid(row=17, column=3)

    e20 = Entry(root)
    l20 = Label(root, text='Crossover function')
    e20.focus_set()
    e20.grid(row=18, column=3)
    l20.grid(row=19, column=3)

    e21 = Entry(root)
    l21 = Label(root, text='Parent function')
    e21.focus_set()
    e21.grid(row=20, column=3)
    l21.grid(row=21, column=3)


    def set_params():
        global start_point, end_point, mutationRate, crossoverProbability, elitism, step, individSize, populationSize, chooseFromAll, crossoverFunc, parentFunc
        start_point = (int(str(e.get()).split(' ')[0]), int(str(e.get()).split(' ')[1]))
        print(start_point)
        end_point = (int(str(e1.get()).split(' ')[0]), int(str(e1.get()).split(' ')[1]))
        crossoverProbability = float(str(e2.get()))
        mutationRate = float(e3.get())
        step = int(e4.get())

    b = Button(root, text="Coord X Start", width=10, command=lambda: set_params())
    b.grid(row = 0, column = 4)

    ex = GUI(root)
    b1 = Button(root, text="Start", command=ex.animation)
    b1.grid(row = 0, column = 5)

    root.mainloop()

if __name__ == '__main__':
    # common parameters for all users
    gameField = Field(size=(600, 600))
    
    gameField.addCircle((90,90), 5)
    gameField.addSquareBlock((21,30),(40,50))
    gameField.newstar((10,20), 20)
    gameField.snowflake((45,50), 5)
    gameField.сhristmas_tree((50,50), 40)
    
    
    start_point = (10, 10)
    end_point = (150, 100)

    mutationRate = 0.1
    crossoverProbability = 0.7
    elitism = True
    step = 2
    individSize = 800
    populationSize = 100
    chooseFromAll = False
    crossoverFunc = 2
    #'wheel', 'tournament'(for tournament - small tournament size 2-3),'elit_tournament'(for elit big tournamentSize)
    parentFunc = 'wheel'
    anime()