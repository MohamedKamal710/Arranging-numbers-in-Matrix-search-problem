from ctypes.wintypes import CHAR
import sys
import numpy as np
import copy


class Node:
    def __init__(self,state,parent,action):
        self.state = state
        self.parent = parent
        self.action = action
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1


#Implementation of Queue Datastructure with its main functions
class Queue:
    def __init__(self):
        self.queue = []
    
    def push(self,node):
        self.queue.append(node)

    def containsState(self,state):
        return any(np.array_equal(state,node) for node in self.queue)

    def isEmpty(self):
        return len(self.queue) == 0
    
    def pop(self):
        if self.isEmpty():
            return Exception("Empty Queue")
        else:
            nodeToReturn = self.queue[0]
            self.queue = self.queue[1:]
            return nodeToReturn

    def peak(self):
        if self.isEmpty():
            return Exception("Empty Queue")
        else:
            return self.queue[0]



#Implementatino of Stack(inherit the queue class and override pop and peak methods)
class Stack(Queue):
    def pop(self):
        if self.isEmpty():
            return Exception("Empty Stack")
        else:
            nodeToReturn = self.queue[-1]
            self.queue = self.queue[:-1]
            return nodeToReturn
    
    def peak(self):
        if self.isEmpty():
            return Exception("Empty Stack")
        else:
            return self.queue[-1]



#Implementation of priority Queue for the sake of heuristic
class priorityQueue(Queue):
    def pop(self):
        if self.isEmpty():
            return Exception("Empty Queue")
        else:
            min = sys.maxsize
            toReturn = None
            for node in self.queue:
                if(min > node[0]):
                    min = node[0]
                    toReturn = node
            self.queue.remove(toReturn)
            return toReturn
                



class MatrixPuzzle():

    def __init__(self,initialState , goalState,algo):
        self.startState = initialState
        self.goalState = goalState
        self.matrixSize = len(initialState)
        self.visitedStates = set()
        self.algorithmType = algo




    def emptyVisitedStates(self):
        self.visitedStates = set()

    

    def isVisited(self,state):
        matrixText = ""
        for i in range(self.matrixSize):
            for j in range(self.matrixSize):
                matrixText+= str(state[i][j])
        # for mat in self.visitedStates:
        #     if(np.array_equal(mat,state)):
        if matrixText in self.visitedStates:
            return True 
        return False       



    def addToVisited(self,state):
        matrixText = ""
        for i in range(self.matrixSize):
            for j in range(self.matrixSize):
                matrixText+= str(state[i][j])
        self.visitedStates.add(matrixText)
    


    #Given a node, return all its possible neighbors, in other words it returns all the states that can be reached from the given state.
    def nextState(self,currentNode):
        states = []
        tempState = currentNode.state.copy()
        for i in range(self.matrixSize):
            for j in range(self.matrixSize):
                if(j+1 < self.matrixSize):
                    tempVar = tempState[i][j]
                    tempState[i][j] = tempState[i][j+1]
                    tempState[i][j+1] = tempVar
                    # if(i == 0 and j == 0):
                    #     print(self.visitedStates)
                    #if(not self.isVisited(tempState)):
                    tempNode = Node(tempState,currentNode,action=None)
                    states.append(tempNode)
                tempState = currentNode.state.copy()
                if(i+1 < self.matrixSize):
                    tempVar = tempState[i][j]
                    tempState[i][j] = tempState[i+1][j]
                    tempState[i+1][j] = tempVar
                    #if(not self.isVisited(tempState)):
                    tempNode = Node(tempState,currentNode,action=None)
                    states.append(tempNode)
                tempState = currentNode.state.copy()
        return states
    


    #Function that is responsible for printing a given state..
    def printState(self,state):
        for i in range(self.matrixSize):
            for j in range(self.matrixSize):
                print(state[i][j], end="  ")
            print()
        print("*********************")



    #This function checks if the given state meets the goal state and return true if so, otherwise false
    def isGoal(self,node):
        if(np.array_equal(node.state,self.goalState)):
            return True
        return False



    #This function takes a node as an input, and keeps tracking back to the root node so we can get the path to the solution
    def getSolution(self,node):
        sol = []
        print("\nSolution is found in depth = ",node.depth)
        while node.parent is not None:
            sol.append(node.state)
            node = node.parent
        sol.append(node.state)
        return sol



    #Solve function decides which algorithm to use to solve the problem according the user input
    def Solve(self):
        solution = None
        statesVisited = 0
        if self.algorithmType.lower() == 'bfs':
           solution,statesVisited = self.SolveByBFS()
        elif self.algorithmType.lower() == 'astar':
            solution,statesVisited = self.SolveByAstar()
        elif self.algorithmType.lower() == 'id':
           solution,statesVisited = self.SolveByIDDFS()
        else:
            print("The algorithm you have entered is not supported.")
        
        if(solution is None):
            print("Couldn't resolve the problem.")
        else:
            lastSol = self.getSolution(solution)
            printSolution(lastSol,statesVisited)



    #Function that runs the BFS algorithm in order to solve the problem
    def SolveByBFS(self):
        allPossibleStates = Queue()
        # solution = None
        statesVisited = 0
        mainNode = Node(self.startState,parent=None,action=None)
        allPossibleStates.push(mainNode)
        self.addToVisited(mainNode.state)
        while not allPossibleStates.isEmpty():
            statesVisited+=1
            stateToCheck = allPossibleStates.pop()
            print("States that have been expanded == ",statesVisited)
            # self.printState(stateToCheck.state)
            if(self.isGoal(stateToCheck)):
                return stateToCheck,statesVisited

            for node in self.nextState(stateToCheck): 
                if not self.isVisited(node.state):
                    self.addToVisited(node.state)
                    allPossibleStates.push(node)

        return None,None



    #Here are 2 functions that are related to the iterative deepening algorithm..
    #First function calls the second function every time with increasing depth
    def SolveByIDDFS(self):
        solution = None
        depth = 0 
        rootNode = Node(self.startState,None,None)
        while solution is None:
            solution,statesVisited = self.DLS(rootNode,depth)
            if(solution is not None):
                return solution,statesVisited
            depth+=1

    def DLS(self,node,limitDepth):
        print("Starting Search with depth limit ",limitDepth)
        print()
        statesVisited = 0
        statesStack = Stack()
        statesStack.push(node)
        self.emptyVisitedStates()
        self.addToVisited(node.state)
        while not statesStack.isEmpty():
            statesVisited+=1
            currentNode = statesStack.pop()
            
            print("States that have been checked == ",statesVisited)
            # self.printState(currentNode.state)
            if self.isGoal(currentNode):
                return currentNode,statesVisited
            if currentNode.depth < limitDepth:
                for eachNode in reversed(self.nextState(currentNode)):
                    if not self.isVisited(eachNode.state):
                        self.addToVisited(eachNode.state)
                        statesStack.push(eachNode)
        return None,None



    #All methods that are related to A* algorithm
    ## -------------------------------------------

    #returns the distance of the num of its original place using manhatten block.
    def calculateManhatten(self,indexI,indexJ,num):
        for i in range(self.matrixSize):
            for j in range(self.matrixSize):
                if(num == self.goalState[i][j]):
                    return abs(indexI - i) + abs(indexJ - j)
        return -1



    #returns the heuristical cost
    def heuristic(self,state):
        h = 0
        for i in range(self.matrixSize):
            for j in range(self.matrixSize):
                h += self.calculateManhatten(i,j,state[i][j])
        return h



    #Function f , returns the heuristic + original cost   
    def normalCost(self,node):
        return self.heuristic(node.state) + node.depth



    #Return next states with calculated heuristical cost
    def getNextStatesWithHeuristics(self,node):
        possibleStates = self.nextState(node)
        statesWithH = []
        for eachN in possibleStates:
            statesWithH.append((self.normalCost(eachN),eachN))
        return statesWithH



    #Solve the problem with A* algorithm using priority queue
    def SolveByAstar(self):
        prioStates = priorityQueue()
        statesVisited= 0
        rootNode = Node(self.startState,None,None)
        prioStates.push((self.normalCost(rootNode),rootNode))
        self.addToVisited(rootNode.state)
        while not prioStates.isEmpty():
            statesVisited +=1
            tempN = prioStates.pop()[1]
            print("States that have been checked == ",statesVisited)
            # self.printState(tempN.state)
            if(self.isGoal(tempN)):
                return tempN,statesVisited
            
            for st in self.getNextStatesWithHeuristics(tempN):
                if not self.isVisited(st[1].state):
                    prioStates.push(st)
                    self.addToVisited(st[1].state)
        return None,None
    
#---------------------------------------------------------



#Function that is responsible of printing the solution from the start state to goal state and all the states that lie between them.
def printSolution(states,expandedStates):
    print("\n------- Solution ------- \n")
    print("States that have been expanded :: ",expandedStates)
    c = 0
    for w in reversed(states):
        c+=1
        print()
        for j in range(len(w)):
            print(w[j])
        # print(w, '\n')
        print()
        if c < len(states):
            print('   ▼   ')
    print("------ END OF SOLUTION -------")




# startState = np.array([[5,3,2,4],[1,6,7,8],[10,9,11,12],[14,13,15,16]]) 
inpt = np.loadtxt(sys.argv[1] , dtype='i' , delimiter=" ")
algo = (sys.argv[2])
if(len(inpt) == 3):
    game = MatrixPuzzle(inpt,np.array([[1,2,3],[4,5,6],[7,8,9]]),algo)
    game.Solve()
else:
    game = MatrixPuzzle(inpt,np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]),algo)
    game.Solve()
