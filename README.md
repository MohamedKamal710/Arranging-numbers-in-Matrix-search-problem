# Arranging numbers in Matrix search problem
Implementation of IDDFS , BFS , A-star Algorithms to solve a problem

## The problem
Given a Matrix of numbers with size 3x3 (1-9 no duplicates) or 4x4 (1-12 no duplicates), The goal is to arrange numbers in an increasing order.
In each move we can only switch to adjacent numbers , Vertical or Horizontal.
For example:
given a matrix 3X3 
![image](https://user-images.githubusercontent.com/52383427/176709626-d667aa98-80bc-4645-b03f-95964bc76b26.png)

The Goal:
![image](https://user-images.githubusercontent.com/52383427/176710329-f794f198-3780-4c2f-9b67-b1121f777e48.png)

The States that have passed from first till the goal:
![image](https://user-images.githubusercontent.com/52383427/176710439-af0b5522-77cf-401e-82a7-6073fa3145cb.png)


---------------------------------------------------------------------------------------

3 Algorithms were implemented in order to solve the problem:
BFS , Iterative Deepning DFS , A-Star(*).

---------------------------------------------------------------------------------------
The format in which you can run the script:

python Matrix_Search_Problem.py  input_.txt  "algorithm_name" 

algorithm_name could be = "bfs" or "iddfs" or "astar"
