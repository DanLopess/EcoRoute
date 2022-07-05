# SmartEcoRoute

Many cities around the world are starting to implement smart Ecopoints.

The purpose of this system is to efficiently find the best route in order to pass through all Ecopoints of a city, using Genetic Algorithms and Ant Colony Optimization. As a research purpose it should also be interesting to evaluate the performance of each.

## Techniques & Approach

The test case being considered has around 100 ecopoints.

To find the best route two **Heuristic Optimization** algorithms were implemented:

- Genetic Algorithm
- Ant Colony Optimization

## Libraries Used

- Genetic Algorithm: **deap**
- Ant Colony Optimization: **ACO-Pants**

## How to Run

To run this project you need to create an environment using the specification file for conda environments: `environment.yml`.

You can create a new environment with the following command:

`conda env create -f environment.yml`

To run each of the algorithms you can simply define the points that you want the algorithm to find a route for, which can be done through the file `input/list_of_points.csv`. Also note that if no points are defined it will mean that the algorithm should find a path through all nodes defined on the distances matrix (`data/DistancesMatrix.xlsx`).

Then you can run the commands:

`python genetic_algorithms.py`
or
`python ant_colony_optimization.py`

## Results

The results for this project can be found at `results_report.pdf`.
