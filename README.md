# Support-Vector-Machine
Linear support vector machine implementation that uses the CVXOPT library for performing optimisation step of finding decision boundary.

## Output
Output appears in the following order:
1) Training Accuracy of Scikit learn library
2) Testing accuracy of Scikit learn library
3) CVXOPT optimization of full datasetreadouts
4) Training/Testing accuracy table in order of Scikit Learn, primal soft, primal hard, dual soft, dual hard
5) Table about decision boundary parameters
6) Dual/Primal parameter comparison table

## Definitions:
There are 4 SVM implementations present that all solve the same problem using different formulations of the problem.

**Primal**: The original formulation of the problem.

**Dual**: The dual form of the original problem.

**Hard**: Uses hard margins where all points of one class must fall on one side of the decision boundary.

**Soft**: Uses soft margins where points may fall either side of the boundary.

For more informations on what the primal and dual problems are and what is meant by soft and hard margins see here:
https://en.wikipedia.org/wiki/Support-vector_machine
