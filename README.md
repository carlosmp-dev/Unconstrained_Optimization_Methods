Unconstrained Optimization Algorithms and Application in Image Analysis

This repository contains my project for ECE505, which explores various algorithms for unconstrained optimization and demonstrates their implementation. Through this project, I experimented with different optimization algorithms, comparing their performances on specific mathematical problems and applying them to a practical image processing task.

Project Overview

The project is divided into two main parts:

Part 1: Algorithmic Implementation

In this section, I implemented several optimization algorithms and tested them using specific problems provided in the course materials. The algorithms include:
	1.	Steepest Descent Algorithm
	•	Implemented to solve unconstrained optimization problems using a linear search algorithm for parameter tuning.
	2.	Newton’s Algorithm
	•	Tested using the same problem set as the Steepest Descent Algorithm to observe its convergence and efficiency.
	3.	BFGS Quasi-Newton Algorithm
	•	This quasi-Newton method was implemented to provide an efficient approach to unconstrained optimization, particularly suitable for complex, high-dimensional problems.
	4.	Conjugate Gradient Algorithm
	•	An iterative method for solving large systems of linear equations, optimized with a line search algorithm.

Each algorithm implementation includes detailed parameterization and test results, which are presented through graphs and tables.

Part 2: Application in Mammogram Image Analysis (Optional)

This optional section involves applying optimization techniques to image analysis, specifically for estimating cancerous pixel proportions in a mammogram. The project uses a Gaussian distribution model to distinguish cancerous regions from non-cancerous backgrounds based on pixel intensity. By estimating the distribution parameters using maximum likelihood estimation (MLE), the goal is to calculate the proportion of cancerous pixels in the image.

Project Structure

	•	functions.py: Contains implementations of the above algorithms with functions to perform parameter tuning, optimization calculations, and graphing of results.
	•	Report: A PDF report with detailed explanations of each algorithm, parameter tuning methods, and a summary of the test results, along with tables and graphs.

How to Use

	1.	Clone the repository.
	2.	Open functions.py to access individual function implementations for each algorithm.
	3.	Run the functions with your chosen parameters, or use the provided problem sets as inputs to replicate the results from the report.
	4.	Review the report for insights into the performance of each algorithm and the results of the image analysis application.

Results and Discussion

Each algorithm’s performance is documented in the report, with test results that compare the convergence rates, efficiency, and overall performance on specific problem sets. The optional image analysis section provides an example of how optimization techniques can be applied to real-world scenarios.

Requirements

	•	Python 3
	•	Required libraries: numpy, matplotlib, etc. (install them using pip install -r requirements.txt)

Future Work

Further improvements could involve optimizing the algorithms for larger datasets or extending the image analysis application to include more advanced machine learning techniques.