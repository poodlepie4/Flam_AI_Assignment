Research and Development / AI Assignment
Problem Statement
Find the values of unknown parameters (θ, M, X) in the given parametric equation:

x = (t*cos(θ) - e^(M|t|)*sin(0.3t)sin(θ) + X)
y = (42 + tsin(θ) + e^(M|t|)*sin(0.3t)*cos(θ))

Approach
The dataset xy_data.csv was provided containing points (x, y) on the curve for 6 < t < 60.
The equation parameters θ, M, and X were optimized using nonlinear least squares (L1 distance minimization) in Python.
Parameter ranges were constrained as per problem statement:
0° < θ < 50°
-0.05 < M < 0.05
0 < X < 100
Optimization was performed using scipy.optimize.least_squares method.
Final parameters were used to visualize the curve in Desmos.
Final Parameter Values
Parameter	Symbol	Value
Theta (radians)	θ	0.523598303175
M	M	0.029999996873
X	X	54.999998
Final Parametric Equation
(x(t), y(t)) = (tcos(0.523598303175) - e^(0.029999996873abs(t))sin(0.3t)sin(0.523598303175) + 54.999998, 42 + tsin(0.523598303175) + e^(0.029999996873*abs(t))sin(0.3t)*cos(0.523598303175))


Explanation of Process
The Python script reads (x, y) data points.
It defines the given parametric equations for x(t) and y(t).
It computes the error between predicted and actual (x, y).
Using optimization, it minimizes the L1 distance to get best-fit values.
Results are visualized using Desmos for verification.
