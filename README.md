The following details are available at [Craft](https://craft.tianrui-qi.com/admm-for-svm):
- Instruction about alternating direction method of multipliers (ADMM) and augmented Lagrangian method (ALM)
- Derivation of ADMM for SVM
- Simulation results in MATLAB

## Formulate the primal problem

Given a set of data $\{ \left( \mathbf{x}_i, \mathbf{y}_i \right) \}^N_{i=1}$ 
with $\mathbf{x}_i \in \mathbb{R}^p$ and the label 
$\mathbf{y}_i \in  \{ +1, -1 \}$ for each $i$, the linear support vector machine
(SVM) aims at finding a hyperplane  $\mathbf{x}\mathbf{w}+b = 1$ to separate the
two classes of data points. Suppose these points can be strictly separated; 
namely, there exists $\left( \mathbf{w}, b \right)$ such that 
$\mathbf{x}_i\mathbf{w} +b \geq 1$ if $\mathbf{y}_i = +1$ and 
$\mathbf{x}_i\mathbf{w} +b \leq 1$ if $\mathbf{y}_i = -1$, or equivalently 
$\mathbf{y}_i \left( \mathbf{x}_i \mathbf{w} +b \right)\geq 1$ for all $i$. To 
find a hyperplane with the maximum margin between the two classes, we solve the 
following problem:

$$\max_{\mathbf{w} ,b} \frac{1}{\left\Vert \mathbf{w} \right\Vert  } :\mathbf{y}_{i} \left( \mathbf{x} \mathbf{w}_{i} +b\right)  \geq 1,\forall i=1,\cdots ,N$$

or equivalently

$$\min_{\mathbf{w} ,b} \frac{1}{2} \left\Vert \mathbf{w} \right\Vert^{2}  :\mathbf{y}_{i} \left( \mathbf{x} \mathbf{w}_{i} +b\right)  \geq 1,\forall i=1,\cdots ,N$$

where $\left\Vert\cdot\right\Vert$ denotes the Euclidean norm (or 2-norm). 

While the feasibility of the above problems depends on the separability of the 
data, a soft-margin SVM is employed if the data points can not be strictly 
separated by solving the problem:

$$\min_{\mathbf{w} ,b} \sum^{N}_{i=1} \mathbf{t}_{i}+\frac{\lambda }{2} \left\Vert \mathbf{w} \right\Vert^{2}  :\mathbf{y}_{i} \left( \mathbf{x} \mathbf{w}_{i} +b\right)  \geq 1-\mathbf{t}_{i},\mathbf{t}_{i}\geq 0,\forall i=1,\cdots ,N$$

Note that at the optimal solution 
$\left( \bar{\mathbf{w} } ,\bar{b} ,\bar{\mathbf{ t}} \right)$ , we must have 
$\bar{\mathbf{t} }_{i} =\max \left( 0,1-\mathbf{y}_{i} \left( \mathbf{x}_{i} \bar{\mathbf{w} } +\bar{b} \right)  \right)$. 
Thus, we can rewrite the equation into 

$$\min_{\mathbf{w} ,b} \sum^{N}_{i=1} \max \left( 0,1-\mathbf{y}_{i} \left( \mathbf{x}_{i} \bar{\mathbf{w} } +\bar{b} \right)  \right)  +\frac{\lambda }{2} \left\Vert \mathbf{w} \right\Vert^{2}  \qquad \lambda > 0$$

## Reference

1. Rensselaer Polytechnic Institute, Spring 2022, MATP 4820 Computational Optimization by Prof. Yangyang Xu, *Final Project of MATP4820*.
2. Ye, Gui–Bo, Yifei Chen, and Xiaohui Xie. "Efficient variable selection in support vector machines via the alternating direction method of multipliers." *Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics*. JMLR Workshop and Conference Proceedings, 2011. [[link](https://proceedings.mlr.press/v15/ye11a/ye11a.pdf)]
