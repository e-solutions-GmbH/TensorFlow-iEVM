# Benchmark

## MNIST

./examples/benchmark.py

### NVIDIA Tesla V100 16GB & Intel Xeon Silver 4116 CPU @ 2.10GHz (left) vs. Intel Core i7-10875H CPU @ 2.30GHz (right)

| Benchmark                                                                               | Eager Execution Time [s] | Non-Eager Execution Time [s] |     | Eager Execution Time [s] | Non-Eager Execution Time [s] |
|-----------------------------------------------------------------------------------------|--------------------------|------------------------------|-----|--------------------------|------------------------------|
| One single small fit / predict                                                          | 5.3 / 0.15               | 8.16 / 0.79                  |     | **1.63 / 1.11**          | **2.71 / 1.2**               |
| One single large fit / predict                                                          | **20.89 / 0.94**         | **12.13 / 1.55**             |     | 23.99 / 11.04            | 18.74 / 9.02                 |
| Partial Fit avg</p>tail_size=100, batch_size=128</p>tail_size=25, batch_size=16         | </p>1.0</p>1.16          | </p>0.95</p>1.04             |     | **</p>0.75</p>1.05**     | **</p>0.84</p>0.99**         |
| Partial Fit avg ('max')</p>tail_size=100, batch_size=128</p>tail_size=25, batch_size=16 | </p>0.94</p>0.59         | </p>1.76</p>2.2              |     | **</p>0.87</p>0.26**     | **</p>1.08</p>0.93**         |
| Partial Fit avg ('all')</p>tail_size=100, batch_size=128</p>tail_size=25, batch_size=16 | </p>0.91</p>0.6          | </p>1.68</p>2.19             |     | **</p>0.44</p>0.21**     | **</p>0.87</p>0.89**         |
| Model Reduction Bisection Set Cover</p>(after each partial fit; avg)                    | 24.46                    | 1.16                         |     | **10.98**                | **0.09**                     |
| Model Reduction Weighted Set Cover</p>(after each partial fit; avg)                     | 0.95                     | 0.1                          |     | **0.35**                 | **0.02**                     |
| GridSearchCV avg over multiple fold-fits                                                | **1.47**                 | **1.06**                     |     | 1.67                     | 1.28                         |

