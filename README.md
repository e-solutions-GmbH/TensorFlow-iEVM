# Extreme Value Machine - TensorFlow

1. [Introduction](#introduction)
2. [Install](#install)
3. [Limitations](#limitations)
4. [Roadmap](#roadmap)
5. [Example Results](#example-results)
6. [References](#references)
7. [Related Projects](#related-projects)

## Introduction

Implementation of the Extreme Value Machine (EVM) using [TensorFlow](https://www.tensorflow.org/)
operations.
This implementation is largely compatible with [scikit-learn](https://scikit-learn.org/).

This implementation comprises the incremental EVM [1], vanilla EVM [2], and C-EVM [3].
If you use one of the EVMs in a scientific publication, please cite the appropriate reference,
see [References](#references).

## Install

### Requirements

- Python &ge;3.6

| Dependency       | Minimum Version | Purpose                           |
|------------------|-----------------|-----------------------------------|
| numpy            | 1.19.5          | required - usage                  |
| scipy            | 1.6             | required - usage                  |
| scikit-learn     | 1.2             | required - usage                  |
| tensorflow(-cpu) | 2.4.0           | required - GPU and/or CPU usage   |
| evm              | 0.1.2           | (optional) - additional unittests |
| pandas           | 1.0.5           | (optional) - example benchmark    |

### Setup

1. Set up a Python environment, e.g. with virtualenv or conda, and activate it
2. The `tf_ievm` package can be installed via pip, this will also install required dependencies, e.g. tensorflow.
   ```bash
   pip install git+https://github.com/e-solutions-GmbH/TensorFlow-iEVM.git
   ``` 
3. (Optional) Clone the repository and check the unit tests. This may take more than 10 minutes. Additional packages are
   required to run all unit tests, see [requirements.txt](requirements.txt), otherwise they will be skipped.
   ```bash
   git clone https://github.com/e-solutions-GmbH/TensorFlow-iEVM.git
   python -m unittest discover test/ -v
   ```
4. (Optional) Have a look at `examples/`

### Limitations

1. The implementation of the incremental EVM is not optimal and does not use all possible acceleration effects. Some of
   them are cumbersome to implement in TensorFlow and therefore may not result in acceleration at all.
2. When using a GPU, the load is relatively irregular, perhaps this can be eliminated/improved by
   using `tf.data.Dataset`. Execution on the GPU is not necessarily faster, as there is always a certain overhead.
3. No limitation, but a hint: Incremental learning is meant to train a model over time depending on a data stream. If
   all data is available from the beginning, a single fit is always faster than several small partial fits!

## Roadmap

- [x] make pip-installable
- [ ] improve compatibility with scikit-learn
- [ ] check whether replacing the custom batch processing with `tf.data.Dataset` leads to improved calculation
  performance

## Example Results

### Benchmark - Computation Time

See results [here](./examples/results/benchmark.md).

### Model Reduction - Toy Example

See results [here](./examples/results/model_reduction.md).

## References

[1]
T. Koch, F. Liebezeit, C. Riess, V. Christlein, and T. Köhler,
**“Exploring the Open World Using Incremental Extreme Value Machines”**,
*IEEE 26th International Conference on Pattern Recognition (ICPR)*, pp. 2792-2799, 2022.
[[Link](https://ieeexplore.ieee.org/abstract/document/9956423)]

```bibtex
@inproceedings{koch2022ievm,
    title = {Exploring the Open World Using Incremental Extreme Value Machines},
    author = {Koch, Tobias and Liebezeit, Felix and Riess, Christian and Christlein, Vincent and K{\"o}hler, Thomas},
    booktitle = {2022 26th International Conference on Pattern Recognition (ICPR)},
    pages = {2792--2799},
    year = {2022},
    organization = {IEEE}
}
```

[2]
E. M. Rudd, L. P. Jain, W. J. Scheirer, and T. E. Boult,
**“The Extreme Value Machine”**,
*IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)*, vol. 40, no. 3, pp. 762–768, 2017.
[[Link](https://ieeexplore.ieee.org/abstract/document/7932895)]

```bibtex
@article{rudd2017evm,
    title = {The Extreme Value Machine},
    author = {Rudd, Ethan M and Jain, Lalit P and Scheirer, Walter J and Boult, Terrance E},
    journal = {Transactions on Pattern Analysis and Machine Intelligence (PAMI)},
    volume = {40},
    number = {3},
    pages = {762--768},
    year = {2017},
    publisher = {IEEE}
}
```

[3]
J. Henrydoss, S. Cruz, C. Li, M. Günther, and T. E. Boult,
**“Enhancing Open-Set Recognition Using Clustering-Based Extreme Value Machine (C-EVM)”**,
*International Conference on Big Data (Big Data)*, pp. 441–448, 2020.
[[Link](https://ieeexplore.ieee.org/abstract/document/9378012)]

```bibtex
@inproceedings{henrydoss2020cevm,
    title = {Enhancing Open-Set Recognition Using Clustering-Based Extreme Value Machine (C-EVM)},
    author = {Henrydoss, James and Cruz, Steve and Li, Chunchun and G{\"u}nther, Manuel and Boult, Terrance E},
    booktitle = {International Conference on Big Data (Big Data)},
    pages = {441--448},
    year = {2020},
    organization = {IEEE}
}
```

## Related Projects

1. [[Link](https://bitbucket.org/vastlab/evm)] - Original Python implementation of the vanilla EVM