# Stochastic Variance Reduction Gradient Descent (SVRG) optimizer for Keras


## Reference
https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf


## Test results on MNIST
The following figure shows the train loss.
![MNIST](MNIST.png)
__________________________________
The results on test set:
| Metric  | SVRG       | SGD        |
| --------| --------- | ---------- |
| Loss    | 0.0893619 | 0.05765046 |
| Accuracy| 0.9722    | 0.9817     |

## Usage
To install the package use the following command:
> pip install svrg-optimizer-keras

In the code you can use it as a standard Keras optimizer

> from SVRGoptimizerKeras.optimizer import SVRG
>
> ... 
>
> optimizer = SVRG(lr=0.001)
> model.compile(loss=...,optimizer=optimizer)

An example is test/MNIST.py file.

### Parameters
* lr: learning rate
* decay: learning rate decay
* mean_calculation_step: number of steps for statistics collection
* update_step: number of steps for actual updating and using the collected statistics. After the predefined step number the statistics collecting starts again.

