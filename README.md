# Time-Series Learn Then Test

A time-series implementation of the [Learn Then Test] (LTT) risk-controlling prediction framework. <br>

### Learn Then Test
The [Learn Then Test] method is a post-processing procedure applied to model predictions in order to control a risk, for example 
false alarm rate. The prediction model is treated as a black box. The [Learn Then Test] calibration process allows finite sample guarantees on the expectancy of the chosen risk. <br>

The LTT method divides the risk-control problem into two well-known problems in statistics – computing p-values and multiple testing correction, both adjusted to time-series and implemented in this repository.

Risk controlling prediction applies post-processing to model predictions in order to control a risk. The predictions are transformed by a function $g_\lambda$ that is dependent on a parameter $\lambda$ , which is calibrated using a calibration set of samples. For example, if we have a model telling dog and cat images apart by assigning probabilities to each (e.g. the image is 70% likely to be a “dog” and 30% to be a “cat”) and we want to control the false prediction of the “dog” label, the post-processing can be a function that favors the “cat” label, e.g., choose “dog” only if the “dog” probability is more than λ=60% . If the threshold $\lambda$ is calibrated using the proposed process, we can get finite sample guarantees on the expectancy of the risk. In our example, we can demand a false detection rate of “dog” to be less than 5%. This is done by using calibration to get a $\lambda$ value that will guarantee this condition when the results are averaged over many experiments.

### Time-Series Adaptation
In order to adapt the LTT framework to time-series data, we had to find a $g_\lambda$ function that relates to the sequential nature of the input. The function includes two steps of temporal aggregation:
  1. A window majority vote is done over the prediction $Y[t]$ with windows of length $w$ so only windows with more than $\lambda$ fraction of positives are considered positive: <br>
  $$Y_{MV}[t] = {𝟙}\left[\frac{Y[t - w +1] + Y[t - w +2] + ... + Y[t]}{w} > \lambda\right]$$
  2. The time series is max-pooling action over the lower temporal resolution with k pooling rate: <br>
  $$Y_{pooled}[n] = max\left(Y_{MV}[(n-1)w_2], Y_{MV}[(n-1)w_2 +1], ... ,    Y_{MV}[nw_2]\right)$$

With the majority vote indented to add robustness against noise in the model predictions and the pooling to mitigate the effect of point mistakes.
When fixing $w$, $k$ and calibrating only $\lambda$, this post-processing satisfies the LTT theorem conditions.

### Synthetic Data Experiment
The time-series adaptation of the LTT method to a time-series case validation is demonstrated in a synthetic data experiment. The synthetic data includes signals with fixed length windows of ones randomly spread and zeros elsewhere, representing periods of events that occur occasionally. The synthetic model predictions are equal to the ground truth anywhere but certain points for which the label was flipped according to a determined distribution. The accuracy, false alarm rate and the chosen $\lambda$, with and without pooling, can be tested for different values of $\alpha$.

### In this repository, you will find:
* Time-series adaptation for the [Learn Then Test] framework. This includes:
  * Multiple hypothesis correction- implementation of the Bonferroni correction and [Fixed Sequence Control]
  * P-values calculation- implementation of the Central Limit Therorm
  * Temporal aggregation- implementation of Majority Vote and Max-Pooling
* Synthetic data platform for experiments

For further reading, see our publication: # TODO 

### Citations: 
* Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., and Lei, L. (2022a). Learn then test: Calibrating predictive algorithms to achieve risk control. arxiv.org. Available at: https://arxiv.org/abs/2110.01052 [Accessed October 3, 2022].).
* Angelopoulos, A. N., and Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. doi: 10.48550/arxiv.2107.07511.

[Learn Then Test]: https://arxiv.org/abs/2110.01052
[Fixed Sequence Control]: https://arxiv.org/abs/2107.07511
