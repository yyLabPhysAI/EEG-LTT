from abc import ABC, abstractmethod
from typing import Optional, Protocol, Sequence

import numpy as np
import scipy.stats

EPS = 1e-8


class Metric(Protocol):
    """
    A protocol defining the metric API used by the LTT calibration
    """

    @abstractmethod
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        pass


def false_alarm_rate(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    A concrete implementation of false alarm rate

    Args:
        y: Ground troth label
        y_pred: Model prediction

    Returns: The false alarm rate as a fraction of the samples

    """
    return np.sum(y_pred[y == 0]) / len(y)


def accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    A concrete implementation of an accuracy metric

    Args:
        y: Ground troth label
        y_pred: Model prediction

    Returns: The fraction of model predictions matching the label
    """
    return np.mean(y == y_pred)


class MultipleHypothesisCorrector:
    """
    An API for Family-Wise Error Rate (FWER) multiple hypothesis correction method
    """

    def __init__(self, fwer: float = 0.05):
        """
        Args:
            fwer: The required Family-Wise Error Rate (FWER), sometimes denoted with an
            alpha
        """
        self.fwer = fwer

    @abstractmethod
    def __call__(self, p_values: Sequence[float]) -> Sequence[bool]:
        pass


class BonferroniCorrection(MultipleHypothesisCorrector):
    """
    The Bonferroni correction, for a set of m hypotheses tested, rejects the null for:
    p < alpha / m

    Bounding the expected FWER under alpha.

    Notes:
        1. The Bonferroni correction is a "strict" corrector, sometimes causing low
        statistical power.
        2. The Bonferroni correction doesn't require any condition on the dependency
        between hypotheses.

    """

    def __call__(self, p_values: Sequence[float]) -> Sequence[bool]:
        threshold = self.fwer / len(p_values)
        return tuple(p < threshold for p in p_values)


class FixedSequenceControl(MultipleHypothesisCorrector):
    """
    Fixed Sequence Control correction. For a sorted sequence of p-values, returns all
    rejects null for:
        p < alpha

    Notes:
        1. The Fixed Sequence Control correction require monotonous p-values to be
        valid.
    """

    def __call__(self, p_values: Sequence[float]) -> Sequence[bool]:

        p_vec = np.array(p_values)

        if not np.all(np.diff(p_vec) > -1e9):
            raise ValueError(
                "Non sorted p-values given to a Fixed Sequence Control "
                "MHC,this violates the THM conditions and not allowed."
            )

        return tuple(p < self.fwer for p in p_values)


class RiskControllingPrediction(ABC):
    """
    An API for a Risk Controlling Prediction (RCP) is described in:

    Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control
    Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Cand`es, Michael I. Jordan,
    Lihua Lei

    Uses calibration of prediction post-processing to
    """

    def __init__(
        self,
        alpha: float,
        controlled_metric: Metric,
        optimized_metric: Metric,
        mhc: MultipleHypothesisCorrector,
    ):
        """
        Args:
            alpha: A boundary to calibrate the RCP to bound the risk to
            controlled_metric: The controlled metric
            optimized_metric: A second metric to optimize, used to choose the lambda
            between the values that passed the calibration
            mhc:
        """
        self._alpha = alpha
        self._delta = mhc.fwer
        self._lam: Optional[float] = None
        self._controlled_metric = controlled_metric
        self._optimized_metric = optimized_metric
        self._mhc = mhc

    def calibrate(
        self,
        calibration_labels: Sequence[np.ndarray],
        calibration_predictions: Sequence[np.ndarray],
        lam_list: Sequence[float],
        calibration_optimization_labels: Optional[Sequence[np.ndarray]] = None,
    ):
        """
        Calibrates the LTT lambda value

        Args:
            calibration_labels: Calibration set ground truth label, used with the
            controlled metric to find the controlling lambda values subset
            calibration_predictions: Model predictions for the calibration set
            lam_list: A list of lambda values to try
            calibration_optimization_labels: An optional additional label to use in the
            optimized metric. If not provided, the calibration_labels are used for both.
        """
        p_list = []
        for lam in lam_list:
            self._lam = lam
            p = self._calculate_p_value(calibration_labels, calibration_predictions)
            p_list.append(p)

        significant_lam = np.array(lam_list)[np.array(self._mhc(p_list))]
        optimized_metrics = []
        for lam in significant_lam:
            self._lam = lam
            m = self.evaluate_metric(
                calibration_labels,
                calibration_optimization_labels
                if calibration_optimization_labels
                else calibration_predictions,
                self._optimized_metric,
            )
            optimized_metrics.append(np.mean(m))

        self._lam = significant_lam[np.argmax(np.array(optimized_metrics))]

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        Runs the calibrated rist controlling prediction

        Args:
            predictions: Model predictions to adjust with the RCP

        Returns:

        """
        if self._lam is None:
            raise RuntimeError(
                f"{self.__class__.__name__} cannot control the risk "
                f"without being calibrated first, call `calibrate()`"
                f"before trying to `predict()`"
            )

        return self._T(predictions)

    def evaluate_metric(
        self,
        labels: Sequence[np.ndarray],
        predictions: Sequence[np.ndarray],
        metric: Metric,
    ) -> Sequence[float]:
        """
        Evaluates the given metric for a set of predictions and corresponding labels.

        Args:
            labels: Ground truth labels
            predictions: Model predictions
            metric: A metric to evaluate

        Returns: The metric per sample

        """
        return tuple(
            metric(label, self._T(prediction))
            for label, prediction in zip(labels, predictions)
        )

    @abstractmethod
    def _calculate_p_value(
        self,
        calibration_labels: Sequence[np.ndarray],
        calibration_predictions: Sequence[np.ndarray],
    ) -> float:
        pass

    @abstractmethod
    def _T(self, y_pred: np.ndarray) -> np.ndarray:
        pass

    @property
    def alpha(self):
        """
        A boundary to calibrate the RCP to bound the risk to
        """
        return self._alpha

    @property
    def delta(self):
        """
        The level of certainty on the RCP metric control
        """
        return self._delta

    @property
    def lam(self):
        """
        Chosen lambda metric
        """
        return self._lam

    @property
    def controlled_metric_name(self):
        return self._controlled_metric.__name__

    @property
    def optimized_metric_name(self):
        return self._optimized_metric.__name__

    @property
    def mhc_name(self):
        return self._mhc.__class__.__name__


class CLTRiskControllingPrediction(RiskControllingPrediction, ABC):
    def _calculate_p_value(
        self,
        calibration_labels: Sequence[np.ndarray],
        calibration_predictions: Sequence[np.ndarray],
    ) -> float:
        m_vec = np.array(
            self.evaluate_metric(
                calibration_labels, calibration_predictions, self._controlled_metric
            )
        )
        return 1 - scipy.stats.norm.cdf(
            (self.alpha - np.mean(m_vec)) / (np.std(m_vec) + EPS)
        )


class MajorityVoteRiskControllingPrediction(RiskControllingPrediction, ABC):
    def __init__(
        self,
        alpha: float,
        controlled_metric: Metric,
        optimized_metric: Metric,
        mhc: MultipleHypothesisCorrector,
        window_length: int = 10,
    ):
        self._window_length = window_length
        super().__init__(alpha, controlled_metric, optimized_metric, mhc)

    def _T(self, y_pred: np.ndarray) -> np.ndarray:
        count_vec = np.convolve(y_pred, np.ones(self._window_length), mode="same")
        return count_vec >= self._window_length * self.lam


class PoolingMajorityVoteRiskControllingPrediction(
    MajorityVoteRiskControllingPrediction, ABC
):
    def __init__(
        self,
        alpha: float,
        controlled_metric: Metric,
        optimized_metric: Metric,
        mhc: MultipleHypothesisCorrector,
        window_length: int = 10,
        pooling_block: int = 1,
    ):
        self.pooling_block = pooling_block
        super().__init__(alpha, controlled_metric, optimized_metric, mhc, window_length)

    def evaluate_metric(
        self,
        labels: Sequence[np.ndarray],
        predictions: Sequence[np.ndarray],
        metric: Metric,
    ) -> Sequence[float]:
        pooled_labels = [self.pool(y) for y in labels]
        return super().evaluate_metric(pooled_labels, predictions, metric)

    def pool(self, y: np.ndarray):
        n = len(y)
        return y[n % self.pooling_block :].reshape(n // self.pooling_block, -1).max(1)

    def _T(self, y_pred: np.ndarray) -> np.ndarray:
        calibrated_y = super()._T(y_pred)
        return self.pool(calibrated_y)


class CLTMajorityVoteRCP(
    CLTRiskControllingPrediction, MajorityVoteRiskControllingPrediction
):
    pass


class CLTPoolingMajorityVoteRCP(
    CLTRiskControllingPrediction, PoolingMajorityVoteRiskControllingPrediction
):
    pass
