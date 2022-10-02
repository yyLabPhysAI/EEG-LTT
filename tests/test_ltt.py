from abc import ABC, abstractmethod

import pytest

import numpy as np

from eeg_ltt.ltt import (
    BonferroniCorrection,
    CLTMajorityVoteRCP,
    CLTPoolingMajorityVoteRCP,
    RiskControllingPrediction,
    accuracy,
    false_alarm_rate,
)


class BaseTestRCP(ABC):
    @abstractmethod
    def rcp(self) -> RiskControllingPrediction:
        pass

    @pytest.fixture
    def data(self):
        locations = [np.random.rand(1000) < 0.01 for _ in range(1000)]
        gt = [np.convolve(loc, np.ones(8), mode="same") > 0 for loc in locations]
        y_pred = [
            g * (np.random.rand(1000) > 0.1) + (np.random.rand(1000) < 0.1) for g in gt
        ]
        return gt, y_pred

    def test_calibrate(self, data):
        gt, y_pred = data
        rcp = self.rcp()
        rcp.calibrate(gt, y_pred, lam_list=[0.1, 0.8, 1.0])
        assert rcp.lam in [0.8, 1.0]

    def test_predict(self, data):
        gt, y_pred = data
        rcp = self.rcp()

        with pytest.raises(RuntimeError):
            rcp.predict(y_pred[0])

        rcp.calibrate(
            gt,
            y_pred,
            lam_list=[0.8, 1.0],
        )
        y_corr = rcp.predict(y_pred[0])
        assert y_pred[0].shape == y_corr.shape

    def test_evaluate_metric(self, data):
        gt, y_pred = data
        rcp = self.rcp()
        rcp.calibrate(gt, y_pred, lam_list=[0.6])
        assert (
            accuracy(gt[0], rcp._T(y_pred[0]))
            == rcp.evaluate_metric(gt, y_pred, accuracy)[0]
        )

    def test_thm(self, data):
        gt, y_pred = data
        rcp = self.rcp()

        train_test_split = 500
        gt_train, gt_test = gt[:train_test_split], gt[train_test_split:]
        y_pred_train, y_pred_test = y_pred[:train_test_split], y_pred[train_test_split:]

        rcp.calibrate(gt_train, y_pred_train, lam_list=[0.1, 0.8, 1.0])

        assert (
            np.mean(
                np.array(rcp.evaluate_metric(gt_test, y_pred_test, false_alarm_rate))
            )
            < rcp.alpha
        )

        rcp._lam = 0.1
        assert (
            np.mean(
                np.array(rcp.evaluate_metric(gt_test, y_pred_test, false_alarm_rate))
            )
            > rcp.alpha
        )


class TestCLTMajorityVoteRCP(BaseTestRCP):
    def rcp(self) -> RiskControllingPrediction:
        return CLTMajorityVoteRCP(
            alpha=0.015,
            controlled_metric=false_alarm_rate,
            optimized_metric=accuracy,
            mhc=BonferroniCorrection(),
            window_length=10,
        )


class TestCLTPoolingMajorityVoteRCP(BaseTestRCP):
    def rcp(self) -> CLTPoolingMajorityVoteRCP:
        return CLTPoolingMajorityVoteRCP(
            alpha=0.15,
            controlled_metric=false_alarm_rate,
            optimized_metric=accuracy,
            mhc=BonferroniCorrection(),
            window_length=10,
            pooling_block=10,
        )

    def test_evaluate_metric(self, data):
        gt, y_pred = data
        rcp = self.rcp()
        rcp.calibrate(gt, y_pred, lam_list=[0.6])
        assert (
            accuracy(rcp.pool(gt[0]), rcp._T(y_pred[0]))
            == rcp.evaluate_metric(gt, y_pred, accuracy)[0]
        )

    def test_predict(self, data):
        gt, y_pred = data
        rcp = self.rcp()

        with pytest.raises(RuntimeError):
            rcp.predict(y_pred[0])

        rcp.calibrate(
            gt,
            y_pred,
            lam_list=[0.8, 1.0],
        )
        y_corr = rcp.predict(y_pred[0])
        assert rcp.pool(y_pred[0]).shape == y_corr.shape
