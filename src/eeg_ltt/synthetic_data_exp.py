import argparse
import itertools as it
import logging
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_ltt.ltt import (
    BonferroniCorrection,
    CLTMajorityVoteRCP,
    CLTPoolingMajorityVoteRCP,
    FixedSequenceControl,
    accuracy,
    false_alarm_rate,
)


def synthetic_data(
    num_samples: int = 10000,
    signal_length: int = 1000,
    window_length: int = 8,
    error_rate: float = 0.1,
):
    """
    Create synthetic data for the experiment.

    Create arrays that represent the ground truth labeling and a noisy oracle
    predictions for an event prediction task of temporal signals. The prediction labels
    represent a temporal interval on which an alarm is appropriate with binary labels,
    e.g.:
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,]
    Is a 20 sample long signal with a pre-event prediction window length of 4.

    Args:
        num_samples: Number of samples to create
        signal_length: Signal lengths in samples
        window_length: The pre-event prediction window length
        error_rate: Error rate of the noisy oracle

    Returns:
        Two corresponding sequences of arrays, the ground truth labels and the
        predictions of the noisy oracle.

    """

    locations = [np.random.rand(signal_length) < 0.01 for _ in range(num_samples)]
    gt = [
        np.convolve(loc, np.ones(window_length), mode="same") > 0 for loc in locations
    ]
    y_pred = [
        g * (np.random.rand(signal_length) > error_rate)
        + (np.random.rand(signal_length) < error_rate)
        for g in gt
    ]
    return gt, y_pred


def test_ltt(alpha, num_samples, pooling, corrector):
    """
    Preforms a single LTT synthetic data experiment.

    Args:
        alpha: The LTT alpha parameter: the bound for the controlled risk. Specifically,
        the allowed false alarm rate.
        num_samples: Number of samples to create for each experiment.
        pooling: Whether to pool the predictions during post-processing
        corrector: Multiple hypothesis corrector to use.

    Returns: a tuple of floats: (False Alarm Rate, Accuracy, Lambda, Alpha)

    """

    if corrector == "bon":
        mhc = BonferroniCorrection()
    elif corrector == "fsc":
        mhc = FixedSequenceControl()
    else:
        raise ValueError(f"Unknown MHC procedure {corrector}. Available are:bon, fsc")

    if pooling:
        rcp = CLTPoolingMajorityVoteRCP(
            alpha=alpha,
            controlled_metric=false_alarm_rate,
            optimized_metric=accuracy,
            mhc=mhc,
            window_length=10,
            pooling_block=10,
        )
    else:
        rcp = CLTMajorityVoteRCP(
            alpha=alpha,
            controlled_metric=false_alarm_rate,
            optimized_metric=accuracy,
            mhc=mhc,
            window_length=10,
        )

    gt, y_pred = synthetic_data(
        num_samples,
    )

    train_test_split = num_samples // 2
    gt_train, gt_test = gt[:train_test_split], gt[train_test_split:]
    y_pred_train, y_pred_test = y_pred[:train_test_split], y_pred[train_test_split:]

    try:

        rcp.calibrate(gt_train, y_pred_train, lam_list=[0.01 * i for i in range(100)])
        far = np.mean(
            np.array(rcp.evaluate_metric(gt_test, y_pred_test, false_alarm_rate))
        )
        acc = np.mean(np.array(rcp.evaluate_metric(gt_test, y_pred_test, accuracy)))

        if not far < rcp.alpha:
            raise RuntimeError(
                "LTT procedure didn't control the risk, it is likely to "
                "represent an implementation bug or a too small number "
                "of repetitions. "
            )

        far, acc, lam, alpha = far, acc, rcp._lam, alpha

    except ValueError:

        logging.warning("On, no, couldn't find any bounding lambda. retrying...")
        far, acc, lam, alpha = test_ltt(alpha, num_samples, pooling, corrector)

    return far, acc, lam, alpha


def main():
    # argparsing
    parser = argparse.ArgumentParser(
        description="Run the time series LTT experiment with synthetic data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("Name", metavar="name", type=str, help="Experiment name to use")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to create for the experiment",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=200,
        help="Number of repetitions " "of each experiment",
    )
    parser.add_argument(
        "--corrector",
        type=str,
        default="bon",
        help="Multiple Hypothesis Corrector to use, avalable are: bon "
             "(Bonferroni correction) and fsc (fixed sequence correction)",
    )
    parser.add_argument(
        "-p",
        "--no_pooling",
        action="store_true",
        help="Disable pooling in the temporal aggregation of the time series LTT",
    )
    parser.add_argument(
        "-n",
        "--no_parallel",
        action="store_true",
        help="Disable multiprocessing in the experiment run",
    )

    args = parser.parse_args()

    alpha_list = [1, 0.1, 0.05, 0.001, 0.0005]

    res = []
    n_jobs = None if args.no_parallel else -1
    pool = Parallel(n_jobs=n_jobs, verbose=100)
    for a in alpha_list:
        print(f"Calculating for alpha={a}, reps={args.reps}")
        res.append(
            pool(
                delayed(test_ltt)(
                    a, args.num_samples, not args.no_pooling, args.corrector
                )
                for _ in range(args.reps)
            )
        )

    df = pd.DataFrame(
        columns=["False Alarm Rate", "Accuracy", "Lambda", "Alpha"],
        data=list(it.chain(*res)),
    )
    df.to_csv(f"df_{args.Name}.csv")
    for metric in ["mean", "std", "max", "min"]:
        df.groupby("Alpha").agg(metric).to_csv(f"{metric}s_{args.Name}.csv")


if __name__ == "__main__":
    sys.exit(main())
