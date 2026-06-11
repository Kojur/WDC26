import numpy as np
from src.evaluate import result_outcome, rps, log_loss, evaluate


def test_result_outcome():
    assert result_outcome(2, 1) == 0
    assert result_outcome(1, 1) == 1
    assert result_outcome(0, 2) == 2


def test_rps_perfect_is_zero():
    assert rps([1.0, 0.0, 0.0], 0) == 0.0


def test_rps_known_value():
    # forecast [0.5,0.3,0.2], outcome=home(0): cum=[.5,.8], obs cum=[1,1]
    # ((.5-1)^2 + (.8-1)^2)/2 = (.25 + .04)/2 = 0.145
    np.testing.assert_allclose(rps([0.5, 0.3, 0.2], 0), 0.145)


def test_log_loss_decreases_with_confidence():
    assert log_loss([0.7, 0.2, 0.1], 0) < log_loss([0.4, 0.3, 0.3], 0)


def test_evaluate_aggregates():
    preds = [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]]
    outs = [0, 2]
    out = evaluate(preds, outs)
    assert set(out) == {"rps", "log_loss", "accuracy"}
    assert out["accuracy"] == 1.0
