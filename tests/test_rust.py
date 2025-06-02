import numpy as np

from mystique.utils import DTWDistance
from . import constants
from mystique import rust


def test_np_linalg_norm():
    np_result = np.linalg.norm(
        constants.cost_w["treatment_vector"]
        - constants.cost_w["cntrl_feat_mtx"][:, 0].reshape(-1, 1)
    )
    rust_result = rust.calc_norm_l2(
        constants.cost_w["treatment_vector"]
        - constants.cost_w["cntrl_feat_mtx"][:, 0].reshape(-1, 1)
    )
    assert round(np_result, 12) == round(rust_result, 12)


def test_cost_w():
    result = rust.cost_w(
        constants.cost_w["w"],
        constants.cost_w["treatment_vector"],
        constants.cost_w["cntrl_feat_mtx"],
        constants.cost_w["lam"],
    )
    assert round(result, 12) == round(0.006557374673869128, 12)


def test_norm_x_over_v():
    result = rust.norm_x_over_v(
        constants.norm_x_over_v["x"], constants.norm_x_over_v["v"]
    )

    assert round(result, 12) == round(0.04259768538873669, 12)


def test_dtwdistance():
    dtw_result = DTWDistance(constants.dtw_distance["s1"], constants.dtw_distance["s2"])
    rust_result = rust.dtw_distance(
        constants.dtw_distance["s1"], constants.dtw_distance["s2"]
    )

    assert round(dtw_result, 12) == round(rust_result, 12)
