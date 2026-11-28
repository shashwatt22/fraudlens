import numpy as np

from src.v3.behavior import FEATURES, WEEK, anchor_keys, behavioral_block


def test_uses_exact_seven_day_window_and_unlimited_prior_offset_history():
    times = [0, 1, 2, WEEK, WEEK + 1, WEEK + 2]
    keys = [1] * len(times)
    values = np.column_stack(([100, 2, 6, 10, 14, 20], [2, 4, 6, 8, 10, 12],
                              [1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6]))
    output = behavioral_block(times, keys, values)
    assert output.dtype == np.float32 and output.shape == (6, 13) and len(set(FEATURES)) == 13
    np.testing.assert_allclose(output[4, :4], [3., 6., np.std([2., 6., 10.]), 8.])
    np.testing.assert_allclose(output[5, :4], [3., 10., np.std([6., 10., 14.]), 10.])
    np.testing.assert_allclose(output[5, 4:], [6., np.std([2, 4, 6, 8, 10]), 6.,
                                              3., np.std([1, 2, 3, 4, 5]), 3.,
                                              -3., np.std([-1, -2, -3, -4, -5]), -3.])


def test_equal_timestamps_share_prior_statistics_and_keep_row_local_deviations():
    values = np.array([[6, 8, 10, 12], [40, 30, 20, 10], [2, 4, 6, 8], [14, 16, 18, 20]])
    result = behavioral_block([20, 30, 10, 20], [0, 0, 0, 0], values)
    np.testing.assert_array_equal(result[0, [0, 1, 2, 4, 5, 7, 8, 10, 11]],
                                  result[3, [0, 1, 2, 4, 5, 7, 8, 10, 11]])
    np.testing.assert_array_equal(result[[0, 3], 3], [4, 12])
    np.testing.assert_array_equal(result[[0, 3], 6], [4, 12])
    np.testing.assert_allclose(result[1, :4], [3, 22/3, np.std([6, 2, 14]), 40-22/3])


def test_future_values_and_rows_do_not_change_existing_prefix():
    times = np.array([1, 2, 2, 3, 4, 5, 6])
    keys = np.array([0, 1, 1, 0, 1, 0, 2])
    measures = np.arange(28).reshape(7, 4)
    prefix = behavioral_block(times[:4], keys[:4], measures[:4])
    np.testing.assert_array_equal(behavioral_block(times, keys, measures)[:4], prefix)
    measures[4:] *= 1_000_000
    np.testing.assert_array_equal(behavioral_block(times, keys, measures)[:4], prefix)


def test_missing_key_cold_start_and_nonfinite_values_are_separate():
    times = [1, 2, 3, 4, 5]
    keys = [-1, 0, 0, 0, 0]
    values = [[50, 50, 50, 50], [np.nan, 3, np.inf, 7], [6, np.nan, 4, 9],
              [np.nan, 5, 2, 11], [10, 7, 8, 13]]
    result = behavioral_block(times, keys, values)
    assert np.isnan(result[0]).all()
    assert result[1, 0] == 0 and np.isnan(result[1, 1:]).all()
    np.testing.assert_array_equal(result[2, :7], [1, np.nan, np.nan, np.nan, 3, 0, np.nan])
    np.testing.assert_array_equal(result[3, :4], [2, 6, 0, np.nan])
    np.testing.assert_array_equal(result[4, :4], [3, 6, 0, 4])
    np.testing.assert_array_equal(result[3, 7:10], [4, 0, -2])


def test_anchor_key_uses_rounded_d1_offsets_and_requires_all_fields():
    matrix = np.array([[20, 5, 99.7], [20, 5, 100.1], [20, 5, 101.1], [20, np.nan, 100], [20, 6, 100]])
    assert anchor_keys(matrix, ["card1", "addr1", "day_minus_D1"]).tolist() == [0, 0, 1, -1, 2]
