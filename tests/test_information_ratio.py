import statistics

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

import ffn


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64", object])
@pytest.mark.parametrize("duplicate_columns", [False, True])
def test_information_ratio_preserves_columnwise_results(dtype, duplicate_columns):
    index = pd.date_range("2024-01-01", periods=5)
    returns = pd.DataFrame(
        {
            "varying": [0.0, 0.2, -0.1, np.nan, 0.3],
            "constant": [0.1] * 5,
            "missing": [np.nan] * 5,
            "single": [np.nan, np.nan, 0.2, np.nan, np.nan],
            "zero": [0.0] * 5,
        },
        index=index,
        dtype=dtype,
    )
    if duplicate_columns:
        returns.columns = ["asset"] * len(returns.columns)
    returns.columns.name = "assets"
    original = returns.copy()
    benchmark = pd.Series(0.0, index=index, dtype=dtype)
    # pandas 1.x uses object-dtype reductions for nullable inputs.
    expected = pd.Series(
        [ffn.calc_information_ratio(returns.iloc[:, column], benchmark) for column in range(len(returns.columns))],
        index=returns.columns,
        dtype=object if returns.mean().dtype == object else float,
    )

    for actual in (ffn.calc_information_ratio(returns, benchmark), returns.calc_information_ratio(benchmark)):
        pd.testing.assert_series_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(returns, original)


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0)])
@pytest.mark.parametrize("dtype", ["float64", "Float64", object])
def test_information_ratio_preserves_empty_frames(shape, dtype):
    returns = pd.DataFrame(index=range(shape[0]), columns=range(shape[1]), dtype=dtype)

    result = ffn.calc_information_ratio(returns, returns)

    expected = pd.Series(0.0, index=returns.columns, dtype=object if returns.mean().dtype == object else float)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64", object])
def test_information_ratio_treats_rounding_residue_as_no_tracking_error(dtype):
    # (r + c) - r is c plus rounding residue that scales with r. Dividing by that
    # residue gave ratios near 1e15; it is the zero-tracking-error case.
    index = pd.date_range("2024-01-01", periods=250, freq="B")
    r = pd.Series(np.random.default_rng(0).normal(0.0, 0.02, 250), index=index, dtype=dtype)

    assert ffn.calc_information_ratio(r + 0.0005, r) == 0.0
    assert ffn.calc_information_ratio(pd.Series(0.001, index=index), pd.Series(0.0, index=index)) == 0.0
    assert ffn.calc_prob_mom(r + 0.0005, r) == 0.5

    frame = pd.DataFrame({"residue": r + 0.0005, "varying": r * 2})
    result = ffn.calc_information_ratio(frame, r)
    assert result["residue"] == 0.0
    assert np.isclose(result["varying"], r.mean() / r.std(ddof=1))


def test_information_ratio_keeps_a_small_real_tracking_error():
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    benchmark = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, -0.01], index=index)
    diff = pd.Series([1e-9, -1e-9, 2e-9, 0.0, 1e-9, 3e-9], index=index)

    expected = diff.mean() / diff.std(ddof=1)

    assert np.isclose(ffn.calc_information_ratio(benchmark + diff, benchmark), expected, rtol=1e-4)


@pytest.mark.parametrize("metric", ["calc_information_ratio", "calc_prob_mom"])
@pytest.mark.parametrize("shape", ["frame_series", "series_frame", "frame_frame"])
@pytest.mark.parametrize("duplicate_columns", [False, True])
def test_tracking_error_tolerance_is_columnwise(metric, shape, duplicate_columns):
    quiet = pd.Series([1e-17, 2e-17, 4e-17, 3e-17])
    frame = pd.DataFrame({"quiet": quiet, "large": quiet * 1e16})
    if duplicate_columns:
        frame.columns = ["asset", "asset"]
    frame.columns.name = "assets"
    benchmark = pd.Series(0.0, index=frame.index)
    function = getattr(ffn, metric)
    expected = pd.Series([function(frame.iloc[:, i], benchmark) for i in range(2)], index=frame.columns)
    if shape == "series_frame":
        result = function(benchmark, -frame)
    else:
        if shape == "frame_frame":
            benchmark = frame * 0
        result = function(frame, benchmark)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "metric,dtype",
    [("calc_information_ratio", "float64"), ("calc_information_ratio", "Float64"), ("calc_information_ratio", object), ("calc_prob_mom", "float64"), ("calc_prob_mom", "Float64")],
)
@pytest.mark.parametrize("missing", ["unaligned", "nan"])
@pytest.mark.parametrize("reverse", [False, True])
def test_tracking_error_tolerance_ignores_unpaired_observations(metric, dtype, missing, reverse):
    returns = pd.Series([1e-17, 2e-17, 4e-17, 3e-17], dtype=dtype)
    benchmark = pd.Series(0.0, index=returns.index, dtype=dtype)
    if reverse:
        returns, benchmark = benchmark, returns
    function = getattr(ffn, metric)
    expected = function(returns, benchmark)
    returns.loc[4] = 1.0
    if missing == "nan":
        benchmark.loc[4] = np.nan

    assert np.isclose(function(returns, benchmark), expected)
    frame = pd.DataFrame({"asset": returns})
    assert np.isclose(function(frame, benchmark).iloc[0], expected)


@pytest.mark.parametrize("metric", ["calc_information_ratio", "calc_prob_mom"])
def test_tracking_error_tolerance_preserves_each_columns_precision(metric):
    benchmark = pd.Series(np.random.default_rng(0).normal(0.0, 0.02, 250))
    quiet = pd.Series(np.resize([1e-10, 2e-10, 4e-10, 3e-10], len(benchmark)))
    returns = pd.DataFrame({"single": (benchmark + 0.0005).astype("float32"), "double": benchmark + quiet})
    benchmarks = pd.DataFrame({"single": benchmark.astype("float32"), "double": benchmark})
    function = getattr(ffn, metric)
    expected = pd.Series(
        [0.0 if metric == "calc_information_ratio" else 0.5, function(returns["double"], benchmarks["double"])],
        index=returns.columns,
    )

    pd.testing.assert_series_equal(function(returns, benchmarks), expected)


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"]
)
@pytest.mark.parametrize("shape", ["series_series", "frame_series", "series_frame", "frame_frame"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("metric", ["calc_information_ratio", "calc_prob_mom"])
def test_integer_subtraction_reaches_risk_statistics(dtype, shape, partial, metric):
    limits = np.iinfo(dtype.lower())
    index = pd.date_range("2024-01-01", periods=5 if partial else 4)
    values = [0, 1, 2, 1] if dtype.lower().startswith("u") else [limits.min, 0, limits.max, 1]
    returns = pd.Series(np.array(values, dtype=dtype.lower()), index=index[:4], dtype=dtype, name="asset")
    benchmark_index = index[[0, 1, 2, 4]] if partial else index
    benchmark = pd.Series(np.ones(4, dtype=dtype.lower()), index=benchmark_index, dtype=dtype, name="asset")
    if partial and dtype[0].isupper():
        returns.iloc[1] = pd.NA
    left, right = returns.to_dict(), benchmark.to_dict()
    # Python integers avoid overflow in the oracle; only observed date pairs determine the statistics and Student-t sample size.
    differences = [int(left[date]) - int(right[date]) for date in index if date in left and date in right and not pd.isna(left[date])]
    ratio = statistics.mean(differences) / statistics.stdev(differences)
    expected = ratio if metric == "calc_information_ratio" else t.cdf(ratio * np.sqrt(len(differences)), len(differences) - 1)
    first, second = shape.split("_")
    data = returns.to_frame() if first == "frame" else returns
    other = benchmark.to_frame() if second == "frame" else benchmark
    original, original_other = data.copy(), other.copy()
    for result in (getattr(ffn, metric)(data, other), getattr(data, metric)(other)):
        # Ratios retain ordinary floating precision; the public subtraction test checks exact integers.
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-15)
        if isinstance(result, pd.Series):
            pd.testing.assert_index_equal(result.index, (data if first == "frame" else other).columns)
    (pd.testing.assert_frame_equal if first == "frame" else pd.testing.assert_series_equal)(data, original)
    (pd.testing.assert_frame_equal if second == "frame" else pd.testing.assert_series_equal)(other, original_other)


@pytest.mark.parametrize("dtype", ["uint64", "UInt64"])
@pytest.mark.parametrize("shape", ["frame_series", "series_frame", "frame_frame"])
def test_integer_subtraction_converts_only_new_object_differentials_for_momentum(dtype, shape):
    maximum = 2**64 - 1
    returns = pd.Series(np.array([0, maximum, 2, maximum - 2], dtype="uint64"), dtype=dtype, name="asset")
    benchmark = pd.Series(np.full(4, maximum, dtype="uint64"), dtype=dtype, name="asset")
    differences = [-maximum, 0, 2 - maximum, -2]
    ratio = statistics.mean(differences) / statistics.stdev(differences)
    first, second = shape.split("_")
    data = returns.to_frame() if first == "frame" else returns
    other = benchmark.to_frame() if second == "frame" else benchmark
    # SciPy must receive numeric statistics even when exact subtraction needs object storage.
    np.testing.assert_allclose(data.calc_information_ratio(other), ratio)
    result = data.calc_prob_mom(other)
    np.testing.assert_allclose(result, t.cdf(ratio * 2, 3))
    assert result.dtype.kind == "f"


@pytest.mark.parametrize("metric", ["calc_information_ratio", "calc_prob_mom"])
def test_integer_subtraction_keeps_date_and_column_pairing(metric):
    index = pd.date_range("2024-01-01", periods=4)
    returns = pd.DataFrame({"alpha": [0, 3, 2, 1], "beta": [4, 0, 4, 3]}, index=index, dtype="UInt8")
    benchmark = pd.DataFrame({"alpha": [1, 2, 3, 1], "beta": [3, 1, 2, 4]}, index=index, dtype="UInt8")
    benchmark = benchmark.loc[:, ["beta", "alpha"]].iloc[::-1]
    # Distinct per-label values expose positional subtraction after either axis is reordered.
    differences = [[-1, 1, -1, 0], [1, -1, 2, -1]]
    ratios = [statistics.mean(values) / statistics.stdev(values) for values in differences]
    expected = ratios if metric == "calc_information_ratio" else t.cdf(np.asarray(ratios) * 2, 3)
    result = getattr(returns, metric)(benchmark)
    np.testing.assert_allclose(result, expected)
    pd.testing.assert_index_equal(result.index, returns.columns)
