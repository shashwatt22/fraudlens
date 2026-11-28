import numpy as np
import pytest

from src.v3.behavior import FEATURES
from src.v3.torch_rich_features import RichNeuralFeatures, input_inventory, transform_block, validate_features


def test_keeps_values_and_only_gradient_varying_missingness():
    train = np.array([[1, 0, np.nan, 6], [3, 0, np.nan, np.nan], [5, 1, np.nan, 10]], dtype=np.float32)
    prep = RichNeuralFeatures([1]).fit(train)
    np.testing.assert_array_equal(prep.selected, [0, 1, 2, 3, 7])
    np.testing.assert_allclose(prep.imputer.statistics_, [3, np.log(3), 0, 8], atol=1e-6)
    probe = np.array([[np.nan, 999, 5, np.nan]], dtype=np.float32)
    np.testing.assert_array_equal(prep._transform_raw(probe)[0, [0, 1, 2, 7]], [3, 0, 5, 1])
    assert transform_block(prep, train).dtype == np.float32
    np.testing.assert_array_equal(transform_block(prep, train), prep.transform(train))


def test_inventory_keeps_value_and_flag_identities_distinct():
    from types import SimpleNamespace

    old = SimpleNamespace(selected=np.array([1, 3]))
    new = SimpleNamespace(selected=np.array([0, 1, 2]))
    inventory = input_inventory(["amount", "card"], [1], old, new)
    assert inventory["control_omitted_values"] == ["amount"]
    assert inventory["control_missing_indicators"] == ["card"]
    assert inventory["challenger_frequency_values"] == ["card"]
    assert inventory["challenger_missing_indicators"] == ["amount"]


@pytest.mark.parametrize("field", ["TransactionID", "TransactionDT", "isFraud", "entity_key", FEATURES[0]])
def test_rejects_prohibited_columns(field):
    with pytest.raises(ValueError):
        validate_features(["TransactionAmt", field])
