import os
import shutil
import numpy as np
from pmemo import memo, memo_cache_get, memo_cache_set

def test_memo():
    MEMO_DIR = "./.cache-memo"

    if os.path.isdir(MEMO_DIR):
        print("Cleaning", MEMO_DIR)
        shutil.rmtree(MEMO_DIR)

    @memo(debug=True)
    def my_fn(a1, b):
        return a1 + b

    @memo(debug=True, use_torch_serdes=False)
    def my_fn2(a2, b):
        return a2 + b

    @memo(debug=True)
    def t_sum(v):
        return np.sum(v)

    @memo(debug=True, use_torch_serdes=False)
    def t_sum_t(v):
        return np.sum(v)

    v = np.random.randn(100)

    md = "123123mdsfgsdfg"

    for i in range(2):
        assert my_fn(1, 2, _memo_meta=md) == 3
        assert my_fn2(1, 2, _memo_meta=md) == 3
        assert float(t_sum(v, _memo_meta=md)) == float(np.sum(v))
        assert float(t_sum_t(v, _memo_meta=md)) == float(np.sum(v))


MEMO_DIR = "./.cache-test-memo"


def setup_module():
    if os.path.isdir(MEMO_DIR):
        shutil.rmtree(MEMO_DIR)


def test_memo_cache_get_set():
    key_dict = {"fn": "test", "args": [1, 2], "kwargs": {"a": 1}}
    value1 = np.array([1, 2, 3])
    tag = "my_tag"
    memo_cache_set(key_dict, value1, tag, MEMO_DIR, use_torch_serdes=False)
    assert np.array_equal(
        memo_cache_get(key_dict, tag, MEMO_DIR, use_torch_serdes=False), value1
    )

    # Update the cache value
    value2 = np.array([4, 5, 6])
    memo_cache_set(key_dict, value2, tag, MEMO_DIR, use_torch_serdes=False)
    assert np.array_equal(
        memo_cache_get(key_dict, tag, MEMO_DIR, use_torch_serdes=False), value2
    )


# def test_memo_cache_get_set_torch():
#     key_dict = {"fn": "test", "args": [1, 2], "kwargs": {"a": 1}}
#     value = np.array([1, 2, 3])
#     tag = "my_tag"
#     memo_cache_set(key_dict, value, tag, MEMO_DIR, use_torch_serdes=True)
#     assert np.array_equal(memo_cache_get(key_dict, tag, MEMO_DIR, use_torch_serdes=True), value)

# def test_memo_decorator_compat():
#     @memo(debug=True, memo_dir=MEMO_DIR)
#     def my_fn(a, b):
#         return a + b

#     key_dict = {"fn": "my_fn", "args": [1, 2], "kwargs": {}}
#     value = my_fn(1, 2)
#     assert memo_cache_get(key_dict, None, MEMO_DIR, use_torch_serdes=False) == value

# def test_memo_decorator_compat_torch():
#     @memo(debug=True, memo_dir=MEMO_DIR, use_torch_serdes=True)
#     def my_fn(a, b):
#         return a + b

#     key_dict = {"fn": "my_fn", "args": [1, 2], "kwargs": {}}
#     value = my_fn(1, 2)
#     assert np.array_equal(memo_cache_get(key_dict, None, MEMO_DIR, use_torch_serdes=True), value)


def test_non_existent_key():
    key_dict = {"fn": "non_existent", "args": [1, 2], "kwargs": {}}
    assert memo_cache_get(key_dict, None, MEMO_DIR, use_torch_serdes=False) is None
    assert (
        memo_cache_get(key_dict, None, MEMO_DIR, use_torch_serdes=False, fallback=1)
        is 1
    )


def teardown_module():
    if os.path.isdir(MEMO_DIR):
        shutil.rmtree(MEMO_DIR)
