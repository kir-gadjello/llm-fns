import hashlib
import inspect
import os
import shutil
import time
import base64

# from functools import reduce
from types import SimpleNamespace
import argparse

import numpy as np
import orjson
from cbor2 import CBORDecodeValueError, CBORTag, dump, dumps, load, loads


def orjson_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.numpy()

    raise TypeError


dict_items_sig = type({}.items())
dict_keys_sig = type({}.keys())
dict_values_sig = type({}.values())


def orjson_default_for_hash(obj):
    if hasattr(obj, "__hashkey"):
        return obj.__hashkey
    elif isinstance(obj, np.ndarray):
        return obj.numpy()
    elif isinstance(obj, SimpleNamespace) or isinstance(obj, argparse.Namespace):
        return vars(obj)
    elif hasattr(obj, "__iter__"):
        print("!!!", obj)
        return [*obj]


def universal_hash(obj, default=orjson_default_for_hash, b64=False):
    h = hashlib.sha256()
    h.update(
        orjson.dumps(
            obj,
            default=default,
            option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY,
        )
    )
    if b64:
        digest = h.digest()
        return base64.urlsafe_b64encode(digest).decode("ascii")
    else:
        return h.hexdigest()


def default_cbor_encoder(encoder, value):
    val = None

    # if isinstance(value, torch.Tensor):
    #    val = value.numpy()
    if isinstance(value, np.ndarray):
        val = value
    else:
        raise Exception("cannot decode value", value)

    encoder.encode(CBORTag(4001, [val.shape, val.dtype.char, val.tobytes()]))


def cbor_tensor_tag_hook(decoder, tag, shareable_index=None):
    if tag.tag != 4001:
        return tag

    if decoder.immutable:
        raise CBORDecodeValueError("MyType cannot be used as a key or set member")

    shape, dtype_char, buf = tag.value

    ret = np.frombuffer(bytearray(buf), dtype=dtype_char).reshape(shape)
    ret.setflags(write=1)
    return ret


def cbor2_dump_ext(data, stream):
    return dump(data, stream, default=default_cbor_encoder)


def cbor2_load_ext(stream):
    return load(stream, tag_hook=cbor_tensor_tag_hook)


def cbor2_dumps_ext(data):
    return dumps(data, default=default_cbor_encoder)


def cbor2_loads_ext(_repr):
    return loads(_repr, tag_hook=cbor_tensor_tag_hook)


def hash_strings(strings):
    h = hashlib.sha256()
    for string in strings:
        h.update(string.encode("utf-8"))
    return h.hexdigest()


DEFAULT_MEMO_DIR = "./.cache-memo"


def memo(
    name=None,
    memo_dir=DEFAULT_MEMO_DIR,
    use_torch_serdes=False,
    debug=False,
    torch=None,
    disable=False,
    tag=None,
):
    def wrapper(fn):
        nonlocal name, memo_dir, use_torch_serdes, debug

        src_name = False
        if name is None:
            name = inspect.getsource(fn)
            src_name = True

        fn_hash = hash_strings([name])

        if not os.path.isdir(memo_dir):
            os.mkdir(memo_dir)

        if debug:
            dbg_name = name if not src_name else f"<fn-name-{fn_hash[:24]}...>"
            print(
                f"[FS_MEMO] new name={dbg_name} memo_dir={memo_dir} use_torch_serdes={use_torch_serdes}"
            )

        def memoized_wrapper(
            *args,
            _memo_force_recompute=False,
            _memo_key=None,
            _memo_meta=None,
            **kwargs,
        ):
            nonlocal name, memo_dir, use_torch_serdes, debug

            if disable:
                if debug:
                    print("[FS_MEMO] disabled, calling fn")
                return fn(*args, **kwargs)

            if debug:
                dbg_name = name if not src_name else f"<fn-name-{fn_hash[:24]}...>"
                print(f"[FS_MEMO] call {dbg_name}")

            key = None
            if _memo_key is not None:
                if isinstance(_memo_key, str):
                    key = hash_strings((fn_hash, _memo_key))
                else:
                    key = universal_hash((fn_hash, _memo_key), b64=True)
            else:
                key = universal_hash([fn_hash, args, kwargs], b64=True)

            if tag is not None:
                if not os.path.isdir(f"{memo_dir}/tag/{tag}"):
                    os.makedirs(f"{memo_dir}/tag/{tag}", exist_ok=True)
                mpath = f"{memo_dir}/tag/{tag}/{key}.bin"
            else:
                mpath = f"{memo_dir}/{key}.bin"

            if not _memo_force_recompute and os.path.isfile(mpath):
                if debug:
                    print(f"[FS_MEMO] cache hit {mpath}")

                if use_torch_serdes:
                    return torch.load(mpath)
                else:
                    with open(mpath, "rb") as f:
                        return cbor2_load_ext(f)
            else:
                if debug:
                    print(f"[FS_MEMO] recomputing {mpath}")
                    t0 = time.time_ns()

                ret = fn(*args, **kwargs)
                t1 = time.time_ns()

                if use_torch_serdes:
                    torch.save(ret, mpath)
                    if debug:
                        file_size = os.path.getsize(mpath)
                        print(
                            f"[FS_MEMO] done in {round((t1-t0)/1e6,3)}ms, size: {(file_size)}"
                        )

                else:
                    with open(mpath, "wb") as f:
                        cbor2_dump_ext(ret, f)
                        if debug:
                            file_size = f.tell()
                            print(
                                f"[FS_MEMO] done in {round((t1-t0)/1e6,3)}ms, size: {(file_size)}"
                            )

                if _memo_meta is not None:
                    meta_path = f"{memo_dir}/metadata.cbor2.bin"
                    with open(meta_path, "ab+") as f:
                        cbor2_dump_ext(
                            dict(
                                _memo_meta=_memo_meta,
                                path=mpath,
                                time=int(time.time()),
                            ),
                            f,
                        )

                return ret

        return memoized_wrapper

    return wrapper


def memo_cache_get(
    key_dict, tag=None, memo_dir=DEFAULT_MEMO_DIR, use_torch_serdes=False, fallback=None
):
    """
    Retrieves a memoized value from cache.

    Args:
        key_dict: Dict containing the key information (e.g., function name, args, kwargs)
        tag: Optional tag for cache organization
        memo_dir: Memo cache directory
        use_torch_serdes: Use Torch serialization (True) or CBOR (False)
        fallback: Default value to return if cache miss

    Returns:
        The cached value or fallback value if cache miss
    """

    if not isinstance(memo_dir, str) or not os.path.isdir(memo_dir):
        memo_dir = DEFAULT_MEMO_DIR

    key = universal_hash(key_dict, b64=True)
    if tag:
        mpath = f"{memo_dir}/tag/{tag}/{key}.bin"
    else:
        mpath = f"{memo_dir}/{key}.bin"

    if not os.path.isfile(mpath):
        return fallback

    if use_torch_serdes:
        import torch

        return torch.load(mpath)
    else:
        with open(mpath, "rb") as f:
            return cbor2_load_ext(f)


def memo_cache_set(
    key_dict, value, tag=None, memo_dir=DEFAULT_MEMO_DIR, use_torch_serdes=False
):
    """
    Sets a value in the memo cache.

    Args:
        key_dict: Dict containing the key information (e.g., function name, args, kwargs)
        value: Value to cache
        tag: Optional tag for cache organization
        memo_dir: Memo cache directory
        use_torch_serdes: Use Torch serialization (True) or CBOR (False)
    """

    if not isinstance(memo_dir, str) or not os.path.isdir(memo_dir):
        memo_dir = DEFAULT_MEMO_DIR

    key = universal_hash(key_dict, b64=True)
    if tag:
        mpath = f"{memo_dir}/tag/{tag}/{key}.bin"
        if not os.path.isdir(f"{memo_dir}/tag/{tag}"):
            os.makedirs(f"{memo_dir}/tag/{tag}", exist_ok=True)
    else:
        mpath = f"{memo_dir}/{key}.bin"

    if use_torch_serdes:
        import torch

        torch.save(value, mpath)
    else:
        with open(mpath, "wb") as f:
            cbor2_dump_ext(value, f)


def memo_cache_delete(key_dict, tag, memo_dir=DEFAULT_MEMO_DIR):
    """
    Clears a value in the memo cache, if any.

    Args:
        key_dict: Dict containing the key information (e.g., function name, args, kwargs)
        tag: Optional tag for cache organization
        memo_dir: Memo cache directory
    """

    if not isinstance(memo_dir, str) or not os.path.isdir(memo_dir):
        memo_dir = DEFAULT_MEMO_DIR

    key = universal_hash(key_dict, b64=True)

    if tag:
        mpath = f"{memo_dir}/tag/{tag}/{key}.bin"
        if not os.path.isdir(f"{memo_dir}/tag/{tag}"):
            return
    else:
        mpath = f"{memo_dir}/{key}.bin"

    if os.path.isfile(mpath):
        os.remove(mpath)


def memo_cache_delete_bytag(tag, memo_dir=DEFAULT_MEMO_DIR):
    """
    Clears all values in the memo cache under a given tag, if any.

    Args:
        key_dict: Dict containing the key information (e.g., function name, args, kwargs)
        tag: Optional tag for cache organization
        memo_dir: Memo cache directory
    """

    if not isinstance(memo_dir, str) or not os.path.isdir(memo_dir):
        memo_dir = DEFAULT_MEMO_DIR

    mpath = None
    if tag:
        mpath = f"{memo_dir}/tag/{tag}"
        if not os.path.isdir(f"{memo_dir}/tag/{tag}"):
            return

    if mpath and os.path.isdir(mpath):
        shutil.rmtree(mpath)
