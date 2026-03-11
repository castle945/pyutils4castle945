from typing import Any, Union, Optional, Dict, List, Tuple
import pu4c.config as cfg
from .utils.common_utils import rpc_func, convert_type

def start_rpc_server():
    import rpyc, pickle
    from rpyc.utils.server import ThreadedServer
    from pu4c.cv import rpc_func_dict as cv_rpc_func_dict
    class RPCService(rpyc.Service):
        def __init__(self):
            super().__init__()
            self.func_map = {
                'print': print,
            }
            self.func_map.update(cv_rpc_func_dict)
            for name, func in self.func_map.items():
                setattr(self.__class__, f'exposed_{name}', self._create_exposed_method(name, func))

        def _create_exposed_method(self, name, func):
            def exposed_method(self, serialized_args, serialized_kwargs):
                args, kwargs = pickle.loads(serialized_args), pickle.loads(serialized_kwargs)
                return pickle.dumps(func(*args, **kwargs))
            exposed_method.__name__ = f'exposed_{name}'
            return exposed_method
        # 如果不动态创建则需要手动逐个创建内容完全一样只有函数名不同的方法
        # def exposed_cloud_viewer(self, serialized_args, serialized_kwargs):
        #     args, kwargs = pickle.loads(serialized_args), pickle.loads(serialized_kwargs)
        #     return pickle.dumps(cloud_viewer(*args, **kwargs))

    server = ThreadedServer(RPCService, port=cfg.rpc_server_port, auto_register=True)
    server.start()

def create_logger(log_file: str = None):
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def deep_equal(
    var1, var2,
    tol: Tuple[float, float] = None,
    ignore_keys: List[str] = [],
    ignore_indices: List[Union[int, List]] = [],
    complex_type: bool = False,
) -> bool:
    """比较两个复杂变量是否相等，支持 dict/list/ndarray 等类型的嵌套
    Args:
        tol (tuple(atol, rtol)): 数值比较的容忍度，例如容许 0.01 级别的误差可设置为 (1e-2, 0)，即设置绝对误差即可
        ignore_keys: 忽略比较字典中某些键
        ignore_indices (list[int | list]): 忽略比较列表或元组中某些索引，例如 [1,[2,[3,4]]] 表示忽略掉遇到的第一层列表下标为 1 的项、第二层列表下标为 2 的项、第三层列表下标为 3,4 的项
        complex_type: 是否为复杂数据类型，即包括非 Python 内置类型或 numpy 类型
    """
    import numpy as np
    import rich
    import math

    def deep_equal_with_reason(var1, var2, reason, ignore_indices):
        # 比较数据类型
        if type(var1) != type(var2):
            return False, f"{reason}: 类型不一致\n" \
                          f"type1: {type(var1)}\n" \
                          f"type2: {type(var2)}\n"
        
        # 如果是字典，则比较键值对
        if isinstance(var1, dict):
            if len(ignore_keys) > 0:
                var1 = {k: v for k, v in var1.items() if k not in ignore_keys}
                var2 = {k: v for k, v in var2.items() if k not in ignore_keys}
            elif var1.keys() != var2.keys():
                return False, f"{reason}: 键不匹配\n" \
                              f"key1: {var1.keys()}\n" \
                              f"key2: {var2.keys()}\n"
            
            for k in var1.keys():
                is_equal, reason = deep_equal_with_reason(var1[k], var2[k], reason, ignore_indices)
                if not is_equal:
                    return False, f"['{k}']{reason}"
            return True, reason
        
        # 如果是列表或元组，则逐元素比较
        if isinstance(var1, (list, tuple)):
            if len(var1) != len(var2):
                return False, f"{reason}: 列表长度不相等\n" \
                              f"len1: {len(var1)}\n" \
                              f"len2: {len(var2)}\n"
            
            cur_ignore_indices, next_ignore_indices = [], []
            for idx in ignore_indices:
                if isinstance(idx, list):
                    next_ignore_indices = idx
                else:
                    cur_ignore_indices.append(idx)
            for i, (x, y) in enumerate(zip(var1, var2)):
                if i in cur_ignore_indices:
                    print(f'skip idx {i}')
                    continue
                is_equal, reason = deep_equal_with_reason(x, y, reason, ignore_indices=next_ignore_indices)
                if not is_equal:
                    return False, f"[{i}]{reason}"
            return True, reason
        
        # 如果是 NumPy 数组，则根据数组内容选择不同的比较方法
        if isinstance(var1, np.ndarray):
            if var1.shape != var2.shape:
                return False, f"{reason}: 数组形状不相等\n" \
                              f"shape1: {var1.shape}\n" \
                              f"shape2: {var2.shape}\n"
            
            if var1.dtype.kind in ['i', 'f']:
                # 数值数组，使用 np.isclose 进行比较 np.isclose: if abs(var1-var2) <= atol + rtol*abs(b)
                mask = np.isclose(var1, var2, atol=tol[0], rtol=tol[1]) if tol is not None else np.isclose(var1, var2) # rtol 相对误差容忍度
                if not np.all(mask):
                    idx = np.argwhere(~mask)[0] # 所有不相等值的索引，这里只打印第一个不相等的索引
                    return False, f"{reason}: 数组值在索引 {tuple(idx)} 处不相等\n" \
                                  f"data1: {var1[idx]}\n" \
                                  f"data2: {var2[idx]}\n"
            else:
                # 非数值数组（如字符串数组），使用 np.array_equal 进行比较
                if not np.array_equal(var1, var2):
                    idx = np.argwhere(var1 != var2)[0]
                    return False, f"{reason}: 数组值在索引 {tuple(idx)} 处不相等\n" \
                                  f"data1: {var1[idx]}\n" \
                                  f"data2: {var2[idx]}\n"
            return True, reason

        if isinstance(var1, float) and tol is not None:
            return (True, reason) if abs(var1 - var2) <= tol[0] + tol[1] * abs(var2) else (False, f"{reason}: 值不相等\n" \
                                                                                                  f"data1: {var1}\n" \
                                                                                                  f"data2: {var2}\n")

        if isinstance(var1, (int, float)) and math.isnan(var1) and math.isnan(var2):
            return (True, reason)

        # 其他类型，直接比较
        return (True, reason) if var1 == var2 else (False, f"{reason}: 值不相等\n" \
                                                           f"data1: {var1}\n" \
                                                           f"data2: {var2}\n")

    if complex_type:
        var1, var2 = convert_type(var1), convert_type(var2)
    is_equal, reason = deep_equal_with_reason(var1, var2, reason='', ignore_indices=ignore_indices)
    if not is_equal:
        rich.print(f"reason: {reason}")
    return is_equal
@rpc_func
def print(
    data, 
    reduce: bool = True, max_len: int = 10, 
    decimals: int = 2, 
    complex_type: bool = False, 
    rpc: bool = False,
) -> None:
    """打印复杂结构体，支持 dict/list/ndarray 等类型的嵌套，由于 VSCode 调试终端中打印效果变差，增加远程调用支持
    Args:
        reduce: 是否缩减列表项，即只打印列表中的一项数据
        max_len: 列表或一维数组最大长度，超过此值则缩减
        decimals: 保留的小数点位数
    """
    from rich import print as rprint
    import numpy as np
    np.set_printoptions(floatmode='fixed', precision=decimals)

    def reduce_list(data):
        if isinstance(data, dict):
            first_key = list(data.keys())[0] if data else None
            if (
                # 必要条件（字典非空 且 存在第一项的值为字典）
                (first_key is not None) and isinstance(data[first_key], dict) and \
                # 所有字典项的值均为相同结构的字典时，缩减
                all([(isinstance(v, dict) and (v.keys() == data[first_key].keys())) for k, v in data.items()])
            ):
                reduced_keys = list(data.keys())[1:]
                reduced_keys = reduced_keys if isinstance(first_key, str) else [str(key) for key in reduced_keys]
                return {first_key: reduce_list(data[first_key]), 'reduced keys': ', '.join(reduced_keys)}
            else:    
                return {k: reduce_list(v) for k, v in data.items()}
        elif isinstance(data, tuple):
            return tuple([reduce_list(i) for i in list(data)]) # 元组一般不长且不嵌套，不考虑缩减
        elif isinstance(data, list):
            if (
                # 必要条件（列表长度大于 1 且列表项需为相同类型），再满足任一缩减条件
                (len(data) > 1) and all([type(data[0]) == type(datai) for datai in data]) and \
                (
                    # 列表长度大于阈值实在太长，缩减
                    (len(data) > max_len) or \
                    # 列表项为相同结构的字典时，缩减
                    (isinstance(data[0], dict) and all([data[0].keys() == datai.keys() for datai in data])) or \
                    # 列表项为相同形状的数组时，缩减
                    (isinstance(data[0], np.ndarray) and all([data[0].shape == datai.shape for datai in data]))
                )
            ):
                return [reduce_list(data[0]), f'reduced, len({len(data)})'] # 大型多维数组最好用 ndarray 存储而非列表，如为列表则逐维考虑
            else:
                return [reduce_list(i) for i in data]
        elif isinstance(data, np.ndarray):
            if (
                # 作为一维数组长度大于阈值，缩减
                (data.shape[0] > max_len) or \
                # 多维数组，缩减
                (len(data.shape) > 1)
            ):
                return f'reduced, shape{data.shape}'
        
        return data
    def round_number(data, decimals=decimals):
        if isinstance(data, dict):
            return {k: round_number(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [round_number(i) for i in data]
        elif isinstance(data, tuple):
            return tuple([round_number(i) for i in list(data)])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return np.round(data, decimals)
        elif isinstance(data, float):
            return round(data, decimals)

        return data

    data = convert_type(data, typeinfo=True) if complex_type else data
    data = reduce_list(data) if reduce else data
    data = round_number(data) if decimals is not None else data
    rprint(data)
