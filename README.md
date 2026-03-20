# castle945 的 Python 工具包

[![PyPI - Version](https://img.shields.io/pypi/v/pu4c)](https://pypi.org/project/pu4c/) ![PyPI - Downloads](https://img.shields.io/pypi/dm/pu4c) [![Coverage Status](https://coveralls.io/repos/github/castle945/pyutils4castle945/badge.svg?branch=main)](https://coveralls.io/github/castle945/pyutils4castle945?branch=main)

### 介绍

#### 设计思路

1. 模块内文件组织，包括 `utils/common_utils.py` `app.py`  ，工具函数写到 `utils` 目录的文件中，应用函数写到 `app.py` 中，参数配置写到 `config.py` 中
2. 模块接口暴露规则
   1. `utils/__init__.py` 中可以导入（不依赖大型库的）代码文件如 `common_utils.py` 中的工具函数，从而支持链式调用这些工具函数如 `pu4c.common.utils.funcxxx()`
   2. 选择部分工具函数（即作为工具函数被其他代码调用，也作为命令行工具在调试终端中调用）和 `app.py` 中的应用函数暴露到 `pu4c/__init__.py` ，从而简化调用和增强兼容性
3. 弹性依赖和快速导包，保证安装时尽量不依赖其他库，使用时安装所需依赖库即可正常运行，保证 `import pu4c` 快速执行，这要求：
   1. `app.py` 中不能导入大型依赖库如 `open3d` ，而是在具体函数中导入
   2. `utils/__init__.py` 中导入的文件中，不能导入大型依赖库

#### 相关说明

- Python 导包有缓存机制，多次导入与单次导入耗时一致，意味着可以把所有的 import 语句写在函数中
- 可以通过 `help(pu4c.xxx.xxx)` 或 `pu4c.xxx.xxx.__doc__` 来查看函数注释
- 注意查看所用 `config.py` 中的配置进行个性化修改

### 安装

```bash
# pip 安装使用
pip install pu4c

# 源码安装
pip install -e .

# 执行单元测试
rm -rf build dist .pytest_cache .coverage work_dirs/*
pip install pytest coverage
pytest tests/
# 本地运行并查看单元测试覆盖率
coverage run --source=pu4c -m pytest tests/
coverage report
# 手动上传到 coveralls(因为 CI/CD 则需要测试数据文件，比较大)
pip install coveralls
export COVERALLS_REPO_TOKEN=XXXXX # 登录/AddRepo/StartUploadingCoverage/...
coveralls

# 打包上传 pypi
pip install setuptools wheel twine
python3 setup.py sdist bdist_wheel
twine upload dist/*
pip install pu4c -i https://pypi.org/simple
```

### 快速演示

| ![demo_det3d.png](docs/demo_det3d.png) | ![demo_seg3d.png](docs/demo_seg3d.png) | ![demo_occ3d.png](docs/demo_occ3d.png) |
| :----------------------------------: | :----------------------------------: | :----------------------------------: |
|          三维目标检测可视化          |          三维语义分割可视化          |          三维占据预测可视化          |

### 快速入门

#### 服务器端数据在本地界面中可视化

```bash
# 本地计算机作为 RPC 服务端
python -c "import pu4c; pu4c.common.start_rpc_server()"
ssh user@ip -R 30570:localhost:30570   # SSH 转发并在使用过程中保持终端 ssh 连接不断开，端口配置位于 pu4c/config.py，参数 -R remote_port:localhost:local_port
ssh user2@ip2 -R 30570:localhost:30570 # 可选，支持同时将多台服务器的端口转发到本机的同一个端口

# 服务器作为 RPC 客户端，可在交互式终端（如调试终端）中使用
import pu4c
pu4c.cv.cloud_viewer(filepath="/datasets/KITTI/object/training/velodyne/000000.bin", num_features=4, rpc=True) # 置 rpc=True 进行远程函数调用

# 注意
# 如果远程函数调用时使用了文件路径作为参数需要确保 RPC 服务器上存在该文件，对于数据集等可以通过 nfs 挂载到相同路径，或者修改 rpc 装饰器修改路径前缀
```

### 许可证

本代码采用 [GPL-3.0](LICENSE) 许可发布，这意味着你可以自由复制和分发软件，无论个人使用还是商业用途，但修改后的源码不可闭源且必须以相同的许可证发布
