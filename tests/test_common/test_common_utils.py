import pu4c
import numpy as np
import shutil

def test_TestDataDB():
    shutil.rmtree('work_dirs/tmpdb', ignore_errors=True)
    shutil.rmtree('work_dirs/catdb', ignore_errors=True)
    
    logger = pu4c.common.create_logger('work_dirs/test_TestDataDB.log')
    datadb = pu4c.common.utils.TestDataDB(dbname='tmpdb', root='work_dirs/')
    datadb.filesize = 1 * 1024  # 1k

    datadb.set("data1", 0)
    datadb.set("data1", 1)      # 覆盖
    assert datadb.get("data1") == 1
    logger.info(datadb.keys_dict)

    datadb.set("data2", np.random.randn(1000, 1000))    # 超过 1k，data1 和 data2 移动到新文件，主数据文件名路径不存在
    datadb.set("data3", 3)                              # 新创建的数据默认创建到主文件
    logger.info(datadb.keys_dict)
    assert datadb.keys_dict['data2'] != datadb.keys_dict['data3']
    
    datadb.rename("data1", "data3") # 仅修改键名路径文件，不移动数据，同时原本 data1 的数据失去管理可以被 gc
    logger.info(datadb.keys_dict)
    assert datadb.keys_dict['data2'] == datadb.keys_dict['data3']

    datadb.remove("data3")
    logger.info(datadb.keys_dict)
    assert "data3" not in datadb.keys_dict

    datadb2 = pu4c.common.utils.TestDataDB(dbname='catdb', root='work_dirs/')
    datadb2.set("data4", 4)
    datadb.copyfrom(datadb2, keys=['data4'])
    assert "data4" in datadb.keys_dict

    datadb.keys_dict.pop("data2")
    datadb.keys_dict.pop("data4")
    deleted_files = datadb.gc()     # 执行垃圾回收，全部清空数据后最多只会保留主数据文件
    assert len(deleted_files) != 0

if __name__ == '__main__':
    test_TestDataDB()
