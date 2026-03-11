import pu4c
import copy

def test_deep_equal():
    datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')
    infos = datadb.get('mmdet3d/nuscenes_infos')
    infos2 = copy.deepcopy(infos)
    infos2['data_list'][0]['sample_idx'] = 100
    infos2['data_list'][1] = None
    infos2['data_list'][0]['instances'][0]['bbox_label'] = 100
    assert pu4c.common.deep_equal(infos, infos2, ignore_keys=['sample_idx'], ignore_indices=[1, [0]]) is True

class Dog:
    def __init__(self, name, num_legs=4):
        self.name = name
        self.num_legs = num_legs
        
def test_print():
    datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')
    infos = datadb.get('mmdet3d/nuscenes_infos')
    infos['data_list'][0]['complex_cls'] = Dog(name='wangcai')
    infos['data_list'][1]['complex_cls'] = Dog(name='laifu')
    pu4c.common.print(infos, complex_type=True)

if __name__ == '__main__':
    test_deep_equal()
    test_print()
