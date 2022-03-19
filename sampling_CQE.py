"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2022/3/7
@description: null
"""
from ComplexTemporalQueryData import *

dataset = ICEWS14()
cache_path = ComplexTemporalQueryDatasetCachePath(dataset.cache_path)
data = ComplexQueryData(dataset, cache_path)
data.preprocess_data_if_needed()
