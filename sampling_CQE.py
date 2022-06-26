"""
@date: 2022/3/7
@description: null
"""
from ComplexTemporalQueryData import *

dataset = ICEWS05_15()
cache_path = ComplexTemporalQueryDatasetCachePath(dataset.cache_path)
data = ComplexQueryData(dataset, cache_path)
data.preprocess_data_if_needed()
