def stream_average(avg_pre, cur_num_index, cur_num):
    """
    calculate the average of number in a stream
    given nums: List[float], calculate avg[n] = sum(nums[:n]) / len(nums[:n]) = sum(nums[:n]) / n
    however, we don't know the length of the stream, so we use a moving average
    so avg[n-1] = sum(nums[:n-1]) / (n-1)
    so n * avg[n] = (n-1) * avg[n-1] + nums[n]
    so avg[n] = avg[n-1] + (nums[n] - avg[n-1]) / n
    we don't need to remember every nums before n
    :param avg_pre: average of previous numbers from index 0 to cur_num_index-1, i.e. avg[n-1]
    :param cur_num_index: current number index starting from 0, i.e. n
    :param cur_num: current number, i.e. nums[n]
    :return: average of current number and previous numbers, i.e. avg[n]
    """
    return avg_pre + (cur_num - avg_pre) / (cur_num_index + 1)
