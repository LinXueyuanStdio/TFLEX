"""
@date: 2022/3/11
@description: null
"""
class DefaultDict(dict):
    def __init__(self, default_value):
        super(DefaultDict, self).__init__()
        self._default_value = default_value

    def __missing__(self, key):
        value = self._default_value(key) if callable(self._default_value) else self._default_value
        self[key] = value
        return value


class LambdaSet(object):
    def __call__(self, key):
        return set()


class LambdaDefaultDict(object):
    def __init__(self, value):
        self._value = value

    def __call__(self, key):
        return self._value()


if __name__ == '__main__':
    import pickle
    import random

    sro_t = DefaultDict(DefaultDict(DefaultDict(LambdaSet())))

    print(sro_t[0][0][0])
    for i in range(10):
        for j in range(10):
            for k in range(10):
                sro_t[i][j][k] = {m for m in range(random.randint(3, 9))}
    print(sro_t[0][0][0])

    with open("output/test.pkl", "wb") as f:
        pickle.dump(sro_t, f)

    with open("output/test.pkl", "rb") as f:
        another = pickle.load(f)
    for i in another:
        for j in another[i]:
            for k in another[i][j]:
                print(another[i][j][k])
    for i in range(10, 12):
        for j in range(10, 12):
            for k in range(10, 12):
                print(another[i][j][k])
