import os
print(os.getcwd())
import sys
sys.path.append(os.getcwd())
import expression
from expression.ParamSchema import EntitySet


if __name__ == "__main__":
    e1 = EntitySet({1})
    q1 = EntitySet({1,2})
    print(e1 in q1)
