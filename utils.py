import pandas as pd

def read_csv_file(method, file, type, count):
    # AttributeError: 'str' object has no attribute 'astype'
        # method = method.astype(str)
        # file = file.astype(str)
        # type = type.astype(str)
        # count = count.astype(str)

    csv_file = pd.read_csv("data/{}/data.{}.{}.{}.csv".format(method, file, type, count))
    
    return csv_file