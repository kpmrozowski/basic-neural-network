import pandas as pd

def read_csv_file(method, file, type, count):
    method = method.astype(str)
    file = file.astype(str)
    type = type.astype(str)
    count = count.astype(str)

    csv_file = pd.read_csv("/data/{}/data.{}.{}.{}".format(method, file, type, count))
    
    return csv_file