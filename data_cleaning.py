import numpy as np
import pandas as pd


def numerical_convertor(df):
    df.replace('Nike' , 0, inplace = True)
    df.replace('Essential', 1, inplace = True)
    df.replace('Hoodie', 0, inplace = True)
    df.replace('T-shirt', 1, inplace = True)
    df.replace('north', 0, inplace = True)
    df.replace('south', 1, inplace = True)
    df.replace('oversize', 0, inplace = True)
    df.replace('perfectly fit', 1, inplace = True)
    df.replace('style does not matter', 2, inplace = True)
    df.replace('party', 0, inplace = True)
    df.replace('casual', 1, inplace = True)
    df.replace('school', 2, inplace = True)
    return df