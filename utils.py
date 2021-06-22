import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def no_bytes(df):
    
    for col in df:
        if df[col].dtype == 'object':
            df[col] = df[col].map(lambda x: x.decode("utf-8"))
            
    return df 


def import_and_decode(file_name):
    df = no_bytes(pd.read_sas(file_name))
    return df