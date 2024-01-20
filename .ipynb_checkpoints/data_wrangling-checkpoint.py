# -*- coding: utf-8 -*-

from import_library import *
from data_wrangling_func import *

data_path = 'data/'

def data_wangling_main():
    data, stock_names = data_wrangling(data_path,save=True)
    selected_columns = data.columns
    correlation_matrix(data,selected_columns,save=True)
    market_data = market_data_generation(data_path)
    beta_result_df = beta_computation(data,market_data,save = True)
    port_cluster(data, stock_names, save = True)
    print("Data wrangling done")
    print("\n")

if __name__ == '__main__':
    data_wangling_main()