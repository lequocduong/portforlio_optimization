# -*- coding: utf-8 -*-
from import_library import *
from optimization_func import *
from data_wrangling_func import read_file_csv 
data_path = 'data/'
savePath = 'Results/'


def input_procesing(stocks_input):
    stocks_list = stocks_input.split(" ")
    return stocks_list

def import_data(data_path):
    # data, mean_returns, cov_returns, portfolio_size = import_data(savePath)
    file = 'stock.csv'
    with open('portfolio.txt') as f:
        stocks_input = f.read()
        
    columns = input_procesing(stocks_input)
    # columns = input
    data = pd.read_csv(data_path + file, sep = ',',header = 0, usecols = columns)
    stock_names = data.columns

    mean_returns = pd.DataFrame(np.mean(data,axis=0),columns=['means'])
    cov_returns = np.cov(data,rowvar=False) # convert each columns --> variables while rows-->observation
    
    portfolio_size = len(stock_names)
    return data, mean_returns, cov_returns, portfolio_size


def main_func(mean_returns,cov_returns,portfolio_size):
    # maximal expected portfolio return computation for the k-portfolio
    result1 = max_returns(mean_returns,portfolio_size)
    max_return_weights = result1.x
    max_exp_port_return = np.matmul( np.array(mean_returns).T,max_return_weights)
    print(f'Maximal Expected Portfolio Return: {max_exp_port_return[0]:.3f}')
    
    # Expected portfolio return computation for the minium risk k-portfolio
    result2 = min_risk_port(cov_returns,portfolio_size)
    min_risk_weights = result2.x
    min_risk_port_return = np.matmul( np.array(mean_returns).T,min_risk_weights)
    print(f'Expected Return of Minium Risk Portfolio: {min_risk_port_return[0]:.3f}')
    
    # Compute efficient set for maximum return and minium risk porfolios
    increment = 0.001 
    low = min_risk_port_return
    high = max_exp_port_return
    
    # iniitialize optimal weight set and risk-return point set
    x_optimal = []
    ex_port_return_point = []
    
    # repeated execution of function min_risk_constraint to determine the efficient set
    while (low < high):
        result3 = min_risk_constraint(mean_returns, cov_returns, portfolio_size, low)
        x_optimal.append(result3.x)
        ex_port_return_point.append(low)
        low = low +  increment
    
    #gather optimal weight set    
    x_optimal_array = np.array(x_optimal)
    
    return x_optimal_array, ex_port_return_point
    
def efficient_frontier_plot(cov_returns,x_optimal_array,ex_port_return_point,save=False):

    #obtain annualized risk for the efficient set portfolios 
    #for trading days = 251
    min_risk_point = np.diagonal(np.matmul((np.matmul(x_optimal_array,cov_returns)),\
                                         np.transpose(x_optimal_array)))
    risk_point =   np.sqrt(min_risk_point*251) 
    
    #obtain expected portfolio annualized return for the 
    #efficient set portfolios, for trading days = 251
    ret_point = 251*np.array(ex_port_return_point) 

    
    NoPoints = risk_point.size
    colours = "blue"
    area = np.pi*3
    
    plt.title('Efficient Frontier for k-portfolio 1 of Dow stocks')
    plt.xlabel('Annualized Risk(%)')
    plt.ylabel('Annualized Expected Portfolio Return(%)' )
    plt.scatter(risk_point, ret_point, s=area, c=colours, alpha =0.5)
    if save:
        title = 'Efficient_Frontier.png'
        if not os.path.exists(f'{savePath}/'):
            os.makedirs( f'{savePath}/')  
        plt.savefig(f'{savePath}/{title}', bbox_inches="tight") 
        plt.show()
    return risk_point,ret_point

def optimal_weight(data,risk_point,x_optimal_array,ret_point,save=False):
    
    # expected_annualized_risk = 32 # Input from external
    expected_annualized_risk = int(input("\nEnter your expected_annualized_risk: "))
    idx = np.where((risk_point > expected_annualized_risk) & (risk_point< expected_annualized_risk+1))[0][0]
    print(f'Stocks: {data.columns.tolist()}')
    print(f'Optimal weight: {np.round(x_optimal_array[idx],3)}')
    print(f'Annualized Risk: {risk_point[idx]:.2f} \nReturn of the efficient set portfolios: {ret_point[idx][0]:.2f}')
    if save:
        title = 'port_weights.txt'
        with open(f'{savePath}/{title}', 'w') as fp:            
            fp.write(f'Stocks: {data.columns.tolist()}\n')
            fp.write(f'Optimal weight: {np.round(x_optimal_array[idx],3)}\n')
            fp.write(f'Annualized Risk: {risk_point[idx]:.2f} \nReturn of the efficient set portfolios: {ret_point[idx][0]:.2f}')

def sharpe_ratio_opt(mean_returns,cov_returns,portfolio_size,Rf = 2.85):
    # 3% -> annual risk free    
    r0 = (np.power((1 + Rf/100 ),  (1.0 / 360.0)) - 1.0) * 100 
    
    #initialization
    x_opt =[]
    min_risk_point = []
    exp_port_return_point =[]
    max_sharpe_ratio = 0
    
    #compute maximal Sharpe Ratio and optimal weights
    shapre_result = max_sharpe_ratio_opt(mean_returns, cov_returns, r0, portfolio_size)
    x_opt.append(shapre_result.x)
    
    #compute risk returns and max Sharpe Ratio of the optimal portfolio   
    x_opt_array = np.array(x_opt)
    risk = np.matmul((np.matmul(x_opt_array,cov_returns)), np.transpose(x_opt_array))
    exp_return = np.matmul(np.array(mean_returns).T,x_opt_array.T)
    ann_risk =   np.sqrt(risk*251) 
    ann_ret = 251*np.array(exp_return) 
    max_sharpe_ratio = (ann_ret-Rf)/ann_risk 
    #display results
    print("\n")
    print('Maximal Sharpe Ratio: ', max_sharpe_ratio, '\nAnnualized Risk (%):  ', \
          ann_risk, '\nAnnualized Expected Portfolio Return(%):  ', ann_ret)
    print('Optimal weights (%):\n',  x_opt_array*100 )

def main_opt():
    data, mean_returns, cov_returns, portfolio_size = import_data(savePath)
    x_optimal_array, ex_port_return_point = main_func(mean_returns,cov_returns,portfolio_size)
    risk_point,ret_point = efficient_frontier_plot(cov_returns,x_optimal_array,ex_port_return_point,save=True)
    optimal_weight(data,risk_point,x_optimal_array,ret_point,save=True)
    sharpe_ratio_opt(mean_returns,cov_returns,portfolio_size,Rf = 2.85)
    print("Optimization part done")     
    
if __name__ == '__main__':
    main_opt()
    