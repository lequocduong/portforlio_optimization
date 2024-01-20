from import_library import *
data_path = 'data/'
savePath = 'Results/'
def convert_percentage_object_2_float(input):
    '''    
    Convert objects value to float ones , using in DataFrame transformation
    ----------
    Attributes:
    input: input data
    ----------
    Example:
    data['OCB_pct'] = data['OCB_pct'].apply(TransformCommaToColons)
    '''    
    return float(input.replace('%',''))

def read_file_csv(data_path,file):
    '''   
    Read csv file 
    ----------
    Attributes:
    file: file
    ----------
    Example:
    ocb_file = 'OCB Historical Data.csv'
    ocb_data = read_file_csv(ocb_file)
    '''
    data = pd.read_csv(data_path + file, sep=',',header = 0)
    data['Change %'] = data['Change %'].apply(convert_percentage_object_2_float)    
    return data

def convert_percentage_object_2_float(input):
    '''    
    Convert objects value to float ones , using in DataFrame transformation
    ----------
    Attributes:
    input: input data
    ----------
    Example:
    data['OCB_pct'] = data['OCB_pct'].apply(TransformCommaToColons)
    '''    
    return float(input.replace('%',''))

def data_wrangling(data_path, save=False):
    '''    
    Transforming and structuring data from one raw form into a desired format
    Only rate of change(ROC) is considered in this function
    missing value will automatically be filled with 0 (the price stays the same)
    ----------
    Attributes:
    data_path: data path
    ----------
    Example:
    data = data_wrangling(data_path)
    '''    
    all_names = os.listdir(data_path) # list all name in a folder
    stock_names = []
    for name in all_names:
        if name[:2] == 'VN' or name[-3:] != 'csv':
            continue
        data_temp = read_file_csv(data_path,name)  
        data_temp = data_temp[['Date','Change %']]
        data_temp.rename(columns={'Change %': f'{name[:3]}' },inplace=True)
        if name == all_names[0]:
            data = data_temp
        else: 
            data = data.merge(data_temp,how='left',on='Date')
        stock_names.append(f'{name[:3]}')
    data.fillna(0,inplace=True)
    if save:
        title = 'stock.csv'
        if not os.path.exists(f'{savePath}/'):
            os.makedirs( f'{savePath}/')  
        data.to_csv(f'{savePath}/{title}')    
    data.drop(columns=['Date'],inplace=True)
    return data,stock_names
    
def correlation_matrix(df,features,title=None,save=False):
    '''    
    Plot Correlation Matrix
    ----------
    Attributes:
    df : pd.DataFrame                
    features: list
        list of features of the df_dataset
    title: str, default: None
        The name of saved file
    save: bool, default: False
        save or not
    ----------
    Example:
    CorrelationMatrix(df_smoothed_scaled_filterd_dataset,orderedFeatured,title=None,save=False)
    '''    
    plt.close('all')
    train = df[features]
    plt.figure(figsize=(15, 15))
    dataplot = sb.heatmap(train.corr(), xticklabels=train.corr().columns, yticklabels=train.corr().columns,  cmap="YlGnBu", annot=True)
    if title == None:
        title = 'corr.png'    
    if save:
        if not os.path.exists(f'{savePath}/'):
            os.makedirs( f'{savePath}/')  
        plt.savefig(f'{savePath}/{title}', bbox_inches="tight")    
    # plt.show()

def market_data_generation(data_path):
    '''    
    Yield market data
    ----------
    Attributes:
    data_path : 
    ----------
    Example:
    market_data = market_data_generation(data_path)
    '''
    market_file = 'VN Index Historical Data.csv'
    market_data = read_file_csv(data_path,market_file)
    market_data = market_data['Change %']
    return market_data

def beta_computation(data,market_data,save=False):
    '''    
    Compute beta coefficient of stocks
    ----------
    Attributes:
    data : df.DataFrame
        stock data
    market_data: df.DataFrame
        market data
    save: bool, default:False
    ----------
    Returns:
    beta_result_df: df.DataFrame
    ----------
    Example:
    Data_wrangling.pybeta_result_df = beta_computation(data,market_data)
    '''
    beta = []
    Var = np.var(market_data, ddof =1 )
    for i in range(len(data.columns)):
        # beta = cov(r_stock_i,r_market)/Var(r_market)
        CovarMat = np.cov(market_data,data.iloc[:,i])
        Covar = CovarMat[1,0] # Take out Cov
        beta.append(Covar/Var)
    
    dict_beta = {
        'stocks' : data.columns,
        'beta': beta
    }
    beta_result_df = pd.DataFrame(dict_beta)
    if save:
        title = 'beta_computation.csv'
        if not os.path.exists(f'{savePath}/'):
            os.makedirs( f'{savePath}/')  
        beta_result_df.to_csv(f'{savePath}/{title}',index=False)
    return beta_result_df


def port_cluster(data,stock_names,save = False):
    '''    
    Portfolio clustering
    ----------
    Attributes:
    data : df.DataFrame
        stock data
    stock_names: list
        stock names
    ----------
    Example:
    port_cluster(data, stock_names, save = True)
    '''
    from sklearn.cluster import KMeans
    mean_returns = pd.DataFrame(np.mean(data,axis=0),columns=['means'])
    cov_returns = np.cov(data, rowvar = False) # convert each columns --> variables while rows-->observation
    cov_returns_df = pd.DataFrame(cov_returns,columns = stock_names, index = stock_names)
    df = pd.concat([mean_returns, cov_returns_df],axis = 1)    
    with open('desidered_cluster.txt') as f:
        n_clusters = f.read()        
    asset_cluster = KMeans(algorithm='auto', max_iter=600, n_clusters = int(n_clusters))
    asset_cluster.fit(df)
    assets = np.array(stock_names)
    centroids = asset_cluster.cluster_centers_
    labels = asset_cluster.labels_
    
    if save:
        title = 'Cluster.txt'
        with open(f'{savePath}/{title}', 'w') as fp:
            for i in range(len(centroids)):                
                clt = np.where(labels == i)
                assetCluster = assets[clt]
                fp.write(f'Cluster: {i+1}\n')
                fp.write(f'{assetCluster}\n')

