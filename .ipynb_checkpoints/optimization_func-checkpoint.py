from import_library import *
data_path = 'data/'
savePath = 'Results/'

def max_returns(mean_returns, portfolio_size):
    '''    
    Maximial Expected Returns (reference)
    ----------
    Attributes:
    mean_returns: 
    portfolio_size: 
    ----------
    Example:
    mean_returns = pd.DataFrame(np.mean(data,axis=0),columns=['means'])
    portfolio_size = len(stock_names)
    res = max_returns(mean_returns,portfolio_size)
    res.x # get optimal value

    Additional:
    min x : c.T @ x --> Minimize
    A_ub @ x <= b_ub ---> inequality constrains
    A_eq @ x = A_ub --> equality constrains
    l<= x <= u (boudary)    
    '''  
    from scipy.optimize import linprog
    c = (np.multiply(-1,mean_returns))
    A = np.ones([portfolio_size,1]).T # weights 
    b = [1] 
    # minimize a linear objective function subject 
    # to linear equality and inequality constraints (of negative function --> max)
    res = linprog(c, # Minimize of negative mean_returns --> maximize
                  A_ub = A,  # (1,5)x(5,1) <= b --> sum of weights < =1 
                  b_ub = b,   #
                  bounds =(0,1),  # weights in range (0,1)
                  method = 'simplex')
    return res

def min_risk_port(cov_returns, portfolio_size):
    '''    
    Minimize risk portfolio
    ----------
    Attributes:
    mean_returns: 
    cov_returns: 
    ----------
    Example:    
    portfolio_size = len(stock_names)
    cov_returns = np.cov(data,rowvar=False) 
    opt = min_risk_port(cov_returns, portfolio_size) 
    opt.x  
    '''  
    from scipy import optimize
    def f(x,cov_returns):
        # typical second orders 
        func = np.matmul(np.matmul(x,cov_returns),x.T)
        return func

    def constraint_eq(x):
        A = np.ones(x.shape)
        b = 1
        # A@ x.T - 1
        constrant_val = np.matmul(A,x.T) - b
        return constrant_val

    x_init = np.repeat(0.1, portfolio_size)
    cons = ({
        'type' : 'eq',
        'fun': constraint_eq,
    })
    lb = 0
    ub =1 
    bnds = tuple([(lb,ub) for x in x_init])
    opt = optimize.minimize(f,
                             x0 = x_init,
                             args = (cov_returns),
                             bounds = bnds,# 0-1
                             constraints = cons, # Sum of weights - 1 =0 
                             tol = 10**-3,
                            )
    return opt

def min_risk_constraint(mean_returns,cov_returns,portfolio_size, R=0.06):
    '''    
    Minimal risk and Maximum return portfolios
    ----------
    Attributes:
    mean_returns: 
    cov_returns: 
    portfolio_size:
    R : value of inequality constraint
    ----------
    Example:    
    portfolio_size = len(stock_names)
    cov_returns = np.cov(data,rowvar=False)     
    opt = min_risk_constraint(mean_returns, cov_returns, portfolio_size, 0.06)
    opt.x
    '''  
    from scipy import optimize
    def f(x,cov_returns):
        # typical second orders  - main function - 
        func = np.matmul(np.matmul(x,cov_returns),x.T)
        return func

    def constraint_eq(x):
        # A@x - B = 0 
        # sum of weights =1 
        A_eq = np.ones(x.shape)
        b_eq = 1
        # A@ x.T - 1 = 0
        eq_constrant_val = np.matmul(A_eq,x.T) - b_eq
        return eq_constrant_val

    def constrain_ineq(x, mean_returns, R):
        # A@x >= R
        # func >= b 
        # max expected return
        A_ineq = np.array(mean_returns).T
        b_ineq = R
        Ineq_constrant_val =  np.matmul(A_ineq,x) - b_ineq
        return Ineq_constrant_val

    x_init = np.repeat(0.1, portfolio_size)
    cons = (
        {'type' : 'eq', 'fun': constraint_eq},
        {'type' : 'ineq', 'fun': constrain_ineq,'args':(mean_returns,R)},
    )
    # x boundarys
    lb = 0
    ub = 1 
    bnds = tuple([(lb,ub) for x in x_init])
    # Main optimizing procedure
    opt = optimize.minimize(f,
                             x0 = x_init,
                             args = (cov_returns),
                             method = 'trust-constr',
                             bounds = bnds,# 0-1
                             constraints = cons, # Sum of weights - 1 =0 
                             tol = 10**-3,
                            )
    return opt

def max_sharpe_ratio_opt(mean_returns, cov_returns, risk_free_rate, portfolio_size):
    '''    
    Maximizing Sharpe Ratio --> compensate the risk_free
    ----------
    Attributes:
    mean_returns: 
    cov_returns: 
    risk_free_rate: daily risk rate return
    portfolio_size:    
    ----------
    Example:    
    Rf = 2.85
    r0 = (np.power((1 + Rf/100 ),  (1.0 / 360.0)) - 1.0) * 100 
    shapre_result = max_sharpe_ratio_opt(mean_returns, cov_returns, r0, portfolio_size)
    '''  
    from scipy import optimize 
    # define maximization of Sharpe Ratio using principle of duality
    def  f(x, mean_returns, cov_returns, risk_free_rate, portfolio_size):
        func_num = np.matmul(np.array(mean_returns).T,x)- risk_free_rate
        func_den = np.sqrt(np.matmul(np.matmul(x, cov_returns), x.T) )        
        func = -(func_num / func_den)
        return func

    #define equality constraint representing fully invested portfolio
    def constraint_eq(x):
        A = np.ones(x.shape)
        b = 1 
        constraint_val = np.matmul(A,x.T)-b 
        return constraint_val    
    
    #define bounds and other parameters
    x_init=np.repeat(0.33, portfolio_size)
    cons = ({'type': 'eq', 'fun':constraint_eq})
    lb = 0
    ub = 1
    bnds = tuple([(lb,ub) for x in x_init])
    
    #invoke minimize solver
    opt = optimize.minimize (f, 
                             x0 = x_init, 
                             args = (mean_returns,cov_returns,risk_free_rate, portfolio_size), method = 'SLSQP',
                             bounds = bnds, 
                             constraints = cons, 
                             tol = 10**-3,
                            )    
    return opt


