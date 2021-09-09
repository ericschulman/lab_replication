import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm

# stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

#organization
from shi_test import *





def regular_test(yn, xn, setup_test):
    ll1, grad1, hess1, params1, ll2, grad2, hess2, params2 = setup_test(yn, xn)
    nobs = ll1.shape[0]
    llr = (ll1 - ll2).sum()
    omega = np.sqrt((ll1 - ll2).var())
    test_stat = llr/(omega*np.sqrt(nobs))
    print('regular: test, llr, omega ----')
    print(test_stat, llr, omega)
    print('---- ')
    return 1*(test_stat >= 1.96) + 2*(test_stat <= -1.96),test_stat


# helper functions for bootstrap

def compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2):
    """required for computing bias adjustement for the test"""
    n = ll1.shape[0]
    hess1 = hess1/n
    hess2 = hess2/n

    k1 = params1.shape[0]
    k2 = params2.shape[0]
    k = k1 + k2
    
    #A_hat:
    A_hat1 = np.concatenate([hess1,np.zeros((k2,k1))])
    A_hat2 = np.concatenate([np.zeros((k1,k2)),-1*hess2])
    A_hat = np.concatenate([A_hat1,A_hat2],axis=1)

    #B_hat, covariance of the score...
    B_hat =  np.concatenate([grad1,-grad2],axis=1) #might be a mistake here..
    B_hat = np.cov(B_hat.transpose())

    #compute eigenvalues for weighted chisq
    sqrt_B_hat= linalg.sqrtm(B_hat)
    W_hat = np.matmul(sqrt_B_hat,linalg.inv(A_hat))
    W_hat = np.matmul(W_hat,sqrt_B_hat)
    V,W = np.linalg.eig(W_hat)

    return V


def bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=0,trials=500):
    nobs = ll1.shape[0]
    
    test_stats = []
    variance_stats = []
    llr = ll1-ll2
     
    for i in range(trials):
        np.random.seed()
        sample  = np.random.choice(np.arange(0,nobs),nobs,replace=True)
        llrs = llr[sample]
        test_stats.append( llrs.sum() )
        variance_stats.append( llrs.var() )


    #final product, bootstrap
    V =  compute_eigen2(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2)
    test_stats = np.array(test_stats+ V.sum()/(2))
    variance_stats = np.sqrt(np.array(variance_stats)*nobs + c*(V*V).sum())

    #set up test stat   
    omega = np.sqrt((ll1 - ll2).var()*nobs + c*(V*V).sum())
    llr = (ll1 - ll2).sum() +V.sum()/(2)
    #print('V ----')
    #print(V.sum()/2)
    #print('----')
    return test_stats,variance_stats,llr,omega

# TODO 4: Get Bootstrap test working

def bootstrap_test(yn,xn,setup_test,c=0,trials=500,alpha=.05):
    ll1,grad1,hess1,params1,ll2,grad2,hess2,params2 = setup_test(yn,xn)

    #set up bootstrap distr
    test_stats,variance_stats,llr,omega  = bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,c=c,trials=trials)
    test_stats = test_stats/variance_stats
    
    #set up confidence intervals
    cv_lower = np.percentile(test_stats, 50*alpha, axis=0)
    cv_upper = np.percentile(test_stats, 100-50*alpha, axis=0)
    #print('---- bootstrap: llr, omega ----')
    #print(llr,omega)
    #print('----')

    return  2*(0 >= cv_upper) + 1*(0 <= cv_lower), cv_lower, cv_upper


def bootstrap_test_distr(test_stats,alpha):
    cv_lower = np.percentile(test_stats, 50*alpha, axis=0)
    cv_upper = np.percentile(test_stats, 100-50*alpha, axis=0)
    
    return  2*(0 >= cv_upper) + 1*(0 <= cv_lower), cv_lower, cv_upper
    
    

def test_table(yn,xn,setup_test, trials=1000):
    
    #bootstrap cv
    ll1,grad1,hess1,params1,ll2,grad2,hess2,params2 = setup_test(yn,xn)
    test_stats,variance_stats,llr,omega  = bootstrap_distr(ll1,grad1,hess1,params1,ll2,grad2,hess2,params2,trials=trials)
    test_stats = test_stats/variance_stats
    
    result_boot, cv_lower1, cv_upper1 = bootstrap_test_distr(test_stats,.1)
    result_boot, cv_lower2, cv_upper2 = bootstrap_test_distr(test_stats,.05)
    result_boot, cv_lower3, cv_upper3 = bootstrap_test_distr(test_stats,.01)
    
    #regular result
    result_class, test_stat = regular_test(yn,xn,setup_test)
    
    #shi results
    result_shi, stat_shi1, cv_shi1= ndVuong(yn,xn,setup_test,alpha=.1)
    result_shi, stat_shi2, cv_shi2= ndVuong(yn,xn,setup_test,alpha=.05)
    result_shi, stat_shi3, cv_shi3= ndVuong(yn,xn,setup_test,alpha=.01)


    print('\\begin{center}\n\\begin{tabular}{ccccc}\n\\toprule')
    print('\\textbf{Version} & \\textbf{Result} & \\textbf{90 \\% CI} & \\textbf{95 \\% CI} & \\textbf{99 \\% CI} \\\\ \\midrule' )
    print('Shi (2015) & H%s & [%.3f, %.3f] & [%.3f, %.3f] & [%.3f, %.3f] \\\\'%(result_shi, 
                                                  stat_shi1- cv_shi1, stat_shi1+ cv_shi1,
                                                  stat_shi2- cv_shi2,stat_shi2+ cv_shi2,
                                                  stat_shi3- cv_shi3,stat_shi3+ cv_shi3))
    print('Classical & H%s & [%.3f, %.3f] & [%.3f, %.3f] & [%.3f, %.3f] \\\\'%(result_class,
                                                 test_stat- 1.645,test_stat+ 1.645,
                                                 test_stat- 1.959,test_stat+ 1.959,
                                                 test_stat- 2.576,test_stat+ 2.576))
    print('Bootstrap & H%s & [%.3f, %.3f] & [%.3f, %.3f] & [%.3f, %.3f] \\\\'%(result_boot,
                                                 cv_lower1,cv_upper1,
                                                 cv_lower2,cv_upper2,
                                                 cv_lower3,cv_upper3))
    print('\\bottomrule\n\\end{tabular}\n\\end{center}')