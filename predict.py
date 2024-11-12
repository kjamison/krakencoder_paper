"""
Functions to fit prediction models to predict demographic variables from connectome data,
and to test the significance of the difference in accuracy of those predictions from different inputs.
"""

import numpy as np
import pandas as pd
import time
import warnings
from tqdm.autonotebook import tqdm

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, mean_squared_error, roc_auc_score
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from scipy.linalg import LinAlgWarning

from scipy import stats
from statsmodels.stats.multitest import multipletests

#these are only needed here if we plot predictions as we loop through
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

from utils import is_numeric_field, add_krakencoder_package

add_krakencoder_package() #add krakencoder package to path if not already there
from krakencoder.train import random_train_test_split_groups


def fit_prediction_model(Data, demo_var_list, input_flavors, cvnum=10, gridloops=3, trainfrac=.8, resample_loops=0, randseed_base=None, 
                  print_prediction=True, plot_prediction=False, include_predicted_and_true_in_output=False,
                  residualize_names=[], residualize_age=True, residualize_sex=True, 
                  auto_expand_gridloops=0, ignore_warnings=True, return_predictor=False):
    """
    Fit a prediction model to predict demographic variables from connectome data.
    
    Parameters:
    Data: dictionary of data, including 'Tdemo' (demographic data), 'subjects' (subject IDs), 'subject_splits' (train/test splits) (See load_study_data)
    demo_var_list: list of str. list of demographic variables to predict
    input_flavors: list of str. list of input data types to use (eg: enc_FCmean)
    cvnum: int. number of cross-validation splits (default=10)
    gridloops: int. number of grid search depth level (default=3)
    auto_expand_gridloops: int. number of times to expand the search range if the best hyperparameter is at the edge of the grid (default=0)
    trainfrac: float. fraction of data to use for training within crossval (default=.8)
    resample_loops: int. number of resampling loops to perform (default=0=no resampling)
    randseed_base: int. random seed to use for resampling (default=None)
    print_prediction: bool. print prediction results to screen as we loop (default=True)
    plot_prediction: bool. plot predicted vs true scatterplots as we loop (default=False)
    include_predicted_and_true_in_output: bool. include predicted and true values in output dataframe. Otherwise only includes performance metrics. (default=False)
    residualize_names: list of str. list of names from demo_var_list to residualize (default=[])
        * For each name in residualize_names, we regress age and/or sex and create a new demo_var_name+'_resid' variable
        * ie: We fit a linear model y=(x1*age + x2*sex + x3), and create a new variable y_resid = y - (x1*age + x2*sex + x3)
    residualize_age: bool. residualize age from residualize_names (default=True)
    residualize_sex: bool. residualize sex from residualize_names (default=True)
    
    Returns:
    pd.DataFrame of performance metrics for each input_flavor and demo_var_list combination
        as well as the actual yholdout_true and yholdout_predicted values if include_predicted_and_true_in_output is True
    """
    if ignore_warnings:
        simplefilter("ignore", category=ConvergenceWarning)
        simplefilter("ignore",category=LinAlgWarning)
    
    Tdemo=Data['Tdemo']
    subjects=Data['subjects']
    subject_splits=Data['subject_splits']

    if residualize_names is None:
        #default
        residualize_names=['varimax_cog','varimax_satisf','varimax_er','nih_totalcogcomp_unadjusted']

    numeric_field=[f for f in Tdemo if is_numeric_field(Tdemo[f])]

    if 'Family_ID' in Tdemo:
        family_id=np.array(Tdemo['Family_ID'].tolist())
    else:
        family_id=np.arange(len(subjects))

    df_performance_resample=[]
    
    predictor_object_list=[]
    
    for iresample in tqdm(range(max(1,resample_loops))):
        if randseed_base is None:
            randseed=None
        else:
            randseed=randseed_base+iresample
        tstart=time.time()
        
        if subject_splits is not None:
            idx_train=subject_splits['subjidx_train']
            #idx_train=np.append(subject_splits['subjidx_train'],subject_splits['subjidx_val'])
            #idx_train=subject_splits['subjidx_val']
            #idx_train=subject_splits['subjidx_test']
            #idx_train=np.append(subject_splits['subjidx_test'],subject_splits['subjidx_val'])
            idx_holdout=subject_splits['subjidx_test']
            #idx_holdout=subject_splits['subjidx_val']
            #idx_holdout=subject_splits['subjidx_train']
            #idx_holdout=np.append(subject_splits['subjidx_test'],subject_splits['subjidx_val'])
            #idx_holdout=subject_splits['subjidx_train'][400:]
            
            if resample_loops > 0:
                #subjlist[subjidx_train],subjlist[subjidx_test],groups[subjidx_train],groups[subjidx_test]
                #idx_train,idx_holdout,_,_=random_train_test_split_groups(groups=family_id[subjidx],subjlist=subjidx,train_frac=trainfrac,seed=randseed)
                
                idx_train,_,_,_=random_train_test_split_groups(groups=family_id[subject_splits['subjidx_train']],subjlist=subject_splits['subjidx_train'],train_frac=trainfrac,seed=randseed)
                idx_holdout,_,_,_=random_train_test_split_groups(groups=family_id[subject_splits['subjidx_test']],subjlist=subject_splits['subjidx_test'],train_frac=trainfrac,seed=randseed)
        else:
            subjidx=np.arange(len(subjects))
            Xbehav=np.stack([Tdemo[s] for s in numeric_field],axis=-1)
            Xbehav_notnan=np.logical_not(np.any(np.isnan(Xbehav),axis=1))
            #subjidx=subjidx[Xbehav_notnan]
            #subjlist[subjidx_train],subjlist[subjidx_test],groups[subjidx_train],groups[subjidx_test]
            idx_train,idx_holdout,_,_=random_train_test_split_groups(groups=family_id[subjidx],subjlist=subjidx,train_frac=trainfrac,seed=randseed)

        #add age+sex regressed versions of varimax variables
        for v in residualize_names:
            if not v in Tdemo:
                continue
            y=Tdemo[v].to_numpy()
            a=Tdemo['Age'].to_numpy()
            is_m=Tdemo['Gender']=='M'
            #A=np.stack((a,is_m),axis=-1).astype(float) #Age, Sex
            #A=np.stack((a,a**2,is_m,np.ones(a.shape)),axis=-1).astype(float) #Age, Age^2, Sex, Offset
            if residualize_age and residualize_sex:
                A=np.stack((a,is_m,np.ones(a.shape)),axis=-1).astype(float) #Age, Sex, Offset
            elif residualize_age and not residualize_sex:
                A=np.stack((a,np.ones(a.shape)),axis=-1).astype(float) #Age, Offset
            elif residualize_sex and not residualize_age:
                A=np.stack((is_m,np.ones(a.shape)),axis=-1).astype(float) #Age, Offset
            
            fitmask=idx_train
            fitmask=fitmask[np.logical_not(np.any(np.isnan(np.hstack((A[idx_train],y[idx_train,np.newaxis]))),axis=1))]
            
            result=np.linalg.lstsq(A[fitmask],y[fitmask],rcond=None)
            beta=result[0]
            #[age sex ones] * beta = y
            y_resid=y-A.dot(beta)
            Tdemo[v+'_resid']=y_resid

        if Data['enc_FCmean'].shape != Data['enc_SCmean'].shape:
            input_flavors=[x for x in input_flavors if not x in ['enc_FCSCmean','enc_FCSCmean_norm','enc_FCSCdist']]
        
        df_performance=pd.DataFrame()

        for ix, xtype in enumerate(input_flavors):
            if xtype in Data:
                Xfull=Data[xtype]
            elif xtype == 'enc_FCSCmean':
                Xfull=Data['enc_FCmean']+Data['enc_SCmean']
            elif xtype == 'enc_FCSCmean_norm':
                Xfull=Data['enc_FCmean_norm']+Data['enc_SCmean_norm']
            elif xtype == 'enc_FCSCcat':
                Xfull=np.hstack((Data['enc_FCmean'],Data['enc_SCmean']))
            elif xtype == 'enc_FCSCcat_norm':
                Xfull=np.hstack((Data['enc_FCmean_norm'],Data['enc_SCmean_norm']))
            elif xtype == 'enc_FCSCdist':
                Xfull=np.sqrt(np.sum((Data['enc_FCmean']-Data['enc_SCmean'])**2,axis=1,keepdims=True))
            elif xtype == 'enc_FCSCdist_norm':
                Xfull=np.sqrt(np.sum((Data['enc_FCmean_norm']-Data['enc_SCmean_norm'])**2,axis=1,keepdims=True))
            elif xtype in Data['data_alltypes']:
                Xfull=Data['data_alltypes'][xtype]
            else:
                raise Exception("Unknown input type: %s" % (xtype))
            
            if Xfull is None:
                continue
            
            for idemo, demo_var_name in enumerate(demo_var_list):
                if print_prediction:
                    print("")
                    print("%d) %s -> %s:" % (iresample,xtype,demo_var_name))
                    print("input data: %dx%d" % (Xfull.shape[0],Xfull.shape[1]))
                    
                dummyvar_full=None

                is_logistic=False
                is_categorical=False
                labelenc=None
                
                if demo_var_name == 'Sex':
                    #demo_var=Tdemo['Gender']=='M' #1 = male
                    labelenc=LabelEncoder()
                    demo_var=pd.Series(labelenc.fit_transform(Tdemo['Gender'])) #Appears to encode M=1, F=0
                    #dummyvar_full=np.array(Tdemo['Strength_AgeAdj']=='M').astype(float)[:,np.newaxis]
                    #print("Adding dummy variable for Strength to predict Sex")
                    is_logistic=True
                    #is_categorical=True
                    
                elif demo_var_name == 'site':
                    labelenc=LabelEncoder()
                    #convert categorical string variable to numerical class:
                    demo_var=pd.Series(labelenc.fit_transform(Tdemo['site']))
                    #scramble rows of demo_var
                    #demo_var=demo_var.sample(frac=1,random_state=0).reset_index(drop=True)
                    is_categorical=True
                
                elif demo_var_name in Tdemo:
                    demo_var=Tdemo[demo_var_name]
                    
                    #dummyvar_full=np.array(Tdemo['Gender']=='M').astype(float)[:,np.newaxis]
                    #print("Adding dummy variable for Sex")
                    
                    if Tdemo[demo_var_name].dtype == np.bool_:
                        is_logistic=True
                        #is_categorical=True
                else:
                    raise Exception("variable name not found: %s" % (demo_var_name))
                    
                if dummyvar_full is None:
                    Xfull_withdummy=Xfull.copy()
                else:
                    Xfull_withdummy=np.concatenate((Xfull,dummyvar_full),axis=1)

                yfull=demo_var
                    
                y_outer=yfull[idx_train]
                family_id_outer=family_id[idx_train]

                idx_train_notnan=idx_train[np.logical_not(np.isnan(y_outer))]
                
                if Data['enc_is_cosinesim']:
                    subset_X=lambda X,idx: X[idx,:][:,idx_train_notnan]
                else:
                    subset_X=lambda X,idx: X[idx,:]
                
                X_outer=subset_X(Xfull_withdummy,idx_train_notnan)
                
                y_outer=yfull[idx_train_notnan]
                family_id_outer=family_id[idx_train_notnan]

                idx_holdout_notnan=idx_holdout[np.logical_not(np.isnan(yfull[idx_holdout]))]

                y_transformer=FunctionTransformer(func=lambda v:v,
                                                inverse_func=lambda v:v)
                
                if is_logistic:
                    #grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
                    #grid={"C":np.logspace(-4,4,15), "l1_ratio":np.linspace(0,1,15), "penalty":["elasticnet"]}# l1 lasso l2 ridge
                    grid={"C":np.logspace(-5,5,9)}# l1 lasso l2 ridge
                    gridspace_info={"C":"logspace"}


                    #reg=LogisticRegression(solver='saga',max_iter=100) #only 'saga' supports elastic

                    #reg=LogisticRegression(penalty='l2') #only 'saga' supports elastic
                    if Data['enc_is_cosinesim']:
                        reg=SVC(kernel='precomputed')
                    else:
                        reg=SVC(kernel='linear')
                    
                    scoring_name='balanced_accuracy'
                    metric_fun=balanced_accuracy_score
                    
                elif is_categorical:
                    grid={"alpha":np.logspace(-5,5,9)}# l1 lasso l2 ridge
                    gridspace_info={"alpha":"logspace"}

                    #reg=LogisticRegression(solver='saga',max_iter=100) #only 'saga' supports elastic

                    reg=RidgeClassifier() #only 'saga' supports elastic
                    
                    
                    scoring_name='balanced_accuracy'
                    metric_fun=balanced_accuracy_score
                    
                else:

                    if Xfull.shape[1]>1:
                        if Data['enc_is_cosinesim']:
                            reg=KernelRidge(kernel='precomputed')
                        else:
                            reg=KernelRidge(kernel='linear')
                        #reg=KernelRidge(kernel='rbf')
                        #reg=KernelRidge(kernel='laplacian')
                    else:
                        reg=Ridge()
                    
                    #print("Forcing Ridge regression, not KernelRidge")
                    #reg=Ridge()
                    
                    #grid={"alpha":np.logspace(-4,4,7)}# l1 lasso l2 ridge
                    grid={"alpha":np.logspace(-5,5,9)}# l1 lasso l2 ridge
                    gridspace_info={"alpha":"logspace"}
                    
                    if isinstance(reg,KernelRidge):
                        grid={"alpha":np.logspace(-5,5,9)}# l1 lasso l2 ridge
                        #grid={"alpha":np.logspace(-4,4,7)}# l1 lasso l2 ridge
                        gridspace_info={"alpha":"logspace"}
                        #since KernelRidge doesn't handle intercepts, need to remove/replace them
                        y_transformer=FunctionTransformer(func=lambda v,m=np.mean(y_outer):v-m,
                                                        inverse_func=lambda v,m=np.mean(y_outer):v+m)


                    #grid={"alpha":np.logspace(-4,4,7),"l1_ratio":np.linspace(0,1,7)}
                    #gridspace_info={"alpha":"logspace","l1_ratio":"linspace"}
                    #reg=ElasticNet(max_iter=1000)

                    #scoring_name=None
                    scoring_name='r2'
                    metric_fun=r2_score
                    #scoring_name='neg_mean_squared_error'
                    #metric_fun=mean_squared_error
                
                cv_splitter=GroupShuffleSplit(n_splits=cvnum, train_size=.8, random_state=randseed)

                #print(iresample,xtype,demo_var_name)
                reg_cv, cv_results = gridsearchcv_iterative(data=dict(X=X_outer, y=y_transformer.transform(y_outer), groups=family_id_outer),
                                                            estimator=reg,
                                                            cv=cv_splitter,
                                                            grid=grid,
                                                            gridspace_info=gridspace_info, 
                                                            num_loops=gridloops, 
                                                            auto_expand_loops=auto_expand_gridloops,
                                                            scoring=scoring_name, 
                                                            data_name='%s->%s' % (xtype,demo_var_name))

                if print_prediction:
                    print("best params:",reg_cv.best_params_)
                    print("best score (n=%d train):" % (len(idx_train_notnan)),reg_cv.best_score_)

                #######
                ytrain=yfull[idx_train_notnan].to_numpy()
                ynew=reg_cv.predict(subset_X(Xfull_withdummy,idx_train_notnan))
                ynew=y_transformer.inverse_transform(ynew)
                train_score=metric_fun(ytrain,ynew)
                
                train_rmse=None
                if scoring_name not in ['accuracy','balanced_accuracy']:
                    train_rmse=np.sqrt(mean_squared_error(ytrain,ynew))
                
                extra_performance_str=""
                train_cc=None
                if scoring_name == "r2":
                    train_cc=np.corrcoef(ytrain,ynew)[0,1]
                    extra_performance_str=", cc=%.4f" % (train_cc)
                    extra_performance_str+=", rmse=%.4f" % (train_rmse)
                if print_prediction:
                    print("train performance (%s, n=%d):" % (scoring_name,len(idx_train_notnan)),train_score,extra_performance_str)

                ########
                yholdout=yfull[idx_holdout_notnan].to_numpy()
                
                ynew=reg_cv.predict(subset_X(Xfull_withdummy,idx_holdout_notnan))
                ynew=y_transformer.inverse_transform(ynew)
                
                holdout_score=metric_fun(yholdout,ynew)
                
                holdout_rmse=None
                if scoring_name not in ['accuracy','balanced_accuracy']:
                    holdout_rmse=np.sqrt(mean_squared_error(yholdout,ynew))
                    
                extra_performance_str=""
                holdout_cc=None
                if scoring_name == "r2":
                    holdout_cc=np.corrcoef(yholdout,ynew)[0,1]
                    extra_performance_str=", cc=%.4f" % (holdout_cc)
                    extra_performance_str+=", rmse=%.4f" % (holdout_rmse)
                    
                if print_prediction:
                    print("holdout performance (%s, n=%d):" % (scoring_name,len(idx_holdout_notnan)),holdout_score,extra_performance_str)

                perf_tmp={'input_name':xtype, 'output_name':demo_var_name, 'metric':scoring_name,
                        'holdout_score':holdout_score,'holdout_cc':holdout_cc,'holdout_num':len(idx_holdout_notnan), 
                        'train_score':train_score,'train_num':len(idx_train_notnan),'train_rmse':train_rmse,'holdout_rmse':holdout_rmse,
                        'data_size':Xfull.shape,'model_params':reg_cv.best_params_,'model':str(reg_cv.best_estimator_),'resample':iresample}
                
                if include_predicted_and_true_in_output:
                    perf_tmp['yholdout_true']=yholdout
                    perf_tmp['yholdout_predicted']=ynew
                    
                df_performance=pd.concat((df_performance,pd.DataFrame([perf_tmp])),ignore_index=True)
                
                predictor_object_list.append({'predictor':reg_cv, 
                                              'idx_train':idx_train_notnan, 
                                              'idx_holdout':idx_holdout_notnan,
                                              'y_transformer':y_transformer,
                                              'metric_fun':metric_fun})
                
                if plot_prediction and iresample==0:
                    plt.figure()
                    #plt.subplot(len(demo_var_list),1,idemo+1)
                    #plt.scatter(yholdout,ynew)
                    if is_logistic or is_categorical:
                        cmlabels=None
                        if labelenc is not None:
                            cmlabels=labelenc.classes_
                        #cmdisp=ConfusionMatrixDisplay.from_predictions(yholdout,ynew, display_labels=cmlabels,normalize='all')
                        cmdisp=ConfusionMatrixDisplay.from_predictions(yholdout,ynew, display_labels=cmlabels)
                        #plt.plot([np.min(yholdout),np.max(yholdout)],[np.min(yholdout),np.max(yholdout)],'k:')
                        #plt.plot([np.min(ynew),np.max(ynew)],[np.min(ynew),np.max(ynew)],'k:')
                        plt.xlabel('predicted %s' % (demo_var_name))
                        plt.ylabel('true %s' % (demo_var_name))
                    else:
                        sns.regplot(x=yholdout,y=ynew,y_jitter=.1,x_jitter=.1,scatter_kws=dict(s=3,alpha=.5))
                        #plt.plot([np.min(yholdout),np.max(yholdout)],[np.min(yholdout),np.max(yholdout)],'k:')
                        #plt.plot([np.min(ynew),np.max(ynew)],[np.min(ynew),np.max(ynew)],'k:')
                        plt.xlabel('true %s' % (demo_var_name))
                        plt.ylabel('predicted %s' % (demo_var_name))
                    plt.title(demo_var_name)
                    plt.show()
        df_performance_resample.append(df_performance.copy())
        
    if return_predictor:
        return pd.concat(df_performance_resample), predictor_object_list
    else:
        return pd.concat(df_performance_resample)

def gridsearchcv_iterative(data,estimator,cv,grid,gridspace_info={}, num_loops=3, scoring=None, auto_expand_loops=0, data_name=None):
    """
    Perform grid search cross-validation with iterative grid search.
    
    Parameters:
    data: dictionary with keys 'X','y','groups'
    estimator: sklearn estimator (eg with fit() method)
    cv: sklearn cross-validation object
    grid: dictionary of hyperparameters to search over, and their initial grid values
        * example: grid={"C":np.logspace(-5,5,9)} will search over 'C' hyperparameter in estimator from 1e-5 to 1e5
        * the initial grid search will use these 9 log-spaced values, and then iterate to choose 9 new values zoomed in around the best one
    gridspace_info: dictionary of how to space the NEW grid values for each iteration
        * example: gridspace_info={"C":"logspace"} will use logspace values for the 'C' hyperparameter. could also be 'linspace'
    num_loops: int. number of grid search depth levels (default=3)
    scoring: str. name of scoring metric to use (default=None). 'r2','accuracy','balanced_accuracy'
    auto_expand_loops: int. number of times to expand the search range if the best hyperparameter is at the edge of the grid (default=0)
    data_name: str (optional). name of data being fit (for printing). (default=None)
    
    Returns:
    reg_cv: sklearn GridSearchCV object, containing the best estimator
    cv_results: dictionary of cross-validation results
    """
    cv_results={'params':[],'mean_test_score':[]}
    is_grid_edge={k:True for k in grid.keys()}
    
    #ignore singular matrix warnings until the last zoom level
    warnings.filterwarnings("ignore", message=r'Singular matrix')
    
    num_loops_orig=num_loops
    
    #for igrid in range(num_loops):
    igrid=0
    while igrid < num_loops:
        did_expand=False
        if igrid>0:
            grid_new={}
            for k in grid.keys():
                if k in gridspace_info and gridspace_info[k] == 'logspace':
                    newrange=lambda v1,v2,n: np.logspace(np.log10(v1),np.log10(v2),n)
                    #define additional ranges for expanding the grid to the left or right
                    dv=np.median(np.diff(np.log10(grid[k])))
                    newrange0=lambda v1,v2,n: np.logspace(np.log10(v1)-dv,np.log10(v2)-dv,n) #shift the range
                    newrangeN=lambda v1,v2,n: np.logspace(np.log10(v1)+dv,np.log10(v2)+dv,n)
                else:
                    newrange=lambda v1,v2,n: np.linspace(v1,v2,n)
                    #define additional ranges for expanding the grid to the left or right
                    dv=np.median(np.diff(grid[k]))
                    newrange0=lambda v1,v2,n: np.linspace(v1-dv,v2-dv,n) #shift the range
                    newrangeN=lambda v1,v2,n: np.linspace(v1+dv,v2+dv,n)
                try:
                    #use argmin(abs(list-x)) to avoid some numerical precision mismatch
                    idx=np.argmin(np.abs(np.array(grid[k])-reg_cv.best_params_[k]))
                    
                    if idx==0:
                        if auto_expand_loops>0:
                            did_expand=True
                            grid_new[k]=newrange0(grid[k][0], grid[k][-1], len(grid[k])) #shift the whole range
                        else:
                            grid_new[k]=newrange(grid[k][0], grid[k][1], len(grid[k]))
                    elif idx==len(grid[k])-1:
                        if auto_expand_loops>0:
                            did_expand=True
                            grid_new[k]=newrangeN(grid[k][0], grid[k][-1], len(grid[k])) #shift the whole range
                        else:
                            grid_new[k]=newrange(grid[k][-2], grid[k][-1], len(grid[k]))
                    else:
                        is_grid_edge[k]=False
                        grid_new[k]=newrange(grid[k][idx-1], grid[k][idx+1], len(grid[k]))
                except:
                    grid_new[k]=grid[k]

            grid=grid_new.copy()
        reg_cv=GridSearchCV(estimator,grid,cv=cv,scoring=scoring)
        
        if did_expand:
            num_loops=num_loops_orig+auto_expand_loops
            
        if igrid==num_loops-1:
            #show singular matrix warnings on last iteration
            warnings.filterwarnings("default", message=r'Singular matrix')
            
        reg_cv.fit(**data)
        
        cv_results['params']+=reg_cv.cv_results_['params']
        cv_results['mean_test_score']=np.concatenate((cv_results['mean_test_score'],reg_cv.cv_results_['mean_test_score']))
        
        igrid+=1
        
    #Print warning message if hyperparameter is at edge of grid
    for k in grid.keys():
        if is_grid_edge[k]:
            if data_name:
                print("Warning: edge of grid for %s (%s): %g. Consider expanding search range" % (data_name,k,reg_cv.best_params_[k]))
            else:
                print("Warning: edge of grid for %s: %g. Consider expanding search range" % (k,reg_cv.best_params_[k]))
    
    return reg_cv, cv_results

def average_flavor_predictions(df_performance, conngroups=None, include_predicted_and_true_in_output=False):
    """
    If we fit predictive models for each connectivity flavor individually, try averaging their predictions together, according to groups (eg SC or FC)
    If we used resampling to generate multiple predictions for each type, average each resample separately
    
    Parameters:
    df_performance: pd.DataFrame. output from fit_prediction_model
    conngroups: dict. dictionary of connectivity flavor to group them into SC,FC,SCFC, or None to infer from flavor names
    include_predicted_and_true_in_output: bool. keep predicted and true values in output dataframe. Otherwise only includes performance metrics. (default=False)
    
    Returns:
    New pd.DataFrame of performance metrics for each input_flavor and demo_var_list combination
        as well as the actual yholdout_true and yholdout_predicted values if include_predicted_and_true_in_output is True
    """
    numresamples=max(df_performance['resample'].to_numpy())+1
    demo_var_list=df_performance['output_name'].unique()
    conntypes=df_performance['input_name'].unique()
    if not conngroups:
        conngroups={}
        for c in conntypes:
            if 'fusion' in c or 'burst' in c or c.startswith('enc_'):
                conngroups[c]=''
            elif 'FC' in c:
                conngroups[c]='FC'
            else:
                conngroups[c]='SC'
                
    for iresample in range(numresamples):
        for idemo, demo_var_name in enumerate(demo_var_list):
            df_tmp=df_performance[(df_performance['output_name']==demo_var_name) & (df_performance['resample']==iresample)]
            
            for g in ['FC','SC','SCFC']:
                if g == 'SCFC':
                    inputname='enc_FCSCmean'
                    averagelist=[c for c in conntypes if conngroups[c] in ['SC','FC']]
                else:
                    inputname='enc_'+g+'mean'
                    averagelist=[c for c in conntypes if conngroups[c]==g]
                if len(averagelist) == 0:
                    continue
                
                y_true=np.stack([np.concatenate(df_tmp[df_tmp['input_name']==c]['yholdout_true'].to_numpy()) for c in averagelist])
                y_pred=np.stack([np.concatenate(df_tmp[df_tmp['input_name']==c]['yholdout_predicted'].to_numpy()) for c in averagelist])
                
                y_true=np.mean(y_true,axis=0)
                y_pred=np.mean(y_pred,axis=0)
                
                if demo_var_name == 'Sex':
                    y_pred=y_pred>=.5
                    metric_fun=balanced_accuracy_score
                    metric_field='holdout_score'
                else:
                    #metric_fun=r2_score
                    metric_fun=lambda x,y:np.corrcoef(x,y)[0,1]
                    metric_field='holdout_cc'
                score=metric_fun(y_true,y_pred)
            
                df_newtmp=df_tmp.iloc[0].copy()
                df_newtmp['input_name']=inputname
                df_newtmp[metric_field]=score
                df_newtmp['yholdout_true']=y_true
                df_newtmp['yholdout_predicted']=y_pred
                df_performance=pd.concat((df_performance,df_newtmp.to_frame().T),ignore_index=True)
                
    if not include_predicted_and_true_in_output:
        df_performance.drop(columns=['yholdout_true','yholdout_predicted'],inplace=True)
    return df_performance

def permutation_test_matched(y1, y2, nsamples=1000, twosided=False):
    """
    Perform a permutation test to compare two paired distributions of values (y1[i] and y2[i] from subject i), and test the significance of the hypothesis that
    mean(y1[i]-y2[i] for all subjects i)>0.
    
    We have a set of values from model1 (y1) and from model2 (pred_proba_model1), and we compute mean(y1-y2).
    We test the significance of this by swapping the values from y1 and y2 many times to generate a null distribution.
    
    Parameters:
    y1: np.array. [nsubj x 1] values from model 1
    y2: np.array. [nsubj x 1] values from model 2
    nsamples: int. number of permutations to perform (default=1000)
    twosided: bool. perform two-sided test (default=False)
    
    Returns:
    observed_difference: float. difference in means between y1 and y2 (no permutations)
    pval: float. permutation p-value for (mean(y1-y2)) > null (one-sided) or abs(mean(y1-y2)) > abs(null) (two-sided)
    """
    if np.prod(np.array(y2).shape) == 1:
        y2=np.ones_like(y1)*y2
    
    assert y1.shape == y2.shape, "y1 and y2 must have the same shape"
    
    y1len=len(y1.ravel())
    
    observed_difference = np.mean(y1.ravel()) - np.mean(y2.ravel())
    
    #we break this up into blocks of at most 10000 permutations to reduce memory usage
    #if nsamples < 10000, just one block
    blocksize=10000
    if nsamples > blocksize:
        blocks=[blocksize]*(nsamples//blocksize)
        if nsamples%blocksize > 0:
            blocks.append(nsamples%blocksize)
    else:
        blocks=[nsamples]
    
    mean_differences_nullperm=[]
    for block in blocks:
        mask = np.random.randint(2, size=[block,y1len])
        p1stack=np.stack([y1.ravel()]*block)
        p2stack=np.stack([y2.ravel()]*block)
        p1 = np.where(mask, p1stack, p2stack)
        p2 = np.where(mask, p2stack, p1stack)
        mean_differences_nullperm+=list(np.mean(p1,axis=1)-np.mean(p2,axis=1))
    mean_differences_nullperm=np.array(mean_differences_nullperm)
    
    if twosided:
        observed_difference=np.abs(observed_difference)
        mean_differences_nullperm=np.abs(mean_differences_nullperm)
        
    pval = np.mean(mean_differences_nullperm >= observed_difference)
    
    return observed_difference, pval

def prediction_comparison_significance_tests(df_performance, demo_var_list, input_flavors, dataname_list=None,
                       permutation_numsamples=10000, permutation_twosided=True):
    """
    Perform significance tests for comparison of performance metrics between different input_flavors for each demo_var_list.
    For each demographic variable in demo_var_list (eg: age), and each input flavor in input_flavors (eg: FC), 
        compare the prediction accuracy between two datasets (eg: encFC vs rawFC)
    
    Parameters:
    df_performance: pd.DataFrame. output from fit_prediction_model
    demo_var_list: list of str. list of demographic variables to compare
    input_flavors: list of str. list of input flavors to compare (eg: 'enc_FCmean' or 'FCcorr_fs86_hpf')
    dataname_list: list of str. list with names of 'dataname' values from df_performance. These are added when we load
        the data, and might have 'enc' or 'raw' in the same dataframe. dataname_list says which of the datasets in the 
        concatenated dataframe to compare. If we have more than 2 datanames, we don't do significance tests.
        If None, just use all 'dataname' values in df_performance (default=None)
    permutation_numsamples: int. number of permutations to perform for permutation significance tests (default=10000)
    permutation_twosided: bool. perform two-sided test (default=True)
    
    Returns:
    pd.DataFrame of significance tests between input_flavors or datanames for each demo_var_list
    """
    if dataname_list is None:
        dataname_list=[None]
        if 'dataname' in df_performance:
            dataname_list=df_performance['dataname'].unique()
            if len(dataname_list) == 1:
                dataname_list=[None]
    
    sigtest_inputs=[]
    for idemo, demo_var_name in enumerate(demo_var_list):
        df_tmp=df_performance[df_performance['output_name']==demo_var_name]
        
        for ix,xtype in enumerate(input_flavors):
            for idata,whichdata in enumerate(dataname_list):
                if whichdata is None:
                    y=df_tmp['holdout_cc'][df_tmp['input_name']==xtype].to_numpy()
                    #y=df_tmp['holdout_rmse'][df_tmp['input_name']==xtype].to_numpy()
                else:
                    y=df_tmp['holdout_cc'][(df_tmp['input_name']==xtype) & (df_tmp['dataname']==whichdata)].to_numpy()
                    #y=df_tmp['holdout_rmse'][df_tmp['input_name']==xtype & df_tmp['dataname']==whichdata].to_numpy()
                    
                if len(y) == 0:
                    continue
                
                #y=y[0]
                y_list=None
                y_list=y.copy()

                if y[0] is None or np.isnan(y[0]):
                    if whichdata is None:
                        y=df_tmp['holdout_score'][df_tmp['input_name']==xtype].to_numpy()
                    else:
                        y=df_tmp['holdout_score'][(df_tmp['input_name']==xtype) & (df_tmp['dataname']==whichdata)].to_numpy()
                    #y=y[0]
                    y_list=y.copy()
                    
                if y[0] is None or np.isnan(y[0]):
                    continue
                
                y=np.mean(y)
                
                sigtest_inputs.append(dict(xtype=xtype,demo_var_name=demo_var_name,dataname=whichdata,y=y,y_list=y_list))
                
    siglist=[]
    for idemo, demo_var_name in enumerate(demo_var_list):
        if len(dataname_list) > 2:
            #dont do significance tests between more than 2 datasets
            continue
        for ix,xtype in enumerate(input_flavors):
            for ix2, xtype2 in enumerate(input_flavors):
                if ix2<=ix:
                    continue
                
                #enc vs enc, raw vs raw (eg: compare encFC vs encSC)
                for idata,whichdata in enumerate(dataname_list):
                    sigtest_inputs1=[x for x in sigtest_inputs if x['xtype']==xtype and x['demo_var_name']==demo_var_name and x['dataname']==whichdata][0]
                    sigtest_inputs2=[x for x in sigtest_inputs if x['xtype']==xtype2 and x['demo_var_name']==demo_var_name and x['dataname']==whichdata][0]
                    
                    y1=sigtest_inputs1['y_list']
                    y2=sigtest_inputs2['y_list']
                    tcompare=stats.ttest_ind(y1,y2,equal_var=False)
                    continue
                    siglist.append(dict(sig1=sigtest_inputs1,sig2=sigtest_inputs2,stat=tcompare,p=tcompare.pvalue))
                                
            #enc vs raw
            #for this demographic variable (eg: age), and this input flavor (eg: FC), 
            # compare the prediction accuracy between two datasets (eg: encFC vs rawFC)
            for idata,whichdata in enumerate(dataname_list):
                for idata2, whichdata2 in enumerate(dataname_list):
                    if idata2<=idata:
                        continue            
                    sigtest_inputs1=[x for x in sigtest_inputs if x['xtype']==xtype and x['demo_var_name']==demo_var_name and x['dataname']==whichdata][0]
                    sigtest_inputs2=[x for x in sigtest_inputs if x['xtype']==xtype and x['demo_var_name']==demo_var_name and x['dataname']==whichdata2][0]
                    
                    y1=sigtest_inputs1['y_list']
                    y2=sigtest_inputs2['y_list']
                    tcompare=stats.ttest_ind(y1,y2,equal_var=False)
                    
                    #pval=tcompare.pvalue
                    statval=np.mean(y1)-np.mean(y2)
                    _,pval=permutation_test_matched(y1,y2,nsamples=permutation_numsamples,twosided=permutation_twosided)
                    
                    #_,pval1=permutation_test_matched(y1,0,nsamples=10000,twosided=False)
                    #_,pval2=permutation_test_matched(y2,0,nsamples=10000,twosided=False)
                    
                    pval1=np.mean(y1<=0)
                    pval2=np.mean(y2<=0)
                    if pval1>0 or pval2>0:
                        print("%10g %10g %10g" % (pval,pval1,pval2),demo_var_name,xtype,whichdata,whichdata2)
                        
                    siglist.append(dict(sig1=sigtest_inputs1,sig2=sigtest_inputs2,stat=tcompare,statval=statval,p=pval))
    
    for sigtype in ['enc_vs_raw','flavor']:
        #for multiple comparisons correction, we might have two scenarios:
        #1. "enc_vs_raw": for demo_var_name (eg age) and the same input flavor (eg FC), compare predictions from two different datanames (eg. encFC vs rawFC)
        #2. "flavor": for demo_var_name (eg age) and the same dataset (within enc), compare predictions from two different input flavors (eg SC vs FC)
        #Each item in siglist will be one of these two types.
        #We will correct for multiple comparisons within each type, but not between types.
        #For instance, group all the enc_vs_raw comparisons for all demographic variables, and correct within that group
        if sigtype=='enc_vs_raw':
            #find all significance comparisons of the "enc_vs_raw" type
            isig=[i for i,x in enumerate(siglist) if x['sig1']['dataname']!=x['sig2']['dataname']]
        else:
            #find all significance comparisons of the "flavor" type
            isig=[i for i,x in enumerate(siglist) if x['sig1']['dataname']==x['sig2']['dataname']]
        if len(isig)==0:
            continue
        pvals_uncorrected=[siglist[i]['p'] for i in isig]
        pvals_corrected=multipletests(pvals_uncorrected,method='fdr_bh')[1]
        for i,isig in enumerate(isig):
            siglist[isig]['p_corrected']=pvals_corrected[i]
            
    return siglist