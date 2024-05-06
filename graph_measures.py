"""
Functions for computing graph metrics on connectivity matrices and comparing those metrics from predicted and true connectomes.
"""
import os
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd

from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
import scipy.stats
from sklearn.metrics import r2_score

from utils import timestamp, krakendir, add_krakencoder_package, add_bctpy_package
from data import get_matching_subject

add_krakencoder_package() #add krakencoder package to path if not already there
from krakencoder.data import clean_subject_list

add_bctpy_package() #add bct package to path if not already there (if downloaded as local source)
import bct
    
def graph_measure_summary(Ctri, tri_axis=1, keep_proportion=None, louvain_iterations=100):
    """
    Compute graph measures for a set of connectivity matrices
    
    Parameters:
    Ctri: numpy array (Nsubj x M) of connectivity matrices (Nsubj=number of matrices, M=number of elements (edges) in the upper triangle of the matrix)
    tri_axis: axis of Ctri that contains the upper triangle of the connectivity matrix
    keep_proportion: float (0-1). if not None, threshold the connectivity matrix to keep this proportion of the strongest connections
    louvain_iterations: int (number of iterations to run the Louvain community detection algorithm. Default 100
    
    Returns:
    pandas dataframe containing the graph measures for each connectivity matrix
    """
    is_1d = Ctri.ndim == 1
    if is_1d == 1:
        Ctri=Ctri[np.newaxis,:]

    n=np.round(np.sqrt(.25+2*Ctri.shape[tri_axis])+.5).astype(int)
    triu=np.triu_indices(n,k=1) #note: this is equivalent to tril(X,-1) in matlab
    
    nontri_axis=max(1-tri_axis,0)

    summary_array=[]

    for i in tqdm(range(Ctri.shape[nontri_axis])):
        C=np.zeros((n,n))
        C[triu]=Ctri[i,:]
        C[triu[1],triu[0]]=Ctri[i,:]
        np.fill_diagonal(C,0) #just to be extra sure

        #abs and set diagonal to zero
        W=np.abs(C)*(1-np.eye(C.shape[0]))
        if keep_proportion is not None:
            W=bct.threshold_proportional(W,keep_proportion)

        C[W==0]=0
        
        L=bct.weight_conversion(W,'lengths')
        
        D=scipy_dijkstra(L)

        cpl,eff_global,ecc,radius,diameter=bct.charpath(D,include_infinite=False)
        #[0] lambda : characteristic path length
        #[1] efficiency : global efficiency (same as mean of "closeness centrality" across nodes)
        #[2] ecc : Nx1 eccentricity at each vertex
        #[3] radius : radius of graph
        #[4] diameter : diameter of graph (furthest connected pair)

        clustcoef=bct.clustering_coef_wu(W)

        ec=bct.eigenvector_centrality_und(W)

        bc=bct.betweenness_wei(L,use_fast_experimental=True)
        bc=bc/((n-1)*(n-2)) #normalize

        _,modularity=bct.modularity_und(W)

        
        modularity_louvain=[bct.community_louvain(W, B='modularity', seed=None)[1] for _ in range(louvain_iterations)]
        modularity_louvain=np.array(modularity_louvain)
        
        assortativity=bct.assortativity_wei(C,0)
        assortativity_abs=bct.assortativity_wei(W,0)

        specrad=np.max(np.abs(np.linalg.eigvals(W)))
        
        deg=bct.degrees_und(W)

        strength_abs=bct.strengths_und(np.abs(W))
        strength=bct.strengths_und(C)
        strength_pos,strength_neg,_,_=bct.strengths_und_sign(C)

        summary_dict=dict(nodes=n, sparsity=np.mean(W[triu]==0),
                          edge_abs_mean=np.mean(W[triu]), edge_abs_stdev=np.std(W[triu]), edge_abs_skew=scipy.stats.skew(W[triu]), 
                          degree_mean=np.mean(deg), degree_stdev=np.std(deg), 
                          strength_mean=np.mean(strength), strength_stdev=np.std(strength),
                          strength_abs_mean=np.mean(strength_abs), strength_abs_stdev=np.std(strength_abs),
                          strength_pos_mean=np.mean(strength_pos), strength_pos_stdev=np.std(strength_pos),
                          strength_neg_mean=np.mean(strength_neg), strength_neg_stdev=np.std(strength_neg),
                          characteristic_path_length=cpl, global_efficiency=eff_global, diameter=diameter, radius=radius, 
                          clustcoef_mean=np.mean(clustcoef), clustcoef_stdev=np.std(clustcoef), 
                          eigen_centrality_mean=np.mean(ec), eigen_centrality_stdev=np.std(ec),
                          betweenness_centrality_mean=np.mean(bc), betweenness_centrality_stdev=np.std(bc),
                          modularity=modularity, assortativity=assortativity, assortativity_abs=assortativity_abs, spectral_radius=specrad,
                          modularity_louvain_mean=np.mean(modularity_louvain), modularity_louvain_stdev=np.std(modularity_louvain))
        
        summary_array+=[summary_dict]
        
    return pd.DataFrame(summary_array)


#compare graph measures summary between predicted and true, by family groups
def graph_measure_summary_comparison(df,df_pred, prediction_type_list, twininfo_full, df_pred_retest=None, inputname=''):
    """
    Compare graph measures between observed and true, by family groups
    
    Parameters:
    df: pandas dataframe containing the observed graph measures
    df_pred: pandas dataframe containing the predicted graph measures
    prediction_type_list: list of prediction types to compare (eg: ['MZ','DZ','kraken+DZ','kraken+MZ'])
    twininfo_full: dict containing twin/family information (see krakencoder/data.py)
    df_pred_retest: pandas dataframe containing predicted graph measures for retest data
    inputname: string (name of the input data)
    
    Returns:
    metric_comparison_array: pandas dataframe containing the comparison of graph measures between predicted and true
    """
    #this is the cutoff where > bootstrap_size gets resampled a bunch of times at bootstrap_size and then we get the mean
    #the idea is to make the comparison more fair for smaller sample sizes (eg: DZ twins only has 13 pairs)
    #network_retest_cc_bootstrap_size=40
    network_retest_cc_bootstrap_size=1e6 #just dont do this for now
    network_retest_cc_bootstrap_count=200
    
    network_retest_cc_permtest_count=1000
    
    network_retest_cc_dict={}
    network_retest_cc_pval_dict={}
    network_retest_cc_stdev_dict={}
    
    network_retest_r2_dict={}
    network_retest_r2_pval_dict={}
    network_retest_r2_stdev_dict={}
    
    retest_cc_self_dict={}
    retest_cc_other_dict={}

    
    subjects_test=[s for s in clean_subject_list(np.array(df_pred['subjects'])) if not s.endswith("_Retest")]
    subjects_retest=[s for s in clean_subject_list(np.array(df['subjects'])) if s.endswith("_Retest") and s.replace("_Retest","") in subjects_test]
    subjects_test_retest=np.concatenate([subjects_test,subjects_retest])
    
    if df_pred_retest is not None:
        subjidx_tmp=np.array([i for i,s in enumerate(clean_subject_list(df_pred_retest['subjects'])) if s.replace("_Retest","") in subjects_test])
        df_pred=pd.concat([df_pred,df_pred_retest.iloc[subjidx_tmp]])

    subjidx_tmp=np.array([i for i,s in enumerate(clean_subject_list(df['subjects'])) if s in subjects_test_retest])
    df=df.iloc[subjidx_tmp]
    
    conntype=df['conntype'].iloc[0]
    fusiontype=inputname
    subjects_pred=np.array([s for s in clean_subject_list(np.array(df_pred['subjects']))])
    
    metric_names=df.keys()
    metric_names=[m for m in metric_names if not m in ['Unnamed: 0','nodes']]
    metric_names=[m for m in metric_names if not m in ['subjects','conntype','inputtype']]
    metric_names=np.array(metric_names)
    
    metric_comparison_array=[]
    
    knum=len(metric_names)

    #prediction_type='MZ'
    for prediction_type in prediction_type_list:
        
        #subjects_full=np.array([s for s in conndata[conntype]['subjects']])
        #subjects_full=np.array(df['subjects'])
        subjects_full=subjects_test_retest
                
        if prediction_type == 'kraken':

            #subjmask=np.array([s in subjects_full for s in subjects_pred])
            subjmask=np.array([s in subjects_full and not s.endswith("_Retest") for s in subjects_pred])
            subjects=subjects_pred[subjmask]
            subjidx=[np.where(subjects_full==s)[0][0] for s in subjects]

            numsubj_orig=len(subjidx)

            df_new=df.iloc[subjidx]
            df_pred_new=df_pred.iloc[subjmask]
            
            df_new=pd.concat([df_new,df_pred_new])

            #predict_cc=np.corrcoef(Ctri[subjidx,:],Ctri_pred[subjmask])[:numsubj_orig,numsubj_orig:]
        
        elif prediction_type.startswith('kraken+'):

            reltype=prediction_type.split('+')[1]
            
            if reltype=='retest':
                subjects=np.array([s.replace("_Retest","") for s in np.array(clean_subject_list(df_pred['subjects'])) if s.endswith("_Retest")])
            else:
                subjects=np.array([s for s in np.array(clean_subject_list(df_pred['subjects'])) if not s.endswith("_Retest")])

            
            subjects_retest, subjmask = get_matching_subject(subjects, match_type=reltype, twininfo=twininfo_full, maximum_age_difference=3, match_sex=True)
            subjects=subjects[subjmask]
            subjects_retest=subjects_retest[subjmask]
            
            #get rid of reciprocal pairs
            subj_together=np.stack([subjects,subjects_retest],axis=-1)
            subj_together=np.sort(subj_together,axis=1)
            subj_together=np.unique(subj_together,axis=0)
            subjects,subjects_retest=subj_together[:,0],subj_together[:,1]
            
            numsubj_orig=len(subjects)
            subjects=np.append(subjects,subjects_retest)
            
            #indexes if test+retest in original df
            subjidx=[np.where(subjects_full==s)[0][0] for s in subjects]

            #predict_cc=np.corrcoef(Ctri_pred[subjidx])[:numsubj_orig,numsubj_orig:]

            df_new=df_pred.iloc[subjidx]

        else:
            if prediction_type == 'retest':
                subjects=np.array([s.replace("_Retest","") for s in np.array(df['subjects']) if s.endswith("_Retest")])
                subjects=np.array([s for s in subjects if not s.startswith('143325')]) #this subject has issues?
            else:
                #subjects=np.array([s for s in conndata[conntype]['subjects'] if not s.endswith("_Retest")])
                subjects=np.array([s for s in np.array(df['subjects']) if not s.endswith("_Retest")])
            
            #if prediction_type == 'MZ':
            #    subjects=np.array([s.replace("_Retest","") for s in conndata[conntype]['subjects'] if s.endswith("_Retest")])
            
            subjects_retest, subjmask = get_matching_subject(subjects, match_type=prediction_type, twininfo=twininfo_full, maximum_age_difference=3, match_sex=True)

            subjects=subjects[subjmask]
            subjects_retest=subjects_retest[subjmask]
            
            #get rid of reciprocal pairs
            subj_together=np.stack([subjects,subjects_retest],axis=-1)
            subj_together=np.sort(subj_together,axis=1)
            subj_together=np.unique(subj_together,axis=0)
            subjects,subjects_retest=subj_together[:,0],subj_together[:,1]
            
            numsubj_orig=len(subjects)
            subjects=np.append(subjects,subjects_retest)
            
            #indexes if test+retest in original df
            subjidx=[np.where(subjects_full==s)[0][0] for s in subjects]

            #predict_cc=np.corrcoef(Ctri[subjidx])[:numsubj_orig,numsubj_orig:]

            df_new=df.iloc[subjidx]

        #retest_cc_self_dict[prediction_type]=np.mean(np.diag(predict_cc))
        #retest_cc_other_dict[prediction_type]=np.mean(predict_cc[np.eye(predict_cc.shape[0])==0])

        print("%s: numsubj_orig: %d" % (prediction_type,numsubj_orig))

        #network_scale_fit=np.ones((knum,2))*np.nan
        network_scale_fit=None
        network_retest_cc=np.ones(knum)*np.nan
        network_retest_cc_pval=np.ones(knum)*np.nan
        network_retest_cc_stdev=np.ones(knum)*np.nan

        network_retest_r2=np.ones(knum)*np.nan
        network_retest_r2_stdev=np.ones(knum)*np.nan
        network_retest_r2_pval=np.ones(knum)*np.nan
        
        
        #for i,(k,v) in enumerate(df_new.items()):
        for i,k in enumerate(metric_names):
            v=np.array(df_new[k])
            x=np.array(v[:numsubj_orig])
            y=np.array(v[numsubj_orig:])
            
            if len(x)>0 and not np.all(x==x[0]):
                if network_retest_cc_bootstrap_size is None or len(x)<=network_retest_cc_bootstrap_size:
                    #cc=[np.corrcoef(x=x,y=y)[0,1]]
                    #cc=[scipy.stats.spearmanr(x,y).correlation]
                    cc_result=scipy.stats.spearmanr(x,y)
                    cc=[cc_result.correlation]
                    ccp=[cc_result.pvalue]
                    
                    r2list=[r2_score(x,y)]
                    r2p=[0]
                    #cc=corravgrank(cc=np.abs(np.subtract.outer(x,y)),sort_descending=False)
                else:
                    #for comparisons with many pairs, use many subsets of a smaller size to approximate what the cc would be for that small size
                    #, to make it more comparable to what we see for those smaller sizes.
                    cc=np.zeros(network_retest_cc_bootstrap_count)
                    r2list=np.zeros(network_retest_cc_bootstrap_count)
                    for iboot in range(network_retest_cc_bootstrap_count):
                        ridx=np.argsort(np.random.rand(len(x)))[:network_retest_cc_bootstrap_size]
                        #cc[iboot]=np.corrcoef(x=x[ridx],y=y[ridx])[0,1]
                        cc[iboot]=scipy.stats.spearmanr(x[ridx],y[ridx]).correlation
                        #cc[iboot]=r2_score(x[ridx],y[ridx])
                        #cc[iboot]=corravgrank(cc=np.abs(np.subtract.outer(x[ridx],y[ridx])),sort_descending=False)
                        r2list[iboot]=r2_score(x[ridx],y[ridx])
                
                if network_retest_cc_permtest_count is not None:
                    cc_perm=np.zeros(network_retest_cc_permtest_count)
                    r2_perm=np.zeros(network_retest_cc_permtest_count)
                    for iperm in range(network_retest_cc_permtest_count):
                        ridx=np.argsort(np.random.rand(len(x)))
                        cc_perm[iperm]=scipy.stats.spearmanr(x[ridx],y).correlation
                        r2_perm[iperm]=r2_score(x[ridx],y)
                    #ccp_perm=[np.mean(np.abs(cc_perm)>np.abs(np.mean(cc)))]
                    ccp=[np.mean(np.abs(cc_perm)>np.abs(np.mean(cc)))]
                    r2p=[np.mean(np.abs(r2_perm)>np.abs(np.mean(r2list)))]
                    
                network_retest_cc[i]=np.mean(cc)
                network_retest_cc_stdev[i]=np.std(cc)
                network_retest_cc_pval[i]=ccp[0]
                network_retest_r2[i]=np.mean(r2list)
                network_retest_r2_stdev[i]=np.std(r2list)
                network_retest_r2_pval[i]=r2p[0]
        network_retest_cc_dict[prediction_type]=network_retest_cc
        network_retest_cc_pval_dict[prediction_type]=network_retest_cc_pval
        network_retest_cc_stdev_dict[prediction_type]=network_retest_cc_stdev
        
        network_retest_r2_dict[prediction_type]=network_retest_r2
        network_retest_r2_pval_dict[prediction_type]=network_retest_r2_pval
        network_retest_r2_stdev_dict[prediction_type]=network_retest_r2_stdev
        
    prediction_type_list=list(network_retest_cc_dict.keys())

    #scalar value for each: [famtype][fusiontype][conntype][metric]
    
    for p in prediction_type_list:
        for m in metric_names:
            metric_comparison_array+=[dict(conntype=conntype,fusiontype=fusiontype,prediction_type=p,metricname=m,
                                           corr=network_retest_cc_dict[p][metric_names==m][0],pvalue=network_retest_cc_pval_dict[p][metric_names==m][0],
                                           r2=network_retest_r2_dict[p][metric_names==m][0],r2_pvalue=network_retest_r2_pval_dict[p][metric_names==m][0]
                                           )]
        #metric_comparison_array+=[dict(conntype=conntype,fusiontype=fusiontype,prediction_type=p,metricname='corrself',corr=retest_cc_self_dict[p])]
        #metric_comparison_array+=[dict(conntype=conntype,fusiontype=fusiontype,prediction_type=p,metricname='corrother',corr=retest_cc_other_dict[p])]

    return pd.DataFrame(metric_comparison_array)

