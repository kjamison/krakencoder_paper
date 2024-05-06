"""
Functions for comparing similarity data between pairs of family levels.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from scipy import stats

from data import load_family_info

def get_family_data_similarity(Data, input_flavor, famlevels, subjidx=None, normalize_similarity=True, 
                               listcompare_max_agediff=1, stat_test_type='roc', violin_max_agediff=3, violin_samesex=True):
    """
    Return a dataframe of inter-subject similarity data for a given input flavor, for PAIRS of family levels
    For each pair of family levels, we generate two lists with matched pairs of family and non-family subjects (or MZ vs non-twin) 
    with matching age and sex. These are used as the INPUT to the statistical tests.
    
    Parameters:
    Data: dict. Data dictionary from load_study_data
    input_flavor: str. name of the input flavor to use
    famlevels: list of int. family levels to compare (unrelated=0, sibling=1, DZ=2, MZ=3, self=4, retest=5)
    subjidx: np.array. indices of subjects to use (default=None, use all subjects)
    normalize_similarity: bool. whether to normalize the similarity matrix by the mean/std of unrelated subjects (default=True)
    listcompare_max_agediff: int. maximum age difference for matching subject pairs in other family levels (default=1)
    stat_test_type: str. type of statistical test to use (default='roc')
        * Note: this is only used to determine what kind of lists to return. The tests are not performed in this function.
    violin_max_agediff: int. maximum age difference for matching subjects in the violin plots (default=3)
    violin_samesex: bool. whether to match sex for pairs when the violin plots (default=True)
    
    Returns:
    df_fam_performance: pd.DataFrame. dataframe of similarity data for each pair of family levels
    """
    if subjidx is None:
        if 'subject_splits' in Data and Data['subject_splits'] is not None:
            subjidx=Data['subject_splits']['subjidx_test']
        else:
            subjidx=np.arange(len(Data['Tdemo']))
    
    faminfo = load_family_info(studyname=Data['studyname'], Tdemo=Data['Tdemo'].copy(), twininfo_file=Data['twininfo_file'], Data_subjects=Data['subjects'],subjidx=subjidx)
    
    famlevel_dict=faminfo['famlevel_dict']
    
    conntype=input_flavor

    datatype='latent'
    datatype_full=datatype
    if Data['rawdata_sim'] is not None:
        datatype='raw'
        datatype_full=datatype+'='+Data['rawdata_sim']
    elif Data['prediction_sim'] is not None:
        datatype='raw'
        datatype_full=datatype+'_pred='+Data['prediction_sim']
    elif Data['latentdata_sim'] is not None:
        datatype='latent'
        datatype_full=datatype+"="+Data['latentdata_sim']
        
    Tdemo=faminfo['Tdemo']
    subjidx=faminfo['subjidx']
    Xfamlevel=faminfo['Xfamlevel']
    Xagediff=faminfo['Xagediff']
    Xsexmatch=faminfo['Xsexmatch']
        
    #dont include self as a family pair
    np.fill_diagonal(Xfamlevel,famlevel_dict['bad'])

    age=Tdemo['Age'].to_numpy()
    sex=Tdemo['Gender'].to_numpy()=='M'
    
    Xsim=None
    if input_flavor == 'encFC':
        Xsim=Data['enc_FCmean']
    elif input_flavor == 'encSC':
        Xsim=Data['enc_SCmean']
    elif input_flavor == 'encFCSC':
        Xsim=(Data['enc_FCmean']+Data['enc_SCmean'])/2
    else:
        Xsim=Data['data_alltypes'][input_flavor]
    
        #Data['data_alltypes']['fs86_ifod2act_volnorm'].shape
    np.fill_diagonal(Xsim,np.nan)
    
    Xsim=Xsim[subjidx,:][:,subjidx] #only keep the subjects we're using (eg: subjidx_test)
    
    #normalize each similarity matrix by the mean/std for unrelated subjectcs
    #Xsim=(Xsim-np.nanmean(Xsim[Xfamlevel==0]))/np.nanstd(Xsim[Xfamlevel==0])
    normmask=Xfamlevel==0
    normmask=normmask & (Xagediff<=violin_max_agediff)
    if violin_samesex:
        normmask=normmask & Xsexmatch
        
    mu_norm=np.nanmean(Xsim[normmask])
    sigma_norm=np.nanstd(Xsim[normmask])
    #sigma_norm=1
    #print(mu_norm,sigma_norm)
    if normalize_similarity:
        Xsim=(Xsim-mu_norm)/sigma_norm
    
    df_fam_performance=pd.DataFrame()

    for ilev0, famlevel0 in enumerate(famlevels):
        for ilev, famlevel in enumerate(famlevels):
            if ilev==0 and ilev0==0:
                #allow ilev,ilev0=0,0 so we get data for unrel violin
                pass
            elif ilev<=ilev0:
                continue
            
            ifam,jfam=np.where(Xfamlevel==famlevel)
                    
            xfam=np.nan*np.ones(len(ifam))
            xother=np.nan*np.ones(len(ifam))
            xother_list=[]
            xfam_samesex=np.zeros(len(ifam)).astype(bool)
            xfam_agediff=np.zeros(len(ifam))
            
            if stat_test_type=='list':
                for c,(i,j) in enumerate(zip(ifam,jfam)):
                    xfam[c]=Xsim[i,j]
                    xfam_samesex[c]=sex[i]==sex[j]
                    xfam_agediff[c]=np.round(np.abs(age[i]-age[j]))
                    if ilev == ilev0:
                        continue
                    #find other non-family members who match the age and sex of this subject's sibling
                    nonsibmatch=(Xfamlevel[i,:]==famlevel0) & (np.abs(age-age[j])<=listcompare_max_agediff) & (sex==sex[j])
                    xother_list.append(Xsim[i,nonsibmatch])
                    if any(~np.isnan(Xsim[i,nonsibmatch])):
                        xother[c]=np.nanmean(Xsim[i,nonsibmatch])
                    #xother[c]=np.nanmedian(Xsim[i,nonsibmatch])
                
                #xother_frac=[np.mean(xother_list[c]>xfam[c]) for c in range(len(xother_list))]    
                xother_frac=[np.mean(xother_list[c]>xfam[c]) if len(xother_list[c])>0 else np.nan for c in range(len(xother_list))]
                
            elif stat_test_type=='sibling_list':
                #########
                if ilev==0 and ilev0==0:
                    #special mode for sibling-only comparison (HCPDev), where the unrelated pairs are matched to the sibling pairs
                    #so skip this unrelated-only output for statistics
                    continue
                for c,(i,j) in enumerate(zip(ifam,jfam)):
                    xfam[c]=Xsim[i,j]
                    #find other non-family members who match the age and sex of this subject's sibling
                    nonsibmatch=(Xfamlevel[i,:]==0) & (np.abs(age-age[j])<=listcompare_max_agediff) & (sex==sex[j])
                    xother_list.append(Xsim[i,nonsibmatch])
                    xother[c]=np.nanmean(Xsim[i,nonsibmatch])
                    #xother[c]=np.nanmedian(Xsim[i,nonsibmatch])
                    xfam_samesex[c]=sex[i]==sex[j]
                    xfam_agediff[c]=np.round(np.abs(age[i]-age[j]))

                xother_frac=[np.mean(xother_list[c]>xfam[c]) for c in range(len(xother_list))]
            else:
                #xfam=Xsim[ifam,jfam]
                #xfam_samesex=Xsexmatch[ifam,jfam]
                #xfam_agediff=Xagediff[ifam,jfam]
                mask=np.ones(ifam.shape)>0
                mask=mask & (Xagediff[ifam,jfam]<=violin_max_agediff)
                if violin_samesex:
                    mask=mask & Xsexmatch[ifam,jfam]
                xfam=Xsim[ifam[mask],jfam[mask]]
                xfam_samesex=Xsexmatch[ifam[mask],jfam[mask]]
                xfam_agediff=Xagediff[ifam[mask],jfam[mask]]
                xother_frac=[np.nan]
            
            perf_tmp={'conntype':conntype,'datatype':datatype,'datatype_full':datatype_full,'famlevel':famlevel,'otherlevel':famlevel0,
                        'sibling_similarity':xfam,'nonsibling_similarity':xother, 
                        'nonsibling_similarity_list':xother_list,'nonsibling_similarity_frac':xother_frac,'nonsibling_similarity_frac_mean':np.mean(xother_frac),
                        'samesex':xfam_samesex, 'agediff':xfam_agediff}
            
            print(famlevel,famlevel0,conntype,datatype,np.mean(xother_frac))
            df_fam_performance=pd.concat((df_fam_performance,pd.DataFrame([perf_tmp])),ignore_index=True)
            
    return df_fam_performance
    

def compare_family_data_similarity(df_fam_performance, input_flavor, famlevels, stat_test_type='roc', auc_pval_nsamples=1000):
    """
    Compare the similarity data for each pair of family levels using a statistical test.
    By default, we use stat_test_type='roc', which performs a permutation test to compare the AUC of the ROC curves
    for paired groups of subjects of different family levels.
    
    Parameters:
    df_fam_performance: pd.DataFrame. dataframe of similarity data for each pair of family levels (from get_family_data_similarity)
    input_flavor: str. name of the input flavor to compare (eg: 'encFC', 'encSC', 'encFCSC', 'FCcorr_fs86_hpf')
    famlevels: list of int. family levels to compare (unrelated=0, sibling=1, DZ=2, MZ=3, self=4, retest=5)
    stat_test_type: str. type of statistical test to use (default='roc')
    auc_pval_nsamples: int. number of permutations to use for the ROC test (default=1000)
    
    Returns:
    stat_list: list of dict. list of statistical test results for each pair of family levels
    Each dict in this list contains the following keys:
        conntype: str. name of the input flavor
        famlevel: int. family level of the first group
        otherlevel: int. family level of the second group
        stat: stats object. statistical test object (not used for ROC test)
        value: float. value of the test statistic (eg: mean AUC difference)
        pvalue: float. uncorrected p-value of the test
        auc_pval_nsamples: int. number of permutations used for the ROC test
    """
    conntype=input_flavor
    stat_list=[]
    #compare latent vs raw by family similarity level
    for ilev0, famlevel0 in enumerate(famlevels):
        for ilev, famlevel in enumerate(famlevels):
            if ilev<=ilev0:
                continue
            connmask=df_fam_performance['conntype']==conntype
            if 'latent' in df_fam_performance['datatype'] and 'raw' in df_fam_performance['datatype']:
                latmask=df_fam_performance['datatype']=='latent'
                rawmask=df_fam_performance['datatype']=='raw'
            elif len(df_fam_performance['datatype_full'].unique())==2:
                utype=df_fam_performance['datatype_full'].unique()
                latmask=df_fam_performance['datatype_full']==utype[0]
                rawmask=df_fam_performance['datatype_full']==utype[1]
                print("latent:", utype[0], ", raw:", utype[1])
            else:
                print("Too many datatypes in family performance dataframe:", df_fam_performance['datatype_full'].unique())

            #predmask=df_fam_performance['datatype']=='pred'
            relmask=(df_fam_performance['famlevel']==famlevel) & (df_fam_performance['otherlevel']==famlevel0)
            
            
            if stat_test_type=='list':
                xfam_latent=df_fam_performance[connmask & latmask & relmask]['sibling_similarity'].to_list()[0]
                xother_latent=df_fam_performance[connmask & latmask & relmask]['nonsibling_similarity_list'].to_list()[0]
                xfam_raw=df_fam_performance[connmask & rawmask & relmask]['sibling_similarity'].to_list()[0]
                xother_raw=df_fam_performance[connmask & rawmask & relmask]['nonsibling_similarity_list'].to_list()[0]
                #xfam_samesex=df_sibling_performance[(df_sibling_performance['conntype']==conntype) & (df_sibling_performance['datatype']=='raw')]['samesex'].to_list()[0]
                
                #loop through every sibling pair. what rank percentile is my sibling more similar than matching non-siblings? (1=sibling is top, 0=sibling is bottom)
                #xfam_frac=np.array([[np.mean(xfam_latent[i]>xother_latent[i]), np.mean(xfam_raw[i]>xother_raw[i])] for i in range(len(xfam_latent)) if xfam_samesex[i]])
                
                xfam_frac_latent=[np.mean(xfam_latent[i]>xother_latent[i]) for i in range(len(xfam_latent))]
                xfam_frac_raw=[np.mean(xfam_raw[i]>xother_raw[i]) for i in range(len(xfam_latent))]

                stat=stats.ttest_rel(xfam_frac_latent,xfam_frac_raw)
                stat_val=stat.statistic
                stat_pval=stat.pvalue
                
            elif stat_test_type=='sibling_list':

                xfam_latent=df_fam_performance[connmask & latmask]['sibling_similarity'].to_list()[0]
                xother_latent=df_fam_performance[connmask & latmask]['nonsibling_similarity_list'].to_list()[0]
                xfam_raw=df_fam_performance[connmask & rawmask]['sibling_similarity'].to_list()[0]
                xother_raw=df_fam_performance[connmask & rawmask]['nonsibling_similarity_list'].to_list()[0]
                #xfam_samesex=df_sibling_performance[(df_sibling_performance['conntype']==conntype) & (df_sibling_performance['datatype']=='raw')]['samesex'].to_list()[0]
                
                #loop through every sibling pair. what rank percentile is my sibling more similar than matching non-siblings? (1=sibling is top, 0=sibling is bottom)
                #xfam_frac=np.array([[np.mean(xfam_latent[i]>xother_latent[i]), np.mean(xfam_raw[i]>xother_raw[i])] for i in range(len(xfam_latent)) if xfam_samesex[i]])
                xfam_frac_latent=[np.mean(xfam_latent[i]>xother_latent[i]) for i in range(len(xfam_latent))]
                xfam_frac_raw=[np.mean(xfam_raw[i]>xother_raw[i]) for i in range(len(xfam_latent))]
                stat=stats.ttest_rel(xfam_frac_latent,xfam_frac_raw)
                
                stat_val=stat.statistic
                stat_pval=stat.pvalue
                
            elif stat_test_type=='ttest':
                xfam_latent=df_fam_performance[connmask & latmask & relmask]['sibling_similarity'].to_list()[0]
                xother_latent=df_fam_performance[connmask & latmask & relmask]['nonsibling_similarity'].to_list()[0]
                xfam_raw=df_fam_performance[connmask & rawmask & relmask]['sibling_similarity'].to_list()[0]
                xother_raw=df_fam_performance[connmask & rawmask & relmask]['nonsibling_similarity'].to_list()[0]
                #xfam_samesex=df_sibling_performance[(df_sibling_performance['conntype']==conntype) & (df_sibling_performance['datatype']=='raw')]['samesex'].to_list()[0]
                
                #loop through every sibling pair. what rank percentile is my sibling more similar than matching non-siblings? (1=sibling is top, 0=sibling is bottom)
                #xfam_frac=np.array([[np.mean(xfam_latent[i]>xother_latent[i]), np.mean(xfam_raw[i]>xother_raw[i])] for i in range(len(xfam_latent)) if xfam_samesex[i]])
                
                xfam_frac_latent=[np.mean(xfam_latent[i]>xother_latent[i]) for i in range(len(xfam_latent))]
                xfam_frac_raw=[np.mean(xfam_raw[i]>xother_raw[i]) for i in range(len(xfam_latent))]

                stat=stats.ttest_rel(xfam_frac_latent,xfam_frac_raw)
                stat_val=stat.statistic
                stat_pval=stat.pvalue
                
            elif stat_test_type=='roc':
                #print(conntype, famlevel_str[famlevel]+"-"+famlevel_str[famlevel0])
                relmask=(df_fam_performance['famlevel']==famlevel) & (df_fam_performance['otherlevel']==famlevel0)
                relmask0=(df_fam_performance['famlevel']==famlevel0)
                
                
                xfam_latent=df_fam_performance[connmask & latmask & relmask]['sibling_similarity'].to_list()[0]
                xother_latent=df_fam_performance[connmask & latmask & relmask0]['sibling_similarity'].to_list()[0]
                xfam_raw=df_fam_performance[connmask & rawmask & relmask]['sibling_similarity'].to_list()[0]
                xother_raw=df_fam_performance[connmask & rawmask & relmask0]['sibling_similarity'].to_list()[0]
                
                #print(xfam_latent.shape)
                #print(xother_latent.shape)
                
                y_isfam=np.zeros(len(xfam_latent)+len(xother_latent))
                y_isfam[:len(xfam_latent)]=1
                
                aucdiff,aucpval, aucpval_abs=roc_permutation_test_between_models(y_isfam, 
                                                                    np.concatenate([xfam_latent,xother_latent]), 
                                                                    np.concatenate([xfam_raw,xother_raw]), nsamples=auc_pval_nsamples)
                
                stat=None
                stat_val=aucdiff
                stat_pval=aucpval_abs
                
                #print(conntype, famlevel_str[famlevel]+"-"+famlevel_str[famlevel0], "latent vs raw", aucdiff,aucpval)
            
            stat_list.append({'conntype':conntype,'famlevel':famlevel,'otherlevel':famlevel0,'stat':stat,'value':stat_val,'pvalue':stat_pval,'auc_pval_nsamples':auc_pval_nsamples})
    return stat_list

def roc_permutation_test_between_models(y_test, pred_proba_model1, pred_proba_model2, nsamples=1000):
    """
    Perform a permutation test to compare two paired ROC curves from two models (matching subjects), and test the significance of the hypothesis that
    AUC from model1 is larger than AUC from model2.
    
    We have a list of true labels for each subject (y_test), and a set of values from model1 (pred_proba_model1) 
    and from model2 (pred_proba_model1) that we expect to be larger for class 1 than class 0. We want to calculate the
    AUC for each model, and then compare those AUCs to tell whether the values are more separable in one model than the other.
    We test the significance of this by swapping the values from model1 and model2 many times to generate a null distribution
    of AUC differences.
    
    Parameters:
    y_test: np.array. [nsubj x 1] true binary labels
    pred_proba_1: np.array. [nsubj x 1] predicted values from model 1
    pred_proba_2: np.array. [nsubj x 1] predicted values from model 2
    nsamples: int. number of permutations to perform (default=1000)
    
    Returns:
    aucdiff: float. difference in AUC between models (no permutations)
    aucpval: float. permutation p-value for (AUC difference) > null (one-sided)
    aucpval_abs: float. permutation p-value for  abs(AUC difference) > abs(null) (two-sided)
    """
    
    auc1 = roc_auc_score(y_test.ravel(), pred_proba_model1.ravel())
    auc2 = roc_auc_score(y_test.ravel(), pred_proba_model2.ravel())
    observed_difference = auc1 - auc2
    
    #we break this up into blocks of at most 10000 permutations to reduce memory usage
    #if nsamples < 10000, just one block
    blocksize=10000
    if nsamples > blocksize:
        blocks=[blocksize]*(nsamples//blocksize)
        if nsamples%blocksize > 0:
            blocks.append(nsamples%blocksize)
    else:
        blocks=[nsamples]
    
    auc_differences_nullperm=[]
    for block in blocks:
        mask = np.random.randint(2, size=[block,len(pred_proba_model1.ravel())])
        p1stack=np.stack([pred_proba_model1.ravel()]*block)
        p2stack=np.stack([pred_proba_model2.ravel()]*block)
        p1 = np.where(mask, p1stack, p2stack)
        p2 = np.where(mask, p2stack, p1stack)
        y_test_flat=y_test.ravel()
        auc1=[roc_auc_score(y_test_flat,p1[i,:]) for i in range(block)]
        auc2=[roc_auc_score(y_test_flat,p2[i,:]) for i in range(block)]
        auc_differences_nullperm+=list(np.array(auc1)-np.array(auc2))
    
    auc_differences_nullperm=np.array(auc_differences_nullperm)
    
    aucpval = np.mean(auc_differences_nullperm >= observed_difference)
    aucpval_abs = np.mean(np.abs(auc_differences_nullperm) >= np.abs(observed_difference))
    
    return observed_difference, aucpval, aucpval_abs


def correct_family_similarity_pvals(stat_list, pval_test_info, input_flavors):
    """
    Correct the p-values in the statistical test results for multiple comparisons.
    If pval_test_info['correction_group']=='flavor', we correct the p-values within each flavor separately.
    Otherwise, we correct the p-values across all items in stat_list.
    
    Parameters:
    stat_list: list of dict. list of statistical test results for each pair of family levels (from compare_family_data_similarity)
    pval_test_info: dict. dictionary of information about the p-value correction method
        * correction_method: str. method to use for multiple comparison correction (eg: 'fdr_bh')
        * correction_group: str. group to use for multiple comparison correction (eg: 'flavor' or 'all')
    input_flavors: list of str. list of input flavors to correct
    
    Returns:
    stat_list: list of dict. list of statistical test results for each pair of family levels, with corrected p-values
    """
    if pval_test_info['correction_group']=='flavor':
        for conntype in input_flavors:
            #multiple comparison correction within flavor
            iconn=[i for i,v in enumerate(stat_list) if v['conntype']==conntype]
            pvals_uncorrected=[stat_list[i]['pvalue'] for i in iconn]
            pvals_corrected=multipletests(pvals_uncorrected,method=pval_test_info['correction_method'])[1]
            for ii,i in enumerate(iconn):
                stat_list[i]['pvalue_corrected']=pvals_corrected[ii]
    else:
        pvals_uncorrected=[v['pvalue'] for v in stat_list]
        pvals_corrected=multipletests(pvals_uncorrected,method=pval_test_info['correction_method'])[1]
        for i in range(len(stat_list)):
            stat_list[i]['pvalue_corrected']=pvals_corrected[i]

    #print("sibling_pval_corrected",sibling_pval_corrected)
    print("ttest uncorrected pval:",[v['pvalue'] for v in stat_list])
    print("ttest corrected pval:",[v['pvalue_corrected'] for v in stat_list])
    
    return stat_list
