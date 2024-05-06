import os
import sys
from datetime import datetime
from tqdm import tqdm
import time
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from utils import timestamp, krakendir, add_krakencoder_package
from data import get_matching_subject
from graph_measures import *

add_krakencoder_package() #add krakencoder package to path if not already there
from krakencoder.data import load_hcp_data, clean_subject_list, get_subjects_from_conndata, merge_conndata_subjects, get_hcp_data_flavors, canonical_data_flavor
from krakencoder.loss import xycorr, columncorr, corravgrank, corrtopNacc, corr_ident_parts

#compute avgrank, avgcorr, etc for measured vs predicted, or measured vs family, or predicted vs predicted.family
def predicted_connectome_performance(Xtrue, Xpred, subjects, prediction_type_list, model_info, twininfo_full):
    """
    Compute prediction performance metrics for a given set of connectomes and predictions
    
    Parameters:
    Xtrue: [Nsubj x Nedges] numpy array. Measured connectome data
    Xpred: [Nsubj x Nedges] numpy array. Predicted connectome data
    subjects: [Nsubj x 1] str array. Subjects in this dataset
    prediction_type_list: list of str. List of prediction types to compare
        * 'kraken': compare subject PREDICTED connectome to their MEASURED connectome
        * 'kraken+<fam>': Compare subject PREDICTED to a related subject's PREDICTED, where <fam> is one of ['unrelated','sibling','DZ','MZ','retest']
        * 'krakenXmeas+<fam>': Compare subject PREDICTED to a related subject's MEASURED, where <fam> is one of ['unrelated','sibling','DZ','MZ','retest']
        * '<fam>': Compare subject MEASURED to a related subject's MEASURED, where <fam> is one of ['unrelated','sibling','DZ','MZ','retest']
    model_info: dictionary with info about inputs and outputs (See generate_model_info_list). Just used to include this info in the output dataframe
    twininfo_full: Dict with twin and family information for all subjects (Loaded from twininfo_*.mat)
    
    Returns: pandas DataFrame with a row for each prediction_type and metric_name, with columns:
        * modeltype: str from ['kraken','sarwar','neudorf','data']. (from generate_model_info_list)
        * modelname: str from ['kraken<krakenmodelspec>','data','sarwar','neudorf']. (from generate_model_info_list)
        * conntype: output connectome flavor (eg FCcorr_fs86_hpf) (from generate_model_info_list)
        * fusiontype: input type for prediction: connectome flavor, or 'fusion*', or '' for modeltype='data' (from generate_model_info_list)
        * prediction_type: str from prediction_type_list
        * metricname: str from ['cc_self','cc_other','cc_top1acc','cc_avgrank', 'cc_resid_self','cc_resid_other','cc_resid_top1acc','cc_resid_avgrank']
        * corr: float. Value of the metric
    """
    #Xtrue, Xpred are both 236xN (includes test and retest)
    #subjects is 236x1 with both "12356" and "123456_Retest"
    
    #Get these from model_info just to pass them along to the output dataframe
    modeltype=model_info['type']
    intype=model_info['input']
    conntype=model_info['output']
    modeltype_str=model_info['modelname']
    
    subjidx_test=np.array([i for i,s in enumerate(subjects) if not s.endswith("_Retest")])
    y_test_mean=np.mean(Xtrue[subjidx_test,:],axis=0,keepdims=True)
    
    subjects_input=subjects.copy()
    
    prediction_metric_dict={}
    
    for prediction_type in prediction_type_list:
        metric_dict={}
        
        if prediction_type == 'kraken':
            #prediction vs measured. Not specific to krakencoder model. Could be other SC2FC model types
            
            #subjmask=np.array([s in subjects_full for s in subjects_pred])
            subjmask=np.array([not s.endswith("_Retest") for s in subjects_input])
            subjects=subjects_input[subjmask]
            subjidx=[np.where(subjects_input==s)[0][0] for s in subjects]

            numsubj_orig=len(subjidx)

            y_true_tri=Xtrue[subjidx]
            y_pred_tri=Xpred[subjidx]
            
        elif prediction_type.startswith('kraken+'):
            #prediction vs prediction for other subjects (eg MZ twin). Not specific to krakencoder model. Could be other SC2FC model types
            reltype=prediction_type.split('+')[1]
            
            if reltype=='retest':
                subjects=np.array([s.replace("_Retest","") for s in subjects_input if s.endswith("_Retest")])
            else:
                subjects=np.array([s for s in subjects_input if not s.endswith("_Retest")])
            
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
            subjidx=[np.where(subjects_input==s)[0][0] for s in subjects]
            
            y_true_tri=Xpred[subjidx][:numsubj_orig]
            y_pred_tri=Xpred[subjidx][numsubj_orig:]
            
        elif prediction_type.startswith('krakenXmeas+'):
            #prediction vs measured for other subjects (eg MZ twin). Not specific to krakencoder model. Could be other SC2FC model types
            reltype=prediction_type.split('+')[1]
            
            if reltype=='retest':
                subjects=np.array([s.replace("_Retest","") for s in subjects_input if s.endswith("_Retest")])
            else:
                subjects=np.array([s for s in subjects_input if not s.endswith("_Retest")])
            
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
            subjidx=[np.where(subjects_input==s)[0][0] for s in subjects]
            
            y_true_tri=Xtrue[subjidx][:numsubj_orig]
            y_pred_tri=Xpred[subjidx][numsubj_orig:]
        else:
            #measured vs measured for other subjects (eg MZ twin)
            if prediction_type == 'retest':
                subjects=np.array([s.replace("_Retest","") for s in subjects_input if s.endswith("_Retest")])
                subjects=np.array([s for s in subjects if not s.startswith('143325')]) #this subject has issues?
            else:
                #subjects=np.array([s for s in conndata[conntype]['subjects'] if not s.endswith("_Retest")])
                subjects=np.array([s for s in subjects_input if not s.endswith("_Retest")])
            
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
            subjidx=[np.where(subjects_input==s)[0][0] for s in subjects]
            
            y_true_tri=Xtrue[subjidx][:numsubj_orig]
            y_pred_tri=Xtrue[subjidx][numsubj_orig:]
            
        cc=xycorr(y_true_tri,y_pred_tri)
        #cc_resid=xycorr(y_true_tri-y_train_mean,y_pred_tri-y_train_mean)
        cc_resid=xycorr(y_true_tri-y_test_mean,y_pred_tri-y_test_mean) #this is what we use in heatmaps
        
        #for i_cc, cc_type in enumerate(['cc','cc_resid','cc.T','cc_resid.T']):
        for i_cc, cc_type in enumerate(['cc','cc_resid']):
            if cc_type == 'cc':
                cc_tmp=cc+0
            elif cc_type == 'cc_resid':
                cc_tmp=cc_resid+0
            elif cc_type == 'cc.T':
                cc_tmp=cc.T
            elif cc_type == 'cc_resid.T':
                cc_tmp=cc_resid.T
            
            cc_avgrank=corravgrank(cc=cc_tmp)
            cc_top1acc=corrtopNacc(cc=cc_tmp,topn=1)
            #cc_top2acc=corrtopNacc(cc=cc_tmp,topn=2)
            cc_self,cc_other=corr_ident_parts(cc=cc_tmp)
            
            metric_dict['%s_self' % (cc_type)]=cc_self
            metric_dict['%s_other' % (cc_type)]=cc_other
            metric_dict['%s_top1acc' % (cc_type)]=cc_top1acc
            metric_dict['%s_avgrank' % (cc_type)]=cc_avgrank
        prediction_metric_dict[prediction_type]=metric_dict
        
    prediction_type_list=list(prediction_metric_dict.keys())
    metric_names=list(prediction_metric_dict[prediction_type_list[0]].keys())
    
    metric_comparison_array=[]
    for p in prediction_type_list:
        for m in metric_names:
            metric_comparison_array+=[dict(modeltype=modeltype,modelname=modeltype_str,conntype=conntype,fusiontype=intype,prediction_type=p,metricname=m,corr=prediction_metric_dict[p][m])]
    
    return pd.DataFrame(metric_comparison_array)

#Quick functions to load HCP data and kraken prediction data into Ctri:
def load_conndata_tri(subjects_orig, conntype, badsubj=None, zero_edge_value_thresh=0):
    """
    Load the measured HCP connectome data for a given type and arrange it into the expected structure
    
    Parameters:
    subjects_orig: list of str. List of subjects to include in the analysis
    conntype: str. Name of the connectome flavor to load
    badsubj: list of str. List of subjects to exclude from the analysis (default=None)
    zero_edge_value_thresh: float. Threshold for setting very small edge values to zero (default=0)
    
    Returns:
    subjects_merged: [Nsubj x 1] str array. Subjects in the merged dataset
    Ctri: [Nsubj x Nedges] numpy array. Measured connectome data
    """
    subjects,conndata=load_hcp_data(subjects=[s for s in subjects_orig if not '_Retest' in s], conn_name_list=[conntype], load_retest=False, quiet=True)

    subjects_retest,conndata_retest=load_hcp_data(subjects=[s for s in subjects_orig if '_Retest' in s], conn_name_list=[conntype], load_retest=True, quiet=True)

    if badsubj is not None:
        conndata = get_subjects_from_conndata(conndata, remove_subjects=badsubj)
        conndata_retest = get_subjects_from_conndata(conndata_retest, remove_subjects=["%s_Retest" % (s) for s in badsubj])

    subjects_merged=np.concatenate([conndata[conntype]['subjects'], conndata_retest[conntype]['subjects']])
    conndata=merge_conndata_subjects([conndata,conndata_retest])
    #subjects_merged=conndata[conntype]['subjects']

    Ctri=conndata[conntype]['data']
    
    #set very tiny values to zero
    Ctri[np.abs(Ctri)<=zero_edge_value_thresh]=0
    
    #if > 25% of subjects have a zero here, make prediction zero
    edges_to_zero=np.mean(np.abs(Ctri)<=zero_edge_value_thresh,axis=0)>.25
    Ctri[:,edges_to_zero]=0 
    
    return subjects_merged, Ctri

def load_data_for_model_performance(modelinfo, subject_splits, zero_edge_value_thresh=0, badsubj=[]):
    """
    Load the data for a given model comparison.
    
    Parameters:
    modelinfo: dictionary with keys (From generate_model_info_list)
    subject_splits: dictionary with keys 'subjects','subjidx_train','subjidx_test','subjidx_retest' (and possibly 'subjidx_bad')
    zero_edge_value_thresh: float. Threshold for setting very small edge values to zero (default=0)
    badsubj: list of str. List of subjects to exclude from the analysis (will add to any bad subjects in subject_splits['subjidx_bad'])
    
    Returns:
    Ctri: [Nsubj x Nedges] numpy array. Measured connectome data
    Ctri_pred: [Nsubj x Nedges] numpy array. Predicted connectome data
    subjects_merged: [Nsubj x 1] str array. Subjects in the merged dataset
    subjidx_test: [Ntest x 1] numpy array. Indexes of test subjects in the merged dataset
    subjidx_retest: [Nretest x 1] numpy array. Indexes of retest subjects in the merged dataset
    """
    modeltype=modelinfo['type']
    predfile=modelinfo['testfile']
    predfile_retest=modelinfo['retestfile']
    intype=modelinfo['input']
    conntype=modelinfo['output']
    modeltype_str=modelinfo['modelname']
    
    if 'subjidx_bad' in subject_splits:
        badsubj=badsubj+subject_splits['subjects'][subject_splits['subjidx_bad']]
    
    if modeltype=='data':
        #subjects_pred=clean_subject_list(np.array(subject_splits['subjects'][subject_splits['subjidx_test']]))
        subj_tmp=clean_subject_list(subject_splits['subjects'])
        subj_tmp_retest=clean_subject_list(np.array(["%s_Retest" % (s)  for s in subject_splits['subjects'][subject_splits['subjidx_retest']]]))
        subjects_pred=np.concatenate((subj_tmp,subj_tmp_retest))
        subjects_pred, Ctri_pred=load_conndata_tri(subjects_pred, conntype, badsubj=badsubj, zero_edge_value_thresh=zero_edge_value_thresh)
    else:
        M=loadmat(predfile,simplify_cells=True)
        
        if modeltype=='sarwar':
            Ctri_pred=M['predicted']
            subjects_pred=clean_subject_list(np.array(subject_splits['subjects'][subject_splits['subjidx_test']]))
        else:
            Ctri_pred=M['predicted_alltypes'][intype][conntype]
            subjects_pred=clean_subject_list(M['subjects'])
        
        goodsubj_idx=np.array([i for i,s in enumerate(subjects_pred) if not s in badsubj])
        Ctri_pred=Ctri_pred[goodsubj_idx,:]
        subjects_pred=subjects_pred[goodsubj_idx]
        
        if predfile_retest and os.path.exists(predfile_retest):
            M_retest=loadmat(predfile_retest,simplify_cells=True)
            subjects_retest=np.array([s+'_Retest' for s in clean_subject_list(np.array(subject_splits['subjects'][subject_splits['subjidx_retest']]))])
        
            subjidx_tmp=np.array([i for i,s in enumerate(subjects_retest) if s.replace("_Retest","") in subjects_pred])
            if modeltype=='sarwar':
                Ctri_pred_retest=M_retest['predicted'][subjidx_tmp,:]
            else:
                Ctri_pred_retest=M_retest['predicted_alltypes'][intype][conntype][subjidx_tmp,:]
            Ctri_pred=np.concatenate([Ctri_pred,Ctri_pred_retest])
            subjects_pred=np.concatenate([subjects_pred,subjects_retest[subjidx_tmp]])
        
    if modeltype=='neudorf':
        #in neudorf per subject z-score: FC = ((FC-mean)/std)/maxabs
        _, conndata_alltypes = load_hcp_data(subjects=subject_splits['subjects'], conn_name_list=[conntype], quiet=False, keep_diagonal=False)
        y_train=conndata_alltypes[conntype]['data'][subject_splits['subjidx_train'],:]
        y_train_subjmean_mean=np.mean(np.mean(y_train,axis=1,keepdims=True))
        y_train_subjstdev_mean=np.mean(np.std(y_train,axis=1,keepdims=True))
        
        y_train_z=y_train-np.mean(y_train,axis=1,keepdims=True)
        y_train_z=y_train/np.std(y_train_z,axis=1,keepdims=True)
        y_train_subjmaxabs_mean=np.mean(np.max(np.abs(y_train_z),axis=1,keepdims=True))
        
        #y_test_mean=np.mean(conndata_alltypes[conntype]['data'][subject_splits['subjidx_test'],:],axis=0,keepdims=True)
        #y_train_mean=np.mean(y_train,axis=0,keepdims=True)
        Ctri_pred=Ctri_pred*y_train_subjmaxabs_mean*y_train_subjstdev_mean + y_train_subjmean_mean
        
    subjects_merged, Ctri=load_conndata_tri(subjects_pred, conntype, badsubj=badsubj, zero_edge_value_thresh=zero_edge_value_thresh)
    #find subjects in common between predicted and true, and keep only those in both
    subjects_common=[s for s in subjects_pred if s in subjects_merged]
    subjidx_tmp=np.array([i for i,s in enumerate(subjects_pred) if s in subjects_common])
    
    subjects_pred=subjects_pred[subjidx_tmp]
    Ctri_pred=Ctri_pred[subjidx_tmp,:]

    subjidx_tmp=np.array([i for i,s in enumerate(subjects_merged) if s in subjects_common])
    subjects_merged=subjects_merged[subjidx_tmp]
    Ctri=Ctri[subjidx_tmp,:]
    
    assert all([i==j for i,j in zip(subjects_pred,subjects_merged)]), "subjects_pred and subjects_merged don't match"
    
    subjidx_test=np.array([i for i,s in enumerate(subjects_merged) if not "_Retest" in s])
    subjidx_retest=np.array([i for i,s in enumerate(subjects_merged) if "_Retest" in s])
    
    return Ctri, Ctri_pred, subjects_merged, subjidx_test, subjidx_retest
        
def generate_model_info_list(model_set=['kraken','sarwar','neudorf']):
    """
    Generate a list of hardcoded data files for a desired comparison.
    
    model_set: str or list of str. Name of the data we want to compute metrics for
        * 'kraken': SC2FC predictions from Kraken model. 
            * 6 connectomes/atlas = (sdstream,ifod2act,fusionSC) -> (FCcorr,FCpcorr)
        * 'sarwar': SC2FC predictions from Sarwar deepnet model. 
            * 4 connectomes/atlas = (sdstream,ifod2act) -> (FCcorr,FCpcorr)
        * 'neudorf': SC2FC predictions from Neudorf graphnet model. 
            * 4 connectomes/atlas = (sdstream,ifod2act) -> (FCcorr,FCpcorr)
        * 'allkraken': Additional Kraken predictions
            * 30 connectomes/atlas = (fusion,fusionFC,fusionSC,fusion.noatlas,fusionFC.noatlas,fusionSC.noatlas) -> (FCcorr,FCcorr_gsr,FCpcorr,SCsdstream,SCifod2act)
        * 'data': Raw HCP data for all 15 flavors (just need to generate graph measures for these)
            * 5 connectomes/atlas = (FCcorr,FCcorr_gsr,FCpcorr,SCsdstream,SCifod2act)
            
    Returns: list of dictionaries with model info with keys:
        * type: 'kraken','sarwar','neudorf','data'
        * input: input connectome type (input connectome flavor, or 'fusion*', or '' for 'data')
        * output: output connectome type (eg FCcorr_fs86_hpf)
        * testfile: path to file with predicted connectomes
        * retestfile: path to file with predicted connectomes for RETEST subjects
        * modelname: name of the model ('data','sarwar','neudorf','kraken<krakenmodelspec>'
    """
    if os.path.exists('/Users/kwj5/Research/krakencoder'):
        krakenmodel_root='/Users/kwj5/Research/krakencoder'
        sarwar_root='/Users/kwj5/Research/krakencoder/sarwar_mapping_SC_FC'
        neudorf_root='/Users/kwj5/Research/krakencoder/neudorf_graphnets'
    elif os.path.exists('/midtier/sablab/scratch/kwj2001/krakencoder'):
        #krakenmodel_root='/midtier/sablab/scratch/kwj2001/krakencoder'
        krakenmodel_root='/midtier/cocolab/colossus/shared_data3/kwj2001/HCP_connae'
        sarwar_root='/midtier/sablab/scratch/kwj2001/sarwar_sc2fc_2024/mapping_SC_FC'
        neudorf_root='/midtier/sablab/scratch/kwj2001/neudorf_graphnets'
    
    if isinstance(model_set,str):
        model_set=[model_set]
    
    model_info=[]
    if 'data' in model_set:
        conntypes=[canonical_data_flavor(c) for c in get_hcp_data_flavors(fc_filter_list=["hpf"])]
        for conntype in conntypes:
            model_info+=[dict(type="data",input='',output=conntype,
                testfile='', retestfile='',modelname='data')]
    
    #################### sarwar fs86
    if 'sarwar' in model_set:
        modelname='sarwar'
        model_info+=[dict(type="sarwar",input='SCsdstream_fs86_volnorm',output='FCcorr_fs86_hpf',
                        testfile='%s/directory_sdstream_20240112_171636/predicted_model.ckpt-20000_scfc_86roi_sdstream_FChpf_203test.mat' % (sarwar_root),
                        retestfile='',modelname=modelname)]
        model_info+=[dict(type="sarwar",input='SCifod2act_fs86_volnorm',output='FCcorr_fs86_hpf',
                        testfile='%s/directory_ifod2act_20240112_171631/predicted_model.ckpt-20000_scfc_86roi_ifod2act_FChpf_203test.mat' % (sarwar_root),
                        retestfile='',modelname=modelname)]

        model_info+=[dict(type="sarwar",input='SCsdstream_fs86_volnorm',output='FCpcorr_fs86_hpf',
                        testfile='%s/directory_sdstreamFCpcorr_20240125_114918/predicted_model.ckpt-20000_scfc_86roi_sdstream_FCpc_hpf_203test.mat' % (sarwar_root),
                        retestfile='',modelname=modelname)]

        model_info+=[dict(type="sarwar",input='SCifod2act_fs86_volnorm',output='FCpcorr_fs86_hpf',
                        testfile='%s/directory_ifod2actFCpcorr_20240116_123450/predicted_model.ckpt-20000_scfc_86roi_ifod2act_FCpc_hpf_203test.mat' % (sarwar_root),
                        retestfile='',modelname=modelname)]

        #################### sarwar shen268
        model_info+=[dict(type="sarwar",input='SCsdstream_shen268_volnorm',output='FCcorr_shen268_hpf',
                        testfile='%s/directory_shen268_sdstream_20240114_215053/predicted_model.ckpt-20000_scfc_268roi_sdstream_FChpf_203test.mat' % (sarwar_root),
                        retestfile='',modelname=modelname)]

        model_info+=[dict(type="sarwar",input='SCifod2act_shen268_volnorm',output='FCcorr_shen268_hpf',
                        testfile='%s/directory_shen268_ifod2act_20240114_215059/predicted_model.ckpt-20000_scfc_268roi_ifod2act_FChpf_203test.mat' % (sarwar_root),
                        retestfile='',modelname=modelname)]


        model_info+=[dict(type="sarwar",input='SCsdstream_shen268_volnorm',output='FCpcorr_shen268_hpf',
                        testfile='%s/directory_shen268_sdstreamFCpcorr_20240125_114918/predicted_model.ckpt-20000_scfc_268roi_sdstream_FCpc_hpf_203test.mat' % (sarwar_root),
                        retestfile='',modelname=modelname)]

        model_info+=[dict(type="sarwar",input='SCifod2act_shen268_volnorm',output='FCpcorr_shen268_hpf',
                        testfile='%s/directory_shen268_ifod2actFCpcorr_20240115_212527/predicted_model.ckpt-20000_scfc_268roi_ifod2act_FCpc_hpf_203test.mat' % (sarwar_root),
                        retestfile='',modelname=modelname)]


        #################### sarwar coco439
        model_info+=[dict(type="sarwar",input='SCsdstream_coco439_volnorm',output='FCcorr_coco439_hpf',
                        testfile='%s/directory_coco439_sdstream_20240122_154753/predicted_model.ckpt-20000_scfc_439roi_sdstream_FChpf_203test.mat' % (sarwar_root),
                        retestfile='',modelname=modelname)]

        model_info+=[dict(type="sarwar",input='SCifod2act_coco439_volnorm',output='FCcorr_coco439_hpf',
                        testfile='%s/directory_coco439_ifod2act_20240122_154748/predicted_model.ckpt-20000_scfc_439roi_ifod2act_FChpf_203test.mat' % (sarwar_root),
                        retestfile='',modelname=modelname)]

        model_info+=[dict(type="sarwar",input='SCsdstream_coco439_volnorm',output='FCpcorr_coco439_hpf',
                        testfile='%s/directory_coco439_sdstreamFCpcorr_20240125_114918/predicted_model.ckpt-20000_scfc_439roi_sdstream_FCpc_hpf_203test.mat' % (sarwar_root),
                        retestfile='',modelname=modelname)]

        model_info+=[dict(type="sarwar",input='SCifod2act_coco439_volnorm',output='FCpcorr_coco439_hpf',
                        testfile='%s/directory_coco439_ifod2actFCpcorr_20240122_154748/predicted_model.ckpt-20000_scfc_439roi_ifod2act_FCpc_hpf_203test.mat' % (sarwar_root),
                        retestfile='',modelname=modelname)]
    
    if 'neudorf' in model_set:
        #################### neudorf fs86
        #neudorf fs86_sdstream->FCpcorr = Fri 7:40pm+22hrs = Saturday 5:40pm
        #neudorf shen268_sdstream->FCPcorr = Fri 7:40pm+73hrs = Monday 8:40pm
        modelname='neudorf'
        model_info+=[dict(type="neudorf",input='SCsdstream_fs86_volnorm',output='FCcorr_fs86_hpf',
                        testfile='%s/neudorf_predicted_203test_4000_epochs_SCsdstreamHPF_86roi_edges_86roi_dti_standardized_updated-1.mat' % (neudorf_root),
                        retestfile='',modelname=modelname)]

        model_info+=[dict(type="neudorf",input='SCifod2act_fs86_volnorm',output='FCcorr_fs86_hpf',
                            testfile='%s/neudorf_predicted_203test_2600_epochs_SCifod2actHPF_86roi_edges_86roi_dti_standardized_updated-1.mat' % (neudorf_root),
                            retestfile='',modelname=modelname)]

        model_info+=[dict(type="neudorf",input='SCsdstream_fs86_volnorm',output='FCpcorr_fs86_hpf',
                            testfile='%s/neudorf_predicted_203test_1900_epochs_SCsdstreampcorrHPF_86roi_edges_86roi_dti_standardized_updated-1.mat' % (neudorf_root),
                            retestfile='',modelname=modelname)]

        model_info+=[dict(type="neudorf",input='SCifod2act_fs86_volnorm',output='FCpcorr_fs86_hpf',
                            testfile='%s/neudorf_predicted_203test_3600_epochs_SCifod2actpcorrHPF_86roi_edges_86roi_dti_standardized_updated-1.mat' % (neudorf_root),
                            retestfile='',modelname=modelname)]

        #################### neudorf shen268
        model_info+=[dict(type="neudorf",input='SCsdstream_shen268_volnorm',output='FCcorr_shen268_hpf',
                            testfile='%s/neudorf_predicted_203test_1000_epochs_SCsdstreamHPF_268roi_edges_268roi_dti_standardized_updated-1.mat' % (neudorf_root),
                            retestfile='',modelname=modelname)]

        model_info+=[dict(type="neudorf",input='SCifod2act_shen268_volnorm',output='FCcorr_shen268_hpf',
                            testfile='%s/neudorf_predicted_203test_3200_epochs_SCifod2actHPF_268roi_edges_268roi_dti_standardized_updated-1.mat' % (neudorf_root),
                            retestfile='',modelname=modelname)]

        model_info+=[dict(type="neudorf",input='SCsdstream_shen268_volnorm',output='FCpcorr_shen268_hpf',
                            testfile='%s/neudorf_predicted_203test_2600_epochs_SCsdstreampcorrHPF_268roi_edges_268roi_dti_standardized_updated-1.mat' % (neudorf_root),
                            retestfile='',modelname=modelname)]

        model_info+=[dict(type="neudorf",input='SCifod2act_shen268_volnorm',output='FCpcorr_shen268_hpf',
                            testfile='%s/neudorf_predicted_203test_2500_epochs_SCifod2actpcorrHPF_268roi_edges_268roi_dti_standardized_updated-1.mat' % (neudorf_root),
                            retestfile='',modelname=modelname)]

        #################### neudorf coco439
        model_info+=[dict(type="neudorf",input='SCsdstream_coco439_volnorm',output='FCcorr_coco439_hpf',
                            testfile='%s/neudorf_predicted_203test_2300_epochs_SCsdstreamHPF_439roi_edges_439roi_dti_standardized_updated-1.mat' % (neudorf_root),
                            retestfile='',modelname=modelname)]

        model_info+=[dict(type="neudorf",input='SCifod2act_coco439_volnorm',output='FCcorr_coco439_hpf',
                            testfile='%s/neudorf_predicted_203test_5000_epochs_SCifod2actHPF_439roi_edges_439roi_dti_standardized_updated-1.mat' % (neudorf_root),
                            retestfile='',modelname=modelname)]

        model_info+=[dict(type="neudorf",input='SCsdstream_coco439_volnorm',output='FCpcorr_coco439_hpf',
                            testfile='%s/neudorf_predicted_203test_3000_epochs_SCsdstreampcorrHPF_439roi_edges_439roi_dti_standardized_updated-1.mat' % (neudorf_root),
                            retestfile='',modelname=modelname)]

        model_info+=[dict(type="neudorf",input='SCifod2act_coco439_volnorm',output='FCpcorr_coco439_hpf',
                            testfile='%s/neudorf_predicted_203test_2100_epochs_SCifod2actpcorrHPF_439roi_edges_439roi_dti_standardized_updated-1.mat' % (neudorf_root),
                            retestfile='',modelname=modelname)]
    
    #################### kraken
    # kraken fs86 (sc->FCcorr and FCpcorr)
    kraken_outputname='20230927_012518_ep002000_mse.w1000'
    #kraken_outputname='20230926_023057_ep002000'
    
    modelname='kraken'+kraken_outputname

    #kraken_predicted_format='HCP_20230927_012518_ep002000_mse.w1000_{split}_fusion_{output}.mat'
    kraken_predicted_format='hcp_20230927_012518_ep002000_{split}_{output}.mat'
    
    if 'kraken' in model_set:
        for atlas in ['fs86','shen268','coco439']:
            for fc in [f'FCcorr_{atlas}_hpf', f'FCpcorr_{atlas}_hpf']:
                for sc in [f'{atlas}_sdstream_volnorm',f'{atlas}_ifod2act_volnorm','fusionSC']:            
                    model_info+=[dict(type="kraken",input=sc,output=fc,
                                    testfile='%s/%s' % (krakenmodel_root,kraken_predicted_format.format(split='test',output=fc)),
                                    retestfile='',modelname=modelname)]
    
    if 'allkraken' in model_set:
        for atlas in ['fs86','shen268','coco439']:
            #for outtype in [f'FCcorr_{atlas}_hpfgsr', f'{atlas}_sdstream_volnorm',f'{atlas}_ifod2act_volnorm']:
            for outtype in [f'FCcorr_{atlas}_hpf', f'FCpcorr_{atlas}_hpf', f'FCcorr_{atlas}_hpfgsr', f'{atlas}_sdstream_volnorm',f'{atlas}_ifod2act_volnorm']:
                for intype in ['fusion','fusionFC','fusionSC','fusion.noatlas','fusionFC.noatlas','fusionSC.noatlas']:
                #for intype in ['fusion','fusionFC','fusionSC']:
                    model_info+=[dict(type="kraken",input=intype,output=outtype,
                                    testfile='%s/%s' % (krakenmodel_root,kraken_predicted_format.format(split='test',output=outtype)),
                                    retestfile='',modelname=modelname)]
    
    for i in range(len(model_info)):
        model_info[i]['retestfile']=model_info[i]['testfile'].replace('203test','42retest').replace("_test_","_retest_")
        
    return model_info

def run_prediction_performance(argv):
    """
    Main function called from CLI to run prediction performance analysis: Generate graph measures, compare predicted connectomes to measured connectomes, etc.
    """

    args=argument_parse_performance(argv)
    
    do_allkraken=args.allkraken
    do_only_rawdata=args.only_raw_data
    do_force_overwrite=args.force_overwrite
    do_only_compare=args.only_compare
    do_only_graph_measure_summary=args.only_graph_measure_summary
    
    do_force_overwrite_network_summary=do_force_overwrite
    do_force_overwrite_comparison=do_force_overwrite
    
    if do_only_compare:
        do_force_overwrite_network_summary=False
    
    do_only_connsim=args.connsim
    do_only_edgecorr=args.edgecorr
    do_krakenXmeas=args.krakenXmeas
    
    do_summary_timestamp=args.add_summary_timestamp
    
    network_stat_dir='%s/graph_measure_summary' % (krakendir())
    if args.statdir:
        network_stat_dir=args.statdir
    if not os.path.exists(network_stat_dir):
        os.makedirs(network_stat_dir)
        

    if do_only_rawdata:
        model_info = generate_model_info_list(['data'])
        do_only_graph_measure_summary=True
    elif do_allkraken:
        model_info = generate_model_info_list(['allkraken'])
    else:
        model_info = generate_model_info_list(['kraken','sarwar','neudorf'])
    
    df_info=pd.DataFrame(model_info)
    
    dfmask=np.ones(df_info.shape[0])>0
    if args.filter_output:
        for x in args.filter_output:
            dfmask=dfmask & (df_info['output'].str.contains(x))
    if args.filter_input:
        for x in args.filter_input:
            dfmask=dfmask & (df_info['input'].str.contains(x))
    if args.filter_type:
        for x in args.filter_type:
            dfmask=dfmask & (df_info['type'].str.contains(x))
    
    df_info=df_info[dfmask]
    
    print(df_info)

    timestamp=datetime.now()
    timestamp_suffix=timestamp.strftime("%Y%m%d_%H%M%S")

    zero_edge_value_thresh=1e-9
    keep_proportion_for_predicted=None
            
    twininfo_file='%s/twininfo_hcp_1206subj.mat' % (krakendir())
    twininfo_full=loadmat(twininfo_file,simplify_cells=True)
    twininfo_full['subject']=clean_subject_list(twininfo_full['subject'])
    twininfo_full['age']=twininfo_full['age'].astype(float)

    badsubj=clean_subject_list(np.loadtxt('%s/badsubjects_fmri_fromwiki_and_manual_35subj.txt' % (krakendir())))
    #badsubj=np.array(list(badsubj)+['143325']) #retest is questionable for this subject

    subject_splits=loadmat('%s/subject_splits_993subj_710train_80val_203test_retestInTest.mat' % (krakendir()),simplify_cells=True)
    subject_splits['subjects']=clean_subject_list(subject_splits['subjects'])
    #################
    
    prediction_type_list=['unrelated','sibling','DZ','MZ','retest','kraken']
    prediction_type_list+=['kraken+unrelated','kraken+sibling','kraken+DZ','kraken+MZ','kraken+retest']
    
    if do_krakenXmeas:
        prediction_type_list+=['krakenXmeas+unrelated','krakenXmeas+sibling','krakenXmeas+DZ','krakenXmeas+MZ','krakenXmeas+retest']
    
    #prediction_type_list=['unrelated','sibling','DZ','MZ','retest','kraken', 'kraken+unrelated','kraken+retest']

    predict_type_short={'unrelated':'unrel',
                        'sibling':'sib',
                        'DZ':'DZ','MZ':'MZ','retest':'retest'}
    
    #################
    df_network_summary_combined=pd.DataFrame()
    df_metrics_combined=pd.DataFrame()

    edgecorr_info=[]
    edgecorr_list=[]
    conn_similarity_list=[]
    
    for i in range(len(df_info)):
        d=df_info.iloc[i]
        
        modeltype=d['type']
        intype=d['input']
        conntype=d['output']
        modeltype_str=d['modelname']
        
        Ctri, Ctri_pred, subjects_merged, subjidx_test, subjidx_retest = load_data_for_model_performance(modelinfo=d, 
                                                                                                    subject_splits=subject_splits, 
                                                                                                    zero_edge_value_thresh=zero_edge_value_thresh, 
                                                                                                    badsubj=badsubj)
        
        print("")
        print("###################################")
        print("%s: %s->%s" % (d['type'],d['input'],d['output']))
        print("  %d test, %d retest" % (len(subjidx_test),len(subjidx_retest)))
        #print("predfile: %s" % (predfile))
        
        y_true_tri=Ctri
        y_pred_tri=Ctri_pred
        cc=xycorr(y_true_tri,y_pred_tri)
        #cc_resid=xycorr(y_true_tri-y_train_mean,y_pred_tri-y_train_mean)
        y_test_mean=np.mean(Ctri[subjidx_test,:],axis=0,keepdims=True)
        cc_resid=xycorr(y_true_tri-y_test_mean,y_pred_tri-y_test_mean) #this is what we use in heatmaps
        
        if do_only_connsim:
            df_connsim=predicted_connectome_performance(y_true_tri, y_pred_tri, subjects=subjects_merged, prediction_type_list=prediction_type_list, model_info=d, twininfo_full=twininfo_full)
            conn_similarity_list.append(df_connsim)
            print(df_connsim.to_string())
            continue
        
        if do_only_edgecorr:
            edge_corr=columncorr(y_true_tri[subjidx_test,:],y_pred_tri[subjidx_test,:],axis=0)
            edgecorr_list.append(edge_corr)
            edgecorr_info.append(dict(modeltype=modeltype,conntype=conntype,intype=intype))
            print(edgecorr_info[-1])
            print("data.shape:",y_true_tri[subjidx_test,:].shape)
            print("edge_corr.shape:",edge_corr.shape)
            print("edge_corr min,max,mean:",np.nanmin(edge_corr),np.nanmax(edge_corr),np.nanmean(edge_corr))
            continue
        
        #for i_cc, cc_type in enumerate(['cc','cc_resid','cc.T','cc_resid.T']):
        for i_cc, cc_type in enumerate(['cc','cc_resid']):
            if cc_type == 'cc':
                cc_tmp=cc+0
            elif cc_type == 'cc_resid':
                cc_tmp=cc_resid+0
            elif cc_type == 'cc.T':
                cc_tmp=cc.T
            elif cc_type == 'cc_resid.T':
                cc_tmp=cc_resid.T
            
            cc_tmp=cc_tmp[subjidx_test,:][:,subjidx_test]
            
            cc_avgrank=corravgrank(cc=cc_tmp)
            cc_top1acc=corrtopNacc(cc=cc_tmp,topn=1)
            cc_top2acc=corrtopNacc(cc=cc_tmp,topn=2)
            cc_self,cc_other=corr_ident_parts(cc=cc_tmp)
            
            #cc_avgrank=np.array(cc_avgrank)
            #cc_top1acc=np.array(cc_top1acc)
            #cc_top2acc=np.array(cc_top2acc)
            #cc_self=np.array(cc_self)
            #cc_other=np.array(cc_other)
            
            print("")
            print("%s, cc shape:" % (cc_type),cc_tmp.shape)
            print("%s self, other:" % (cc_type),cc_self,cc_other)
            print("%s avgrank:" % (cc_type),cc_avgrank)
            print("%s top1acc:" % (cc_type),cc_top1acc)
            print("%s top2acc:" % (cc_type),cc_top2acc)
            
            #add new column to df_info for this cc_type
            if not '%s_self' % (cc_type) in df_info:
                df_info['%s_self' % (cc_type)]=np.nan
                df_info['%s_other' % (cc_type)]=np.nan
                df_info['%s_avgrank' % (cc_type)]=np.nan
                df_info['%s_top1acc' % (cc_type)]=np.nan
            
            #put cc_self value into current row for this column
            
            df_info.at[i,'%s_self' % (cc_type)]=cc_self
            df_info.at[i,'%s_other' % (cc_type)]=cc_other
            df_info.at[i,'%s_top1acc' % (cc_type)]=cc_top1acc
            df_info.at[i,'%s_avgrank' % (cc_type)]=cc_avgrank
        
        outfile_netstat="%s/graph_measure_summary_%s_%s_%s.csv" % (network_stat_dir,modeltype_str,intype,conntype)    
        if do_summary_timestamp:
            outfile_netcompare="%s/graph_measure_comparison_%s_%s_%s_%s.csv" % (network_stat_dir,modeltype_str,intype,conntype,timestamp_suffix)
        else:
            outfile_netcompare="%s/graph_measure_comparison_%s_%s_%s.csv" % (network_stat_dir,modeltype_str,intype,conntype)
        
        if modeltype=='data':
            outfile_netstat="%s/graph_measure_summary_%s.csv" % (network_stat_dir,conntype)
        
        if os.path.exists(outfile_netstat) and not do_force_overwrite_network_summary:
            print("Loading: %s" % (outfile_netstat))
            df_pred=pd.read_csv(outfile_netstat)
        else:
            if do_force_overwrite_network_summary:
                print("force_overwrite: %s" % (outfile_netstat))
            else:
                print("not found: %s" % (outfile_netstat))

            tic=time.time()
            df_pred=graph_measure_summary(Ctri_pred, keep_proportion=keep_proportion_for_predicted)

            df_pred['subjects']=subjects_merged
            df_pred['conntype']=conntype
            df_pred['inputtype']=intype
            
            print("%dx%d inputs took %f seconds" % (*Ctri_pred.shape,time.time()-tic))

            #outfile="%s/graph_measure_summary_kraken%s_%s_retest.csv" % (network_stat_dir,pred,conntype)
            df_pred.to_csv(outfile_netstat)
            print("Saved %s" % (outfile_netstat))
        df_pred['model']=modeltype
        
        if do_only_graph_measure_summary:
            continue
        
        if os.path.exists(outfile_netcompare) and not do_force_overwrite_comparison:
            print("Loading: %s" % (outfile_netcompare))
            df_metrics=pd.read_csv(outfile_netcompare)
        else:
            
            csvfile_data="%s/graph_measure_summary_%s.csv" % (network_stat_dir,conntype)
            if os.path.exists(csvfile_data) and not do_force_overwrite_comparison:
                df=pd.read_csv(csvfile_data)
            else:
                if do_force_overwrite_network_summary:
                    print("force_overwrite: %s" % (csvfile_data))
                else:
                    print("not found: %s" % (csvfile_data))
                tic=time.time()
                df=graph_measure_summary(Ctri, keep_proportion=keep_proportion_for_predicted)

                df['subjects']=subjects_merged
                df['conntype']=conntype
                
                print("%dx%d inputs took %f seconds" % (*Ctri.shape,time.time()-tic))

                #outfile="%s/graph_measure_summary_kraken%s_%s_retest.csv" % (network_stat_dir,pred,conntype)
                df.to_csv(csvfile_data)
                print("Saved %s" % (csvfile_data))
            
            df_pred_retest=None
            df_metrics=graph_measure_summary_comparison(df,df_pred, df_pred_retest=df_pred_retest,inputname=intype, prediction_type_list=prediction_type_list, twininfo_full=twininfo_full)
            df_metrics.to_csv(outfile_netcompare)
            print("Saved %s" % (outfile_netcompare))

        
        df_metrics['model']=modeltype
        
        df_network_summary_combined=pd.concat([df_network_summary_combined,df_pred])
        df_metrics_combined=pd.concat([df_metrics_combined,df_metrics])
    
    if do_only_connsim:
        df_connsim=pd.concat(conn_similarity_list)
        connsimfile='%s/model_connsim_info_%s.csv' % (network_stat_dir,timestamp_suffix)
        df_connsim.to_csv(connsimfile)
        print("Saved %s" % (connsimfile))
        return

    if do_only_edgecorr:
        edgecorrfile='%s/model_edgecorr_info_%s.mat' % (network_stat_dir,timestamp_suffix)
        savemat(edgecorrfile,{'edgecorr_info':edgecorr_info,'edgecorr_list':np.array(edgecorr_list,dtype=object)},format='5',do_compression=True)
        print("Saved %s" % (edgecorrfile))
        return
    
    #print df_info excluding columns with testfile or retestfile in name:
    print(df_info[[c for c in df_info.columns if not 'testfile' in c and not 'modelname' in c]])

    print(df_network_summary_combined)

    infofile='%s/model_comparison_info_%s.csv' % (network_stat_dir,timestamp_suffix)
    df_info.to_csv(infofile)
    print("Saved %s" % (infofile))

def argument_parse_performance(argv):
    parser=argparse.ArgumentParser(description='Generate graph metrics, compare connectome predictions',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--filter_output',action='store',dest='filter_output', nargs='*', help='list of strings to apply to filter output')
    parser.add_argument('--filter_input',action='store',dest='filter_input', nargs='*', help='list of strings to apply to filter input')
    parser.add_argument('--filter_type',action='store',dest='filter_type', nargs='*', help='list of strings to apply to filter type')
    parser.add_argument('--force_overwrite',action='store_true',dest='force_overwrite', help='force overwrite of existing files')
    parser.add_argument('--only_compare',action='store_true',dest='only_compare', help='Do not recompute network metrics. Only compare predictions')
    parser.add_argument('--only_graph_measure',action='store_true',dest='only_graph_measure_summary', help='Only compute graph measure summary')
    parser.add_argument('--raw_data',action='store_true',dest='only_raw_data', help='Only compute graph measure summary on observed data')
    parser.add_argument('--statdir',action='store',dest='statdir', help='Directory to save network statistic files')
    parser.add_argument('--allkraken',action='store_true',dest='allkraken', help='Generate all kraken outputs (not just SC->FC)')
    parser.add_argument('--edgecorr',action='store_true',dest='edgecorr', help='Only compute edgecorr information')
    parser.add_argument('--connsim',action='store_true',dest='connsim', help='Only compute connectome similarity information')
    parser.add_argument('--krakenXmeas',action='store_true',dest='krakenXmeas', help='Include kraken for fam vs measured')
    parser.add_argument('--summary_timestamp',action='store_true',dest='add_summary_timestamp', help='Add timestamp to summary file (to avoid overwriting existing file)')
    
    args=parser.parse_args(argv)
    
    #if len(sys.argv) == 1:
    #    parser.print_help(sys.stderr)
    #    sys.exit(1)
    
    return args

if __name__ == "__main__":
    run_prediction_performance(sys.argv[1:])
