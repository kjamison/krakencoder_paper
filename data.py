"""
Functions for loading data for a specific study (including hardcoded paths), and for filtering and combining data for analysis.
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.io import loadmat
import glob
import re

from utils import krakendir, add_krakencoder_package, normfun

add_krakencoder_package() #add krakencoder package to path if not already there
from krakencoder.data import get_hcp_data_flavors, load_hcp_data, canonical_data_flavor, clean_subject_list

def get_study_info(studyname='HCP', studyfolder=None, retest=False, rawdata_sim=None, prediction_sim=None):
    """
    Get hard-coded filenames for a specific study.
    
    Returns a dictionary with the following keys:
    * studyname: str. 'HCP' or 'HCPA' or 'HCPD'
    * studyfolder: str. path to study folder where data can be found
    * Tdemo: pd.DataFrame. Demographic data for the study
    * subjects: list of str. List of subject IDs
    * subject_splits: OPTIONAL dict (Default=None). Contains the following keys:
        * subjidx_train: np.ndarray. Indices of subjects in the training set
        * subjidx_val: np.ndarray. Indices of subjects in the validation set
        * subjidx_test: np.ndarray. Indices of subjects in the test set
        * subjidx_retest: np.ndarray. Indices of subjects in the retest set
        * subjects: list of str. List of subject IDs
    * twininfo_file: str. Path to .mat file containing twin/family information for HCP study (otherwise None)
    * encfile: str. Path to latent data file
    * encfile2: str. Path to second latent data file (if applicable)
        * for instance, we might have separate latent space files for FC and SC
    * encfile_retest: str. Path to retest latent data file
    * rawdata_file_list: list of str. List of paths to raw data files (we search through these later to find one for each conntype)
    * prediction_sim_file: str. Path to prediction similarity file (if applicable)
    
    """
    
    subject_split_file=None
    subject_splits=None
    encfile=None
    encfile2=None
    encfile_retest=None
    twininfo_file=None
    
    if studyname in ['HCP','HCPYA']:
        studyname='HCP'
        #subject_split_file='%s/subject_splits_889subj_635train_72val_182test_33retestInTest.mat' % (studyfolder) #only those with complete dMRI
        #subject_split_file='%s/subject_splits_930subj_660train_77val_193test_retestInTest_noBehavNan.mat' % (studyfolder) #only those with complete behavioral data
        subject_split_file='%s/subject_splits_958subj_683train_79val_196test_retestInTest.mat' % (studyfolder)
        
        #encfile='%s/connae_allflavors_pc256_transformed_993subj.mat' % (studyfolder)
        #encfile_retest='%s/connae_allflavors_pc256_transformed_retest42.mat' % (studyfolder)
        
        #encfile='%s/connae_allflavors_pc128_transformed_993subj.mat' % (studyfolder)
        #encfile_retest='%s/connae_allflavors_pc128_transformed_retest42.mat' % (studyfolder)
        
        encfile='%s/hcp_20230927_012518_ep002000_mse.w1000_encoded.mat' % (studyfolder) #mse.w1000 this one!!!!!
        encfile_retest='%s/hcp_20230927_012518_ep002000_retest_mse.w1000_encoded.mat' % (studyfolder) #mse.w1000 this one!!!!!
        
        #encfile='%s/hcpFC_20230928_042313_ep005000_mse.w1000_encoded.mat' % (studyfolder) #mse.w1000 #single modality
        #encfile2='%s/hcpSC_20230928_000557_ep005000_mse.w1000_encoded.mat' % (studyfolder) #mse.w1000 #single modality
        
        #encfile='%s/hcp_20240320_182306_ep002000_nogsr_mse.w1000_encoded.mat' % (studyfolder)  #balanced with just 12 flavors (no gsr)
        #encfile_retest='%s/hcp_20240320_182306_ep002000_retest_nogsr_mse.w1000_encoded.mat' % (studyfolder) 
        
        #encfile='raw'
        
        twininfo_file=os.path.join(studyfolder,'twininfo_hcp_1206subj.mat')
        
        unrestricted_demo_file='%s/HCP_demographics/unrestricted_kjamison_2_24_2020_9_44_21.csv' % (studyfolder)
        restricted_demo_file='%s/HCP_demographics/RESTRICTED_kjamison_2_24_2020_9_44_6.csv' % (studyfolder)
        cbig_demo_var_file='%s/HCP_demographics/cbig_MMP_HCP_componentscores_1206subj_1022orig.csv' % (studyfolder)
                
        subject_splits=loadmat(subject_split_file,simplify_cells=True)
        try:
            #convert subject ID from float to str
            subject_splits['subjects']=['%d' % (s) for s in subject_splits['subjects']]
        except:
            pass
        #subject_splits['subjects']=['%d' % (s) for s in subject_splits['subjects']]
        subjects=np.array(subject_splits['subjects'])
        
        #subjects=subjects[subject_splits['subjidx_test']]

        #### load demographic data
        Tunres=pd.read_csv(unrestricted_demo_file)
        Tres=pd.read_csv(restricted_demo_file)
        Tcbig=pd.read_csv(cbig_demo_var_file)

        Tdemo=pd.merge(Tunres,Tres).merge(Tcbig)

        Tdemo['CogTotalComp_Unadj_resid']=Tdemo['CogTotalComp_Unadj']-np.mean(Tdemo['CogTotalComp_Unadj'])
        
        demo_newnames={'Age_in_Yrs':'Age'}
        demo_newnames['CogTotalComp_AgeAdj']='nih_totalcogcomp_ageadjusted'
        demo_newnames['CogEarlyComp_AgeAdj']='nih_eccogcomp_ageadjusted'
        demo_newnames['CogCrystalComp_AgeAdj']='nih_crycogcomp_ageadjusted'
        demo_newnames['CogFluidComp_AgeAdj']='nih_fluidcogcomp_ageadjusted'
        
        demo_newnames['Acquisition']='site' #rename Acquisition quarter to 'site' to model like HCPA/D sites
        
        for k,v in demo_newnames.items():
            Tdemo[v]=Tdemo[k]
        Tdemo=Tdemo.drop(columns=list(demo_newnames.keys()))
        
        Tdemo['Subject']=['%d' % (s) for s in Tdemo['Subject']]

    elif studyname in ['HCPA','HCPAging']:
        studyname='HCPAging'
        subjectfile='%s/hcpaging_subjects_rfMRI_dMRI_complete_716subj.txt' % (studyfolder)
        
        #encfile='%s/hcpaFC_20230921_210947_ep002000_meanfitshift_encoded.mat' % (studyfolder)
        #encfile2='%s/hcpaSC_20230922_012808_ep002000_meanfitshift_encoded.mat' % (studyfolder)
        
        encfile='%s/hcpa_20230927_012518_ep002000_meanfitshift_15flav_encoded.mat' % (studyfolder) #15 flavor mse.w1000
        
        #read table with pandas and skip second row (which is a duplicate of the first row):
        demofile='%s/HCPA_demographics/hcpaging_subject_info.csv' % (studyfolder)
        with open(subjectfile,"r") as f:
            subjects=[s.strip() for s in f.readlines()]
        Tdemo=pd.read_csv(demofile)
        Tdemo['Subject']=['%s_V1_MR' % (s) for s in Tdemo['subjectid']]
        
        #read site info
        Tsite=pd.read_table('%s/HCPA_demographics/ndar_subject01.txt' % (studyfolder),skiprows=[1])
        Tsite['subjectid']=Tsite['src_subject_id']
        Tdemo=pd.merge(Tdemo,Tsite[['subjectid','site','interview_date']],on='subjectid')
        
        demo_newnames={'age':'Age','sex':'Gender','family_user_def_id':'Family_ID'}
        
        for k,v in demo_newnames.items():
            Tdemo[v]=Tdemo[k]
        Tdemo=Tdemo.drop(columns=list(demo_newnames.keys()))
        
    elif studyname in ['HCPD','HCPDev','HCPDevelopment']:
        studyname='HCPDev'
        subjectfile='%s/hcpdev_subjects_rfMRI_dMRI_complete_608subj_exclude_outliers.txt' % (studyfolder)
    
        #encfile='%s/hcpdFC_20230925_161829_ep005000_meanfitshift_encoded.mat' % (studyfolder)
        #encfile2='%s/hcpdSC_20230922_122900_ep005000_meanfitshift_encoded.mat' % (studyfolder)
        
        encfile='%s/hcpd_20230927_012518_ep002000_meanfitshift_15flav_encoded.mat' % (studyfolder) #15 flavor mse.w1000
        
        demofile='%s/HCPD_demographics/hcpdev_subject_info.csv' % (studyfolder)
        with open(subjectfile,"r") as f:
            subjects=[s.strip() for s in f.readlines()]
        Tdemo=pd.read_csv(demofile)
        Tdemo['Subject']=['%s_V1_MR' % (s) for s in Tdemo['subjectid']]

        #read site info
        Tsite=pd.read_table('%s/HCPD_demographics/ndar_subject01.txt' % (studyfolder),skiprows=[1])
        Tsite['subjectid']=Tsite['src_subject_id']
        Tdemo=pd.merge(Tdemo,Tsite[['subjectid','site','interview_date']],on='subjectid')
        
        demo_newnames={'age':'Age','sex':'Gender','family_user_def_id':'Family_ID'}
        for k,v in demo_newnames.items():
            Tdemo[v]=Tdemo[k]
        Tdemo=Tdemo.drop(columns=list(demo_newnames.keys()))
    
    rawdata_file_list=None
    retest_str=''
    if retest:
        retest_str='*+retest*'
    if rawdata_sim == 'cosine':
        rawdata_file_list=glob.glob('%s/rawdata_cosinesim/%s_*_%scosinesim.mat' % (studyfolder,studyname.lower(),retest_str))
    elif rawdata_sim == 'corr':
        rawdata_file_list=glob.glob('%s/rawdata_corrsim/%s_*_%scorrsim.mat' % (studyfolder,studyname.lower(),retest_str))
    
    prediction_sim_file=None
    if prediction_sim == 'cosine':
        prediction_sim_file='%s/kraken_cosinesim/hcp_20230927_012518_ep002000_test+retest_alltypes_cosinesim.mat' % (studyfolder)
    elif prediction_sim == 'corr':
        prediction_sim_file='%s/kraken_cosinesim/hcp_20230927_012518_ep002000_test+retest_alltypes_corrsim.mat' % (studyfolder)
    elif prediction_sim == 'corr_resid':
        prediction_sim_file='%s/kraken_cosinesim/hcp_20230927_012518_ep002000_test+retest_alltypes_corrsim_resid.mat' % (studyfolder)
    
    studyinfo=dict(studyname=studyname, studyfolder=studyfolder,Tdemo=Tdemo, twininfo_file=twininfo_file,
                   subject_splits=subject_splits, subjects=subjects,
                   encfile=encfile, encfile2=encfile2, encfile_retest=encfile_retest, 
                   rawdata_file_list=rawdata_file_list, prediction_sim_file=prediction_sim_file)
    
    return studyinfo

def load_study_data(studyname='HCP', studyfolder=None, combine_type='mean', only_include_conntypes=None, exclude_flavor_string=None, 
                    latent_or_raw_spec=None,
                    prediction_sim_input='fusion',
                    retest=False,
                    sex_filter=None, age_range=None,
                    override_study_info=None):
    """
    Load demographics and data for a specific study. Can be either latent data, raw connectome data, or predicted connectomes (when available for a study).
    Will exclude certain flavors if exclude_flavor_string is provided. Will perform "fusion" by averaging across groups of flavors, if necessary.
    Will also filter by demographic factors
    
    Parameters:
    studyname: str. 'HCP' or 'HCPA' or 'HCPD'
    studyfolder: str. path to study folder where data can be found (default: krakendir())
    latent_or_raw_spec: 'latent' or 'raw' or 'latent=corr' or 'raw=cosine' etc (see next 3 parameters for details)
        *  "latent": we use the 128 dimensional dta from the latent space, and average across flavors
        * "latent=corr" (or =cosine, =dist): we average latent vectors across flavors for each subject, THEN compute inter-subject similarity
        * "latent=corr.post" (or =cosine.post, etc), we average across flavors AFTER similarity calculation for each flavor
            * Thus latent=corr.post more closely matches the "raw=corr" setting 
        * For "raw","raw=cosine" (or =corr, =dist), we compute inter-subject similarity for each flavor, THEN average across flavors
        * Same for "pred=cosine" (or =corr, =dist), same as raw
    exclude_flavor_string: exclude flavors with this string in (eg: 'pcorr' or 'gsr')
    prediction_sim_input: str. For loading predicted connectomes, which input type is used? Default='fusion', where all connectomes for a 
        subject are predicted from the averaged latent vector.
    combine_type: str. 'mean' or 'concat'. If 'mean', will average across flavors. If 'concat', will concatenate across flavors. Default='mean'
    retest: bool. If True, include retest subjects in the data. Default=False
    sex_filter: str or list of str. If provided, only include subjects with this sex. (eg: sex_filter='F'). Default=None
    age_range: list of 2 ints [youngest,oldest]. If provided, only include subjects with age within this range, inclusive. 
        (eg: age_range=[20,30] or age_range=[20,np.inf]). Default=None
    
    Returns:
    Data: dict. Contains the following
    * conntypes: list of str. List of connectome types (flavors) included in the data
    * data_alltypes: dict. Contains an [nsubj x nfeat] np.ndarray for each flavor in conntypes
    * conngroups: dict. containing a group for each flavor group (eg: conngroups['FCcorr_fs86_hpf']='FC', conngroups['fusion']='')
    * enc_mean: np.ndarray. [nsubj x nfeat] matrix of *features* averaged across all flavors (except those with exclude_flavor_string)
        * nfeat can be 128 for latent vectors (for latent_or_raw_spec="latent"), 
            or nsubj for inter-subject similarity (eg: latent_or_raw_spec="latent=corr" or "raw=cosine" etc...)
    * enc_FCmean: np.ndarray. [nsubj x nfeat] matrix of averaged features across FC flavors
    * enc_SCmean: np.ndarray. [nsubj x nfeat] matrix of averaged features across SC flavors
    * enc_FCSCmean: np.ndarray. [nsubj x nfeat] (enc_FCmean + enc_SCmean)/2
        * This balances both modalities if different number of each
    * enc_FCSCcat: np.ndarray. [nsubj x nfeat] cat([enc_FCmean,enc_SCmean],axis=1)
    * enc_mean_norm, enc_FCmean_norm, enc_SCmean_norm: np.ndarray. [nsubj x nfeat] L2-normalized versions of the above (re-normalize after averaging). Only relevant for "latent"
    * latent_average_after_sim: bool. True if latent data is averaged after similarity calculation (eg: latent_or_raw_spec="latent=corr.post")
    * enc_is_raw: bool. True if raw data, False if latent data
    * enc_is_pc: bool. True if data is in principal component space
    * enc_is_cosinesim: bool. True if data is cosine similarity, False for "latent"
    * Arguments passed in or returned from get_study_info:
        * studyname, latentdata_sim, rawdata_sim, prediction_sim, encfile, encfile2, twininfo_file, Tdemo, subjects, ...
    """

    latentdata_sim, rawdata_sim, prediction_sim, simpost_info = parse_data_spec(latent_or_raw_spec, normalize=True, return_post_info=True)
    latent_average_after_sim=simpost_info['latent_average_after_similarity']
    latentdata_sim_orig=simpost_info['latentdata_similarity_orig']

    if studyfolder is None:
        studyfolder=krakendir()
    
    studyinfo = get_study_info(studyname=studyname, studyfolder=studyfolder, retest=retest, rawdata_sim=rawdata_sim, prediction_sim=prediction_sim)
    if override_study_info is not None:
        for k,v in override_study_info.items():
            studyinfo[k]=v
    
    Tdemo=studyinfo['Tdemo']
    subjects=studyinfo['subjects']
    subject_splits=studyinfo['subject_splits']
    encfile=studyinfo['encfile']
    encfile2=studyinfo['encfile2']
    encfile_retest=studyinfo['encfile_retest']
    rawdata_file_list=studyinfo['rawdata_file_list']
    prediction_sim_file=studyinfo['prediction_sim_file']
    twininfo_file=studyinfo['twininfo_file']
    
    enc_is_pc=False
    enc_is_raw=False
    enc_is_cosinesim=False
    
    # do demographic filtering
    demo_filter_mask=np.ones(len(Tdemo)).astype(bool)
    if sex_filter is not None:
        if isinstance(sex_filter,str):
            sex_filter=[sex_filter]
        demo_filter_mask=demo_filter_mask & np.array([x in sex_filter for x in Tdemo['Gender']])
    if age_range is not None:
        demo_filter_mask=demo_filter_mask & (Tdemo['Age'] >= age_range[0]) & (Tdemo['Age'] <= age_range[1])
    
    subjects=[s for s in subjects if s in Tdemo['Subject'][demo_filter_mask].to_numpy()]
    
    #match/sort demographics table so rows exactly match 'subjects' variable
    Tdemo=Tdemo.reindex([Tdemo.index[Tdemo['Subject']==s][0] for s in subjects])

    Tdemo=Tdemo.reset_index(drop=True)
    if not np.all(Tdemo['Subject']==subjects):
        raise Exception("Input subjects do not match after reindex!")
    
    if rawdata_sim is not None:
        if rawdata_sim not in ['cosine','corr']:
            raise Exception("Unknown rawdata_sim: %s" % (rawdata_sim))
        
        enc_is_cosinesim=True
        encfile='rawdata_%ssim' % (rawdata_sim)
        encfile2=None
        conntypes=[canonical_data_flavor(c) for c in get_hcp_data_flavors(fc_filter_list=["hpf"])]
        if only_include_conntypes is not None:
            conntypes=[c for c in conntypes if canonical_data_flavor(c,accept_unknowns=True) in only_include_conntypes]
        if exclude_flavor_string:
            conntypes=[c for c in conntypes if exclude_flavor_string not in c]
            
        data_alltypes={}
        for c in conntypes:
            #futz around with the conntype names to match the rawdata files
            c_filestr=c
            c_filestr=c_filestr.replace('coco439','cocommpsuit439')
            c_filestr=c_filestr.replace('hpfgsr','hpf_gsr')
            a=['cocommpsuit439' if 'coco439' in c else 'shen268' if 'shen268' in c else 'fs86'][0]
            c_filestr=c_filestr.replace(a+"_","")
            if c_filestr.endswith("_FC"):
                c_filestr=c_filestr[:-3]
            if c_filestr.startswith("SC"):
                c_filestr=c_filestr[2:]
            if any(['FCcov' in f for f in rawdata_file_list]):
                #if the cosinesim/corrsim files are using the old naming scheme
                c_filestr=c_filestr.replace("FCcorr","FCcov")
            g = ['fc' if 'FC' in c else 'sc'][0]
            
            connfile=[f for f in rawdata_file_list if re.match('.*%s_%s_%s_[0-9]+subj.*' % (g,a,c_filestr),f.split('/')[-1])][0]
            connsim=loadmat(connfile,simplify_cells=True)
            
            if retest and not any([s.endswith("_Retest") for s in subjects]):
                #if retest and we haven't added the _Retest ids to subjects, add them
                subjects_retest=[s for s in connsim['subject'] if s.endswith("_Retest")]
                subjects_trim=[s.replace("_Retest","") for s in subjects_retest]
                sidx_retest=np.array([i for i,s in enumerate(subjects_trim) if s in subjects])
                subjects_retest=np.array(subjects_retest)[sidx_retest]
                subjects=np.concatenate([subjects,subjects_retest])
            
            sidx=[np.where(connsim['subject']==s)[0][0] for s in subjects]
            for connfield in ['FC','SC','C']:
                if connfield in connsim:
                    data_alltypes[c]=connsim[connfield][sidx,:][:,sidx]
                    break
        
        conngroups=get_conngroup_dict(conntypes)
                
        #enc_cat=combine_input_groups(data_alltypes,conngroups, group_to_combine='SCFC', combine_type='mean', normalize=False)
        enc_cat=None
        enc_mean=combine_input_groups(data_alltypes,conngroups, group_to_combine='SCFC', combine_type='mean', normalize=False)
        enc_FCmean=combine_input_groups(data_alltypes,conngroups, group_to_combine='FC', combine_type='mean', normalize=False)
        enc_SCmean=combine_input_groups(data_alltypes,conngroups, group_to_combine='SC', combine_type='mean', normalize=False)
        
    elif prediction_sim is not None:
        if prediction_sim not in ['cosine','corr','corr_resid']:
            raise Exception("Unknown prediction_sim: %s" % (prediction_sim))
        
        enc_is_cosinesim=True
        encfile='kraken_%ssim' % (prediction_sim)
        encfile2=None
        conntypes=[canonical_data_flavor(c) for c in get_hcp_data_flavors(fc_filter_list=["hpf"])]
        if only_include_conntypes is not None:
            conntypes=[c for c in conntypes if canonical_data_flavor(c,accept_unknowns=True) in only_include_conntypes]
        if exclude_flavor_string:
            conntypes=[c for c in conntypes if exclude_flavor_string not in c]
        
        connsim_alltypes=loadmat(prediction_sim_file,simplify_cells=True)
        connsim_alltypes['subject']=connsim_alltypes['subjects'].copy()
        del connsim_alltypes['subjects'] #just to unify the field name
        connsim_alltypes['subject']=np.array([x.strip() for x in connsim_alltypes['subject']])
        
        
        if retest and not any([s.endswith("_Retest") for s in subjects]):
            #if retest and we haven't added the _Retest ids to subjects, add them
            subjects_retest=[s for s in connsim_alltypes['subject'] if s.endswith("_Retest")]
            subjects_trim=[s.replace("_Retest","") for s in subjects_retest]
            sidx_retest=np.array([i for i,s in enumerate(subjects_trim) if s in subjects])
            subjects_retest=np.array(subjects_retest)[sidx_retest]
            subjects=np.concatenate([subjects,subjects_retest])
            
        #for predictions, we don't predict all subjects, so we need to find which subjects are in the prediction file
        #for those, we fill them into the data_alltypes. For the rest, we set them to nan
        sidx=[np.where(connsim_alltypes['subject']==s)[0][0] if s in connsim_alltypes['subject'] else None for s in subjects]
        
        sidx_notfound=np.array([s is None for s in sidx])
        sidx=np.array([s  if s is not None else 0 for s in sidx])
        
        data_alltypes={}
        for c in conntypes:
            data_alltypes[c]=connsim_alltypes['predicted_alltypes'][prediction_sim_input][c][sidx,:][:,sidx]
            data_alltypes[c][sidx_notfound,:]=np.nan
            data_alltypes[c][:,sidx_notfound]=np.nan
        
        conngroups=get_conngroup_dict(conntypes)
                
        enc_cat=None
        enc_mean=combine_input_groups(data_alltypes,conngroups, group_to_combine='SCFC', combine_type='mean', normalize=False)
        enc_FCmean=combine_input_groups(data_alltypes,conngroups, group_to_combine='FC', combine_type='mean', normalize=False)
        enc_SCmean=combine_input_groups(data_alltypes,conngroups, group_to_combine='SC', combine_type='mean', normalize=False)
        
        
    elif encfile=='raw':
        enc_is_raw=True
        conntypes=get_hcp_data_flavors(fc_filter_list=["hpf"])
        _, conndata_alltypes = load_hcp_data(subjects=subjects, conn_name_list=conntypes, load_retest=False, quiet=False, keep_diagonal=False)
        
        conntypes=list(conndata_alltypes.keys())
        if only_include_conntypes is not None:
            conntypes=[c for c in conntypes if canonical_data_flavor(c,accept_unknowns=True) in only_include_conntypes]
        if exclude_flavor_string:
            conntypes=[c for c in conntypes if exclude_flavor_string not in c]
        
        conngroups=get_conngroup_dict(conntypes)
                
        data_alltypes={k:conndata_alltypes[k]['data'].copy() for k in conntypes}
        
        conndata_alltypes=None
        
        sidx=[i for i,s in enumerate(subjects)]
        
        enc_cat=combine_input_groups(data_alltypes,conngroups, group_to_combine='SCFC', combine_type='concat', normalize=False)
        #enc_mean=combine_input_groups(data_alltypes,conngroups, group_to_combine='SCFC', combine_type=combine_type, normalize=False)
        enc_mean=None
        enc_FCmean=combine_input_groups(data_alltypes,conngroups, group_to_combine='FC', combine_type='concat', normalize=False)
        enc_SCmean=combine_input_groups(data_alltypes,conngroups, group_to_combine='SC', combine_type='concat', normalize=False)
    else:
        Menc=loadmat(encfile,simplify_cells=True)
        
        try:
            #if they are numeric, convert to string
            Menc['subjects']=np.array(['%d' % (s) for s in Menc['subjects']])
        except:
            pass
        Menc['subjects']=np.array([s.strip() for s in Menc['subjects']])
        #match/sort latent space matrices so rows exactly match 'subjects' variable
        sidx=[np.where(Menc['subjects']==s)[0][0] for s in subjects]
        
        ######
        if encfile2 is not None:
            Menc2=loadmat(encfile2,simplify_cells=True)
            try:
                #if they are numeric, convert to string
                Menc2['subjects']=np.array(['%d' % (s) for s in Menc2['subjects']])
            except:
                pass
            Menc2['subjects']=np.array([s.strip() for s in Menc2['subjects']])
            
            if not all([Menc2['subjects'][i]==Menc['subjects'][i] for i in range(len(Menc['subjects']))]):
                raise Exception("Subjects from all encoding files must match")
            
            for k in Menc2['predicted_alltypes']:
                #Menc['predicted_alltypes'][k]=Menc2['predicted_alltypes'][k].copy()
                if k in Menc['predicted_alltypes']:
                    del Menc['predicted_alltypes'][k]
                Menc['predicted_alltypes'][k]=Menc2['predicted_alltypes'][k].copy()
                #Menc['predicted_alltypes'][k]={'encoded':Menc2['predicted_alltypes'][k]['encoded'].copy()}
            for k in Menc2['inputtypes']:
                if not k in Menc['inputtypes']:
                    Menc['inputtypes']=np.append(Menc['inputtypes'],[k])
        
        if retest and encfile_retest is not None:
            print(encfile_retest)
            print(encfile)
            Menc2=loadmat(encfile_retest,simplify_cells=True)
            try:
                #if they are numeric, convert to string
                Menc2['subjects']=np.array(['%d' % (s) for s in Menc2['subjects']])
            except:
                pass
            Menc2['subjects']=np.array([s.strip() for s in Menc2['subjects']])
            
            subjects_trim=[s.replace("_Retest","") for s in Menc2['subjects']]
            #sidx_retest=[np.where(Menc['subjects']==s)[0][0] for s in subjects_trim]
            sidx_retest=np.array([i for i,s in enumerate(subjects_trim) if s in subjects])
            
            if len(Menc['inputtypes']) != len(Menc2['inputtypes']) or not all([Menc2['inputtypes'][i]==Menc['inputtypes'][i] for i in range(len(Menc['inputtypes']))]):
                raise Exception("inputtypes from all encoding files must match")
            
            for k in Menc2['predicted_alltypes']:
                if not k in Menc['predicted_alltypes']:
                    continue
                for kk in Menc2['predicted_alltypes'][k]:
                    Menc['predicted_alltypes'][k][kk]=np.concatenate([Menc['predicted_alltypes'][k][kk],Menc2['predicted_alltypes'][k][kk][sidx_retest,:]],axis=0)
                #Menc['predicted_alltypes'][k]={'encoded':Menc2['predicted_alltypes'][k]['encoded'].copy()}
            
            #Menc['predicted_alltypes']={k:Menc['predicted_alltypes'][k] for k in Menc['predicted_alltypes'] if k in Menc2['predicted_alltypes']}
           
                
            Menc['subjects']=np.concatenate([Menc['subjects'],Menc2['subjects'][sidx_retest]])
            subjects=np.concatenate([subjects,Menc2['subjects'][sidx_retest]])
            sidx=[np.where(Menc['subjects']==s)[0][0] for s in subjects]
        
        ######
        conntypes=[c.strip() for c in Menc['inputtypes']]
        if only_include_conntypes is not None:
            conntypes=[c for c in conntypes if canonical_data_flavor(c,accept_unknowns=True) in only_include_conntypes]
        if exclude_flavor_string:
            conntypes=[c for c in conntypes if exclude_flavor_string not in c]
        
        conngroups=get_conngroup_dict(conntypes)
        
        if 'encoded_alltypes' in Menc:
            data_alltypes={k:Menc['encoded_alltypes'][k][sidx,:] for k in conntypes}
        elif 'encoded' in Menc['predicted_alltypes'][conntypes[0]]:
            data_alltypes={k:Menc['predicted_alltypes'][k]['encoded'][sidx,:] for k in conntypes}
        elif 'transformed' in Menc['predicted_alltypes'][conntypes[0]]:
            enc_is_pc=True
            data_alltypes={k:Menc['predicted_alltypes'][k]['transformed'][sidx,:] for k in conntypes}
        
        if latentdata_sim:
            if latentdata_sim == 'cosine':
                simfun=lambda x: cosine_similarity(x)
            elif latentdata_sim == 'corr':
                simfun=lambda x: np.corrcoef(x)
            elif latentdata_sim == 'dist':
                simfun=lambda x: -euclidean_distances(x)
            else:
                raise Exception("Unknown latentdata_sim: %s" % (latentdata_sim))
            
            if latent_average_after_sim:
                data_alltypes={k:simfun(v) for k,v in data_alltypes.items()}
            
            enc_is_cosinesim=True
        
        if enc_is_cosinesim:
            enc_cat = None
        else:
            enc_cat=combine_input_groups(data_alltypes,conngroups, group_to_combine='SCFC', combine_type='concat', normalize=False)
        enc_mean=combine_input_groups(data_alltypes,conngroups, group_to_combine='SCFC', combine_type=combine_type, normalize=False)
        enc_FCmean=combine_input_groups(data_alltypes,conngroups, group_to_combine='FC', combine_type=combine_type, normalize=False)
        enc_SCmean=combine_input_groups(data_alltypes,conngroups, group_to_combine='SC', combine_type=combine_type, normalize=False)
    
    enc_FCSCmean=(enc_FCmean+enc_SCmean)/2
    enc_FCSCcat=np.hstack((enc_FCmean,enc_SCmean))
    
    enc_mean_norm=normfun(enc_mean)
    enc_SCmean_norm=normfun(enc_SCmean)
    enc_FCmean_norm=normfun(enc_FCmean)
    
    if latentdata_sim and not latent_average_after_sim:
        #latent average BEFORE sim: we have averaged flavors in latent space, THEN do similarity
        #this is what the code does by default
        print("Computing latent subject %s similarity AFTER averaging flavors" % (latentdata_sim_orig))
        print("enc_mean=%dx%d -> Xsim=%dx%d" % (enc_mean.shape[0], enc_mean.shape[1], enc_mean.shape[0], enc_mean.shape[0]))
        
        enc_mean=simfun(enc_mean)
        enc_FCmean=simfun(enc_FCmean)
        enc_SCmean=simfun(enc_SCmean)
        enc_FCSCmean=simfun(enc_FCSCmean)
        enc_FCSCcat=np.hstack((enc_FCmean,enc_SCmean))
        
        enc_mean_norm=simfun(enc_mean_norm)
        enc_FCmean_norm=simfun(enc_FCmean_norm)
        enc_SCmean_norm=simfun(enc_SCmean_norm)
    elif latentdata_sim and latent_average_after_sim:
        #average AFTER sim: should do latentdata_sim, where data_alltypes is similarity, then average
        print("Computed latent subject %s similarity PER flavor, then averaged" % (latentdata_sim_orig))
        print("Xsim=average([%dx%d]*Nflav)" % (enc_mean.shape[0], enc_mean.shape[1]))
    #normfun=lambda x: x/np.sqrt(np.sum(x**2,axis=1,keepdims=True))
    
    Tdemo['studyname']=studyname
    return {'studyname':studyname,'Tdemo':Tdemo,'subject_splits':subject_splits,'sidx':sidx,'subjects':subjects, 'twininfo_file':twininfo_file,
            'data_alltypes':data_alltypes, 'conntypes':conntypes,'conngroups':conngroups,
            'enc_mean':enc_mean, 'enc_FCmean':enc_FCmean, 'enc_SCmean':enc_SCmean, 'enc_mean_norm':enc_mean_norm, 'enc_FCmean_norm':enc_FCmean_norm, 'enc_SCmean_norm':enc_SCmean_norm, 'enc_cat':enc_cat,
            'enc_FCSCmean':enc_FCSCmean, 'enc_FCSCcat':enc_FCSCcat,
            'enc_is_raw':enc_is_raw,'enc_is_pc':enc_is_pc,'enc_is_cosinesim':enc_is_cosinesim,'encfile':encfile, 'encfile2':encfile2,'combine_type':combine_type,
            'latentdata_sim':latentdata_sim,'rawdata_sim':rawdata_sim, 'prediction_sim':prediction_sim, 'latent_average_after_sim':latent_average_after_sim}


def combine_input_groups(data_dict, group_dict, group_to_combine='all', combine_type='mean', normalize=False):
    """
    Combine data from different connectivity types (flavors) into a single matrix, based on their group membership (eg: SC, FC)
    
    Parameters:
    data_dict: dict. Contains [nsubj x nfeat] np.ndarray for each connectivity type (flavor)
    group_dict: dict. Contains a group for each connectivity type (flavor)
    group_to_combine: str. Group to combine (eg: 'SC', 'FC', 'SCFC'). Default='all'
    combine_type: str. 'mean' or 'concat'. If 'mean', will average across flavors. If 'concat', will concatenate across flavors. Default='mean'
    normalize: bool. If True, L2-normalize the combined data after averaging. Default=False
    
    Returns:
    data_combined: np.ndarray. [nsubj x nfeat] matrix of combined data
    """
    if group_to_combine=='SCFC':
        grouplist=['SC','FC']
    else:
        grouplist=[group_to_combine]
    
    if combine_type=='mean':
        data_combined=np.mean(np.stack([data_dict[c] for c in data_dict if group_dict[c] in grouplist],axis=-1),axis=-1)
    elif combine_type in ['concat','concatenate']:
        data_combined=np.concatenate([data_dict[c] for c in data_dict if group_dict[c] in grouplist],axis=1)
    elif combine_type.startswith('concat+pc'):
        concat_pc_num=int(combine_type.replace("concat+pc",""))
        data_temp=np.concatenate([data_dict[c] for c in data_dict if group_dict[c] in grouplist],axis=1)
        data_combined=PCA(n_components=concat_pc_num,random_state=0).fit_transform(data_temp)
    else:
        raise Exception("Unknown combine_type: %s" % (combine_type))
    
    if normalize:
        normfun=lambda x: x/np.sqrt(np.sum(x**2,axis=1,keepdims=True))
        data_combined=normfun(data_combined)
        
    return data_combined


def normalize_sim_fun_name(sim=None):
    if sim in ['cosine','cos']:
        return 'cosine'
    elif sim in ['corr','correlation']:
        return 'corr'
    elif sim in ['dist','distance']:
        return 'dist'
    elif sim in ['cos.post','cosine.post']:
        return 'cosine.post'
    elif sim in ['corr.post','correlation.post']:
        return 'corr.post'
    elif sim in ['corr_resid','correlation_resid']:
        return 'corr_resid'
    else:
        return sim
    
def parse_data_spec(latent_or_raw_spec, normalize=True, return_post_info=False):
    raw_conndata_similarity=None
    predict_conndata_similarity=None
    latent_similarity=None
    
    if latent_or_raw_spec.startswith("raw"):
        if latent_or_raw_spec.startswith("raw="):
            raw_conndata_similarity=latent_or_raw_spec.split("=")[1]
        else:
            raw_conndata_similarity='cosine'
    elif latent_or_raw_spec.startswith("pred"):
        if latent_or_raw_spec.startswith("pred="):
            predict_conndata_similarity=latent_or_raw_spec.split("=")[1]
        else:
            predict_conndata_similarity='cosine'
    elif latent_or_raw_spec.startswith("latent"):
        if latent_or_raw_spec.startswith("latent="):
            latent_similarity=latent_or_raw_spec.split("=")[1]
    
    if normalize:
        latent_similarity=normalize_sim_fun_name(latent_similarity)
        raw_conndata_similarity=normalize_sim_fun_name(raw_conndata_similarity)
        predict_conndata_similarity=normalize_sim_fun_name(predict_conndata_similarity)
    
    if return_post_info:
        latent_average_after_similarity=False
        latentdata_similarity_orig=latent_similarity
        if latent_similarity is not None and latent_similarity.endswith('.post'):
            latent_similarity=latent_similarity.replace('.post','')
            latent_average_after_similarity=True
        post_info=dict(latent_average_after_similarity=latent_average_after_similarity, latentdata_similarity_orig=latentdata_similarity_orig)
        return latent_similarity, raw_conndata_similarity, predict_conndata_similarity, post_info
    else:
        return latent_similarity, raw_conndata_similarity, predict_conndata_similarity

def get_conngroup_dict(conntypes):
    conngroups={}
    for c in conntypes:
        if 'fusion' in c or 'burst' in c:
            conngroups[c]=''
        elif 'FC' in c:
            conngroups[c]='FC'
        else:
            conngroups[c]='SC'
    return conngroups

def get_family_level_dict(return_inverse=False, shortnames=False):
    famlevel_dict={'unrelated': 0, 'sibling': 1, 'DZ': 2, 'MZ': 3, 'self': 4, 'retest': 5, 'bad': 10}
    famlevel_short={'unrelated':'unrel', 'sibling':'sib'}
    if return_inverse:
        famlevel_dict_inv={}
        for k,v in famlevel_dict.items():
            if shortnames and k in famlevel_short:
                k=famlevel_short[k]
            famlevel_dict_inv[v]=k
        famlevel_dict=famlevel_dict_inv
    return famlevel_dict

def load_family_info(studyname, Tdemo, twininfo_file, Data_subjects, subjidx):
    """
    Parse family information for a study, based on the demographics table and twininfo.mat file
    
    Parameters:
    studyname: str. 'HCP' or 'HCPA' or 'HCPD'
    Tdemo: pd.DataFrame. Demographics table for the study (See load_study_data)
    twininfo_file: str. path to twininfo.mat file (for 'HCP' only)
    Data_subjects: list(str). List of subjects in the data (from Data['subjects'])
    subjidx: list(int). List of indices of subjects in Data_subjects that we're using here
    
    Returns:
    dict. Contains the following
    * Tdemo: pd.DataFrame. Demographics table for the study (filtered to only include subjects in Data_subjects)
    * subjidx: list(int). List of indices of subjects in Data_subjects that we're using here
    * Xfammatch: np.ndarray. [nsubj x nsubj] matrix of whether subjects are in the same family
    * Xsexmatch: np.ndarray. [nsubj x nsubj] matrix of whether subjects have the same
    * Xagediff: np.ndarray. [nsubj x nsubj] matrix of age differences between subjects
    * Xfamlevel: np.ndarray. [nsubj x nsubj] matrix of family relationships between subjects
        * Xfamlevel[i,j]=0 if subjects i and j are unrelated
        * Xfamlevel[i,j]=3 if subjects i and j are MZ twins (see famlevel_dict)
        * MZ, DZ, retest only apply to 'HCP'. 'HCPA' and 'HCPD' have some related non-twins
    * famlevel_dict: dict with names for looking up values in Xfamlevel:
    *   unrelated=0, sibling=1, DZ=2, MZ=3, self=4, retest=5, bad=10
    """
    famlevel_dict=get_family_level_dict()
    Tdemo=Tdemo.iloc[subjidx,:] #only keep the subjects we're using (eg: subjidx_test)

    Xfammatch=(Tdemo['Family_ID'].to_numpy()[:,np.newaxis]==Tdemo['Family_ID'].to_numpy()[:,np.newaxis].T)
    Xsexmatch=(Tdemo['Gender'].to_numpy()[:,np.newaxis]==Tdemo['Gender'].to_numpy()[:,np.newaxis].T)
    Xagediff=np.abs((Tdemo['Age'].to_numpy()[:,np.newaxis]-Tdemo['Age'].to_numpy()[:,np.newaxis].T))
    
    if studyname in ['HCPDev','HCPAging']:
        Xfamlevel=famlevel_dict['unrelated']+np.zeros(Xfammatch.shape)
        Xfamlevel[Xfammatch]=famlevel_dict['sibling']

    elif studyname=='HCP':
        twininfo=loadmat(twininfo_file,simplify_cells=True)
        try:
            #convert subject ID from float to str
            twininfo['subject']=['%d' % (s) for s in twininfo['subject']]
        except:
            pass
        #subject_splits['subjects']=['%d' % (s) for s in subject_splits['subjects']]
        twininfo['subject']=np.array(twininfo['subject'])

        twininfo_sidx=np.array([np.where(twininfo['subject']==s)[0][0] for s in Tdemo['Subject']])
        Xfamlevel=twininfo['famlevel'][twininfo_sidx,:][:,twininfo_sidx]
        
        #subjidx_retest=[i for i,s in enumerate(Data['subjects'][subjidx]) if s.endswith("_Retest")]
        if any([s.endswith("_Retest") for s in Data_subjects]):
            #find the non-retest for each retest subject in Data['subjects'], and add a row to Tdemo, Xsexmatch, Xagediff, Xfamlevel
            subj_retest=[s for s in Data_subjects if s.endswith("_Retest")]
            subj_retest_trim=[s.replace("_Retest","") for s in subj_retest]
            sidx_retest=[np.where(Data_subjects[subjidx]==s)[0][0] for s in subj_retest_trim if s in Data_subjects[subjidx]]
            Tdemo_retest=pd.DataFrame()
            for i in sidx_retest:
                Ttmp = Tdemo.iloc[[i]].copy()
                Ttmp['Subject'] = Ttmp['Subject'] + "_Retest"
                Tdemo_retest = pd.concat([Tdemo_retest, Ttmp], ignore_index=True)
            Tdemo=pd.concat((Tdemo,Tdemo_retest),ignore_index=True)
            
            Xsexmatch=(Tdemo['Gender'].to_numpy()[:,np.newaxis]==Tdemo['Gender'].to_numpy()[:,np.newaxis].T)
            Xagediff=np.abs((Tdemo['Age'].to_numpy()[:,np.newaxis]-Tdemo['Age'].to_numpy()[:,np.newaxis].T))
    
            Xfamlevel_retest=famlevel_dict['bad']*np.ones((Xfamlevel.shape[0]+len(sidx_retest),len(sidx_retest)))
            for i,iretest in enumerate(sidx_retest):
                Xfamlevel_retest[iretest,i]=famlevel_dict['retest']
            Xfamlevel=np.hstack((Xfamlevel,Xfamlevel_retest[:Xfamlevel.shape[0],:]))
            Xfamlevel=np.vstack((Xfamlevel,Xfamlevel_retest.T))
            subjidx=np.concatenate([subjidx,[np.where(Data_subjects==s)[0][0] for s in subj_retest if s in Data_subjects]])
    
    return {'Tdemo':Tdemo, 'Xfammatch':Xfammatch, 'Xsexmatch':Xsexmatch, 'Xagediff':Xagediff, 'Xfamlevel':Xfamlevel, 'subjidx':subjidx, 'famlevel_dict': famlevel_dict}


def get_matching_subject(input_subjects, match_type='self', twininfo=None, match_sex=True, maximum_age_difference=None, randseed=None):
    '''
    Find matching subjects for a list of input subjects, based on family relationships
    
    Parameters:
    input_subjects: list(str) of subject IDs for which to find matches
    match_type: {'self', 'retest', 'MZ', 'DZ', 'sib', 'unrelated'} what type of subject to find for each input
    twininfo: dict() from twininfo.mat containing 'subject','age','ismale','famlevel','famlevel_dict'
        (not needed for match_type = 'self' or 'retest')
    match_sex: Should match have same sex as input? default=True
    maximum_age_difference: How close should age be to input subject? (default: None = ignore age)
    randseed: (default=None, no reseeding) seed for random selection when multiple matches are found

    Returns:
    new_subjects: list(str) of new subject IDs, one per input subject (or '' if no match found)
    input_subjects_mask: array(bool) with True/False for each input_subjects/new_subjects entry. False=no match found for that subj
    '''
    match_type=match_type.lower()

    if match_type == 'self':
        new_subjects=input_subjects.copy()
        input_subjects_mask=np.ones(len(input_subjects))==1
    
    elif match_type == 'retest':
        new_subjects=np.array(["%s_Retest" % (s) for s in input_subjects])
        input_subjects_mask=np.ones(len(input_subjects))==1
    
    else:
        if twininfo is None:
            raise Exception("Must provide twininfo for match_type=%s" % (match_type))
    
        if match_type == 'mz':
            famlevel_key='MZ'
        elif match_type == 'dz':
            famlevel_key='DZ'
        elif match_type in ['sib','sibling']:
            famlevel_key='sibling'
        else:
            famlevel_key='unrelated'
        

        twininfo_subj=clean_subject_list(twininfo['subject'])
        twininfo_age=twininfo['age'].astype(float)

        fammat=twininfo['famlevel']==twininfo['famlevel_dict'][famlevel_key]

        if maximum_age_difference is not None:
            agemat=np.abs(np.subtract.outer(twininfo_age,twininfo_age))
            fammat[agemat>maximum_age_difference]=False

        if match_sex:
            sexmat=np.equal.outer(twininfo['ismale'],twininfo['ismale'])
            fammat[sexmat==False]=False

        #set fammat to false for any subjects NOT in input_subjects 
        twininfo_not_in_inputs=[s not in input_subjects for s in twininfo_subj]
        fammat[twininfo_not_in_inputs,:]=False
        fammat[:,twininfo_not_in_inputs]=False

        #find our input_subjects in twininfo
        subjidx=[np.where(twininfo_subj==s)[0][0] for s in input_subjects]

        #find all matching pairs in fammat
        ifam,jfam=np.where(fammat)

        input_subjects_mask=np.zeros(len(subjidx))==1
        new_subjects=['']*len(subjidx)

        if randseed is not None:
            np.random.seed(seed=randseed)

        for i,s in enumerate(subjidx):
            m=ifam==s
            if not np.any(m):
                continue

            newsubjidx=jfam[m]

            input_subjects_mask[i]=True
            
            if(len(newsubjidx)>0):
                ridx=np.random.randint(0,len(newsubjidx))
                #choose randomly if more than one match
                newsubjidx=newsubjidx[ridx]

            new_subjects[i]=twininfo_subj[newsubjidx]
        new_subjects=np.array(new_subjects)

    return new_subjects, input_subjects_mask
