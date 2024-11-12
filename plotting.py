"""
Functions for plotting results of the demographic prediction analysis and family similarity
"""

import matplotlib
#need these to make sure fonts are embedded in svg/pdf
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42

from matplotlib import pyplot as plt

import numpy as np
from PIL import Image
import colorsys
import ipywidgets as widgets
from IPython.display import display


from utils import *
from predict import *
from data import get_family_level_dict

add_krakencoder_package() #add krakencoder package to path if not already there
from krakencoder.plotfigures import shorten_names, flavor2color

def pretty_demo_name(d):
    if d=='varimax_cog':
        d='Cognition(raw)'
    elif d=='varimax_satisf':
        d='Dissatisfaction(raw)'
    elif d=='varimax_er':
        d='Emotion(raw)'
    elif d=='varimax_cog_resid':
        d='Cognition'
    elif d=='varimax_satisf_resid':
        d='Dissatisfaction'
    elif d=='varimax_er_resid':
        d='Emotion'
    elif d=='nih_totalcogcomp_unadjusted_resid':
        d='Cognition.unadj.resid'
    elif d=='nih_totalcogcomp_unadjusted':
        d='Cognition.unadj(raw)'
    elif d=='nih_totalcogcomp_ageadjusted':
        d='TotalCog'
    elif d=='nih_eccogcomp_ageadjusted':
        d='EarlyChildCog'
    elif d=='nih_crycogcomp_ageadjusted':
        d='CrystalCog'
    elif d=='nih_fluidcogcomp_ageadjusted':
        d='FluidCog'
    return d

def pretty_input_name(x, enc_is_cosinesim=False, enc_is_pc=False, rawdata=False):
    #enc_is_cosinesim=Data['enc_is_cosinesim']
    #enc_is_pc=Data['enc_is_pc']
    #rawdata='dataname' in df_performance and len(np.unique(df_performance['dataname']))>1
    if enc_is_cosinesim:
        if x.startswith("enc_"):
            x=x.replace("enc_","raw")
            #x=x.replace("enc_","pcamean")
        x=x.replace("_norm",".norm")
        #x=x.replace("mean",".mean")
        x=x.replace("mean","")    
    elif enc_is_pc:
        if x.startswith("enc_"):
            x=x.replace("enc_","pcacat")
            #x=x.replace("enc_","pcamean")
        x=x.replace("_norm",".norm")
        #x=x.replace("mean",".mean")
        x=x.replace("mean","")
    else:
        if x.startswith("enc_"):
            x=x.replace("enc_","enc")
        x=x.replace("_norm",".norm")
        #x=x.replace("mean",".mean")
        x=x.replace("mean","")
    if rawdata:
        x=x.replace("enc","").replace("raw","")
    return x


def adjust_saturation(color, amount=0.5):
    """
    Adjust the saturation of a color by a certain amount (Mute colors a bit)
    """
    h, l, s = colorsys.rgb_to_hls(*color[:3])
    s *= amount
    newcolor=colorsys.hls_to_rgb(h, l, s)
    if len(color)>3:
        newcolor+=(color[3],)
    return newcolor

def crop_background(img, color=None, outfile=None, dpi=None):
    """
    Crop the background of an image to the smallest rectangle that contains all non-background pixels
    
    img: np.array [x y 3] or str (filename)
    color: color of the background (default is the top-left pixel)
    outfile: (optional) filename to save the cropped image. If 'same' and img was a filename, overwrite the input file
    dpi: (optional) dpi to save the image at (if outfile is not None)
    
    Returns: np.array [x y 3] of the cropped image
    """
    if isinstance(img,str):
        if outfile == 'same':
            outfile=img
        img=Image.open(img)
        inputdpi=img.info['dpi']
        img=np.asarray(img)
        if dpi is None:
            dpi=inputdpi
    else:
        if outfile == 'same':
            outfile=None
    
    if color is None:
        color=img[0,0,:]
            
    mask=np.all(img==color,axis=2)
    mask=np.logical_not(mask)
    if np.sum(mask)==0:
        return img
    ymin,ymax=np.where(np.any(mask,axis=1))[0][[0,-1]]
    xmin,xmax=np.where(np.any(mask,axis=0))[0][[0,-1]]
    
    croppedimg=img[ymin:ymax+1,xmin:xmax+1,:]
    
    if outfile is not None:
        img2 = Image.fromarray(croppedimg)
        img2.save(outfile,dpi=dpi)
        
    return croppedimg

def add_save_buttons(output_dir,output_filename,fig,df,dpi=300):
    """
    Add save buttons to the output of a plot in a jupyter notebook
    """
    os.makedirs(output_dir,exist_ok=True)
    
    savebutton_csv = widgets.Button(description="Save csv")
    savebutton_png = widgets.Button(description="Save png")
    savebutton_dirtxt=widgets.Text(value=output_dir,placeholder='output folder',description='Folder:',disabled=False,visible=False,layout=widgets.Layout(width='90%'))
    savebutton_filetxt=widgets.Text(value=output_filename,
                                    placeholder='output file',description='Filename:',disabled=False,visible=False,layout=widgets.Layout(width='90%'))
    
    
    def save_button_click(b):
        button_outfile=None
        
        if savebutton_dirtxt.value:
            button_outdir=savebutton_dirtxt.value
        else:
            button_outdir=output_dir
        
        if not os.path.isdir(button_outdir):    
            os.makedirs(button_outdir,exist_ok=True)
        
        if savebutton_filetxt.value:
            button_outfile_base=savebutton_filetxt.value.replace("%T",timestamp())
        else:
            button_outfile_base=output_filename.replace("%T",timestamp())
        
        if b == 'csv':
            button_outfile=os.path.join(output_dir,button_outfile_base+".csv")
            df.to_csv(button_outfile,index=False)
        elif b == 'png':
            button_outfile=os.path.join(output_dir,button_outfile_base+".png")
            fig.savefig(button_outfile,dpi=dpi,bbox_inches='tight')
            
        if button_outfile:
            print('Saved %s' % (button_outfile))
            
    savebutton_png.on_click(lambda b: save_button_click("png"))
    savebutton_csv.on_click(lambda b: save_button_click("csv"))
    display(widgets.HBox((savebutton_csv,savebutton_png,widgets.VBox((savebutton_dirtxt,savebutton_filetxt),layout=widgets.Layout(width='50%')))))

def drawbracket(x1,x2,y,dy=0.05,linewidth=1,color='k',text=None,fontsize=12, below=True, label=None):
    """
    Draw a bracket between x1 and x2 at y with an optional text label
    If x1 and x2 are arrays of length 2, draw a "nested" bracket between the two pairs of x values (like with HCP family groups)
    Othewrise draw a single bracket between x1 and x2
    
    x1: float or array-like [2]
    x2: float or array-like [2]
    y: float. y position of the bracket
    dy: float. height of the bracket
    linewidth: float. width of the bracket
    color: str. color of the bracket
    text: str. text to display in the middle of the bracket (optional)
    fontsize: float. font size of the text
    below: bool. If True, draw the bracket below the data, otherwise draw it above (just affects orientation)
    label: str. label for the bracket (optional, used for modifying figure elements later)
    """
    va='top'
    ytxt=y-dy/4
    if not below:
        dy=-dy
        ytxt=y
        va='bottom'
    
    if iterable(x1) and len(x1)==2:
        plt.plot([x1[0],x1[0],x1[1],x1[1]],[y+dy,y+dy/2,y+dy/2,y+dy],'-',color=color,linewidth=linewidth,label=label)
        plt.plot([x2[0],x2[0],x2[1],x2[1]],[y+dy,y+dy/2,y+dy/2,y+dy],'-',color=color,linewidth=linewidth,label=label)
        x12=[np.mean(x1),np.mean(x2)]
        plt.plot([x12[0],x12[0],x12[1],x12[1]],[y+dy/2,y,y,y+dy/2],'-',color=color,linewidth=linewidth,label=label)
        
        xtxt=np.mean([x1[0],x1[1],x2[0],x2[1]])
    else:
        plt.plot([x1,x1,x2,x2],[y+dy,y,y,y+dy],'-',color=color,linewidth=linewidth,label=label)
        xtxt=np.mean([x1,x2])
    
    if text is not None:
        plt.text(xtxt,ytxt,text,fontsize=fontsize,color=color,horizontalalignment='center',verticalalignment=va,label='txt'+label)
        
#################################################################################################
#################################################################################################

def plot_family_similarity_violins(df_fam_performance, conntypes, stat_list, famlevels,
                            pval_test_info, normalize_similarity, unrelated_nonsibling_match=False, 
                            violin_max_agediff=3, violin_samesex=True, output_file=None, figdpi=100,
                            closeall=True):
    """
    Plot the family similarity results as violins for each family level and connection type
    
    Parameters:
    df_fam_performance: pd.DataFrame. Dataframe with family similarity results (output of get_family_data_similarity)
    conntypes: list of str. Connection types to plot
    stat_list: list of dicts. List of statistics to plot (output of compare_family_data_similarity)
    famlevels: list of int. Family levels to plot
    pval_test_info: dict. Information about the p-value test (see input of correct_family_similarity_pvals)
    normalize_similarity: bool. If True, y-axis will be "Normalized similarity", otherwise it will be "Pearson r"
    unrelated_nonsibling_match: bool. If True, "unrelated" violin will be pulled from the sibling pair matching (for HCPDev)
    violin_max_agediff: int. Maximum age difference for pairs to be included in the violin plot
    violin_samesex: bool. If True, only pairs with the same sex will be included in the violin plot
    figdpi: int. DPI of the output figure (default=100)
    closeall: bool. If True (default), close all previous figures before plotting the new one (useful in jupyter notebooks)
    output_file: str. Filename to save the output figure (optional)
    
    Returns: 
    fig. The output figure
    """
    famlevel_dict=get_family_level_dict()
    famlevel_str = get_family_level_dict(return_inverse=True, shortnames=True)
    conntypes_short={k:v for k,v in zip(conntypes,shorten_names(conntypes,remove_common=False))}

    
    ax_fontsize=12
    ax_fontsize_group=14
    title_fontsize=16

    #ax_fontsize=8
    #ax_fontsize_group=10

    violwidth=1/(len(famlevels)+2)
    violgap=1/(len(famlevels)+1)

    colors=['r','lightgray','w']

    if closeall:
        plt.close('all')
        
    #now plot the results
    if len(conntypes)<=3:
        fig=plt.figure(figsize=(len(conntypes)*5,4),facecolor='w',dpi=figdpi)
    else:
        #fig=plt.figure(figsize=(15,15),facecolor='w',dpi=figdpi)
        fig=plt.figure(figsize=(15,10),facecolor='w',dpi=figdpi)

    do_plot_samesex_lines=False

    #datatype_list=['latent','raw']
    datatype_list=df_fam_performance['datatype_full'].unique()

    #datatype_list=['latent','raw','pred']
    
    axlist=[]
    for iconn, conntype in enumerate(conntypes):
        #plt.figure()
        if len(conntypes)<=3:
            ax=plt.subplot(1,len(conntypes),iconn+1)
        else:
            #ax=plt.subplot(int(np.ceil(np.sqrt(len(conntypes)))),int(np.ceil(len(conntypes)/np.ceil(np.sqrt(len(conntypes))))),iconn+1)
            ax=plt.subplot(4,5,iconn+1)
        
        xticks=[]
        xticklabels=[]
        xticks_group=[]
        xticklabels_group=[]

        #for isim,cosinesim in enumerate([False,True]):
        for idatatype,datatype in enumerate(datatype_list):

            conntype_str=conntype
            if conntype in conntypes_short:
                conntype_str=conntypes_short[conntype]
            
            datacolor=colors[idatatype]

            if datatype.startswith('raw'):
                sibsim_titlestr='observed data'
                ylabel='Similarity (cosine)'
            else:
                sibsim_titlestr='latent'
                ylabel='Similarity (corrcoef)'
                datacolor=flavor2color(conntype)
            

            xticks_group.append(idatatype+violgap*(len(famlevels)-1)/2)
            xticklabels_group.append(sibsim_titlestr)
                
            #this loop is just to plot the violins for each famlevel
            for ilev, famlevel in enumerate(famlevels):

                connmask=df_fam_performance['conntype']==conntype
                relmask=df_fam_performance['famlevel']==famlevel
                
                if unrelated_nonsibling_match and famlevel == famlevel_dict['unrelated']:
                    #special mode for sibling-only comparison (HCPDev), where the unrelated pairs are matched to the sibling pairs
                    relmask=df_fam_performance['famlevel']==famlevel_dict['sibling']
                    
                if all(np.array(datatype_list) == ['latent','raw']):
                    df_tmp=df_fam_performance[connmask & (df_fam_performance['datatype']==datatype) & relmask]
                elif all([d in df_fam_performance['datatype'].unique() for d in datatype_list]):
                    df_tmp=df_fam_performance[connmask & (df_fam_performance['datatype']==datatype) & relmask]
                elif all([d in df_fam_performance['datatype_full'].unique() for d in datatype_list]):
                    df_tmp=df_fam_performance[connmask & (df_fam_performance['datatype_full']==datatype) & relmask]
                else:
                    raise ValueError("datatype_list not found in df_fam_performance['datatype'] or df_fam_performance['datatype_full']")
                
                if df_tmp.shape[0]>1:
                    #just take the first one (doesn't matter what the famlevel0 was)
                    df_tmp=df_tmp.iloc[:1,:]

                xfam = df_tmp['sibling_similarity'].to_numpy()[0]
                xfam_samesex = df_tmp['samesex'].to_numpy()[0]
                xfam_agediff = df_tmp['agediff'].to_numpy()[0]

                if unrelated_nonsibling_match and famlevel == famlevel_dict['unrelated']:
                    #special mode for sibling-only comparison (HCPDev), where the unrelated pairs are matched to the sibling pairs
                    xfam=df_tmp['nonsibling_similarity'].to_numpy()[0]
                    
                y=xfam
                ymask=np.ones(len(y)).astype(bool)
                
                #for violin, only show subject pairs with the same sex and within a certain age range
                ymask=ymask & (xfam_agediff<=violin_max_agediff)
                if violin_samesex:
                    ymask=ymask & (xfam_samesex>0)
                    
                y=y[ymask]
                
                x=idatatype+ilev*violgap
                #x=isim+1
                #y=xfam-xother
                #y=y/np.std(xother)
                #y=(y-np.mean(xother))/np.std(xother)
                xticks.append(x)
                xticklabels.append(famlevel_str[famlevel])
                
                ynotnan=~np.isnan(y)
                
                hviol=plt.violinplot(y[ynotnan],positions=[x],showextrema=False,widths=violwidth)

                for p in hviol['bodies']:
                    #p.set_facecolor(colors[isim])
                    p.set_facecolor(datacolor)
                    p.set(edgecolor='k',linewidth=1)
                    p.set_alpha(.5) 
                    p.set_zorder(1)
                    #p.set_alpha(0) 
                
                #ykde=kde.evaluate(y)*violwidth*.5/np.max(kde.evaluate(y))
                #plt.plot(x+2*(np.random.random(len(y))-.5)*ykde*.9,y,'ko',markersize=3,alpha=.1)
                
                qmin,q25,q50,q75,qmax=np.percentile(y[ynotnan],[0,25,50,75,100])
                plt.vlines(x,qmin,qmax,color='k',linewidth=1,zorder=2)
                #plt.vlines(x,q25,q75,color='k',linewidth=9,zorder=2)
                
                plt.plot(x,np.nanmean(y),'ko',markerfacecolor='w',markersize=9,zorder=10)
        

                
        yl=ax.get_ylim()
        ysig=min(yl)+.05*np.abs(np.diff(yl))
        ysigheight=.025*np.abs(np.diff(yl))
        
        for istat,stat_info in enumerate(stat_list):
            if stat_info['conntype'] != conntype:
                continue
            
            statval=stat_info['value']
            #pval_corrected=stat_info['pvalue_corrected']
            #pval=stat_info['pvalue']
            pval_uncorrected=stat_info['pvalue']
            pval_corrected=stat_info['pvalue_corrected']
            
            if pval_test_info['corrected']:
                pval=pval_corrected
            else:
                pval=pval_uncorrected
            
            ilev=[i for i,l in enumerate(famlevels) if l==stat_info['famlevel']][0]
            ilev0=[i for i,l in enumerate(famlevels) if l==stat_info['otherlevel']][0]
            #xsig=[idatatype+ilev0*violgap,idatatype+ilev*violgap]
            xsig=[np.mean([0+ilev0*violgap,0+ilev*violgap]), np.mean([1+ilev0*violgap,1+ilev*violgap])]
            
            xsig=[[0+ilev0*violgap,0+ilev*violgap], [1+ilev0*violgap,1+ilev*violgap]]
            
            if pval >= pval_test_info['threshold']:
                print(' ',conntype,'%-20s' % (famlevel_str[famlevels[ilev]]+"-"+famlevel_str[famlevels[ilev0]]),pval_uncorrected, pval_corrected, statval)
            if pval < pval_test_info['threshold']:
                print('*',conntype,'%-20s' % (famlevel_str[famlevels[ilev]]+"-"+famlevel_str[famlevels[ilev0]]),pval_uncorrected, pval_corrected, statval)
                sigcolor=flavor2color(conntype)
                sigtxt='$\Delta$latent>$\Delta$observed'
                if statval < 0:
                    sigcolor='k'
                    sigtxt='$\Delta$latent<$\Delta$observed'
                
                #plt.plot([-5,xmin=xticks_group[0],xmax=xticks_group[1])
                
                if len(famlevels)<=2:
                    xsig[0]=np.mean(xsig[0])
                    xsig[1]=np.mean(xsig[1])
                else:
                    sigtxt=""
                drawbracket(xsig[0],xsig[1],ysig,ysigheight,linewidth=1,color=sigcolor,text=sigtxt,fontsize=ax_fontsize,label='sig%d' % (istat))
                
                
                #plt.plot([xsig[0],xsig[0],xsig[1],xsig[1]],[ysig+ysigheight,ysig,ysig,ysig+ysigheight],'-',color=sigcolor,linewidth=1,label='sig')
                #plt.text(np.mean(xsig),ysig,sigtxt,fontsize=ax_fontsize*1.5,color=sigcolor,horizontalalignment='center',verticalalignment='top',label='sigtext')
                ysig+=1.5*ysigheight
                
        labelrotation=45
        if labelrotation==45:
            xtick_ha='right'
        else:
            xtick_ha='center'    
        ax.set_xticks(xticks,xticklabels,fontsize=ax_fontsize,rotation=labelrotation,horizontalalignment=xtick_ha)
        ax.tick_params(axis='y',labelsize=ax_fontsize)
        #ax.grid(axis='y',linewidth=.25)
        ax.grid(axis='y',linewidth=1,color='k',alpha=.15)
        ax.xaxis.set_zorder(-1)
        ax.yaxis.set_zorder(-1)

        title_str=conntype_str.replace("_"," ")
        title_str=title_str.replace("enc","fusion")
        #title_str=title_str.replace("FCSC","FC+SC")
        ax.set_title(title_str,fontsize=title_fontsize,color=flavor2color(conntype))

        
        axlist.append(ax)
        #ax=plt.gca()

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.xaxis.set_ticks_position("bottom")
        #get outer extent of ax 
        if labelrotation==45:
            ax2.spines["bottom"].set_position(("axes", -0.25))
        else:
            ax2.spines["bottom"].set_position(("axes", -0.1))
        ax2.tick_params(axis='x',length=0)
        ax2.spines["bottom"].set_visible(False)
        ax2.set_xticks(xticks_group,xticklabels_group,fontsize=ax_fontsize_group)
        
        #plt.show()
    #add a green and magent dummy line to current axis to get a legend:

    if normalize_similarity:    
        ylabel='Normalized similarity'
    else: 
        ylabel='Pearson $r$'
        

    axlist[0].set_ylabel(ylabel,fontsize=ax_fontsize)
        
    if do_plot_samesex_lines:
        # add green and magenta dummy lines to get a legend
        plt.plot([], [], 'm-', label='Same Sex')
        plt.plot([], [], 'g-', label='Opposite Sex')
        plt.legend(fontsize=ax_fontsize)

    #make yaxes uniform, and shift significance markers
    yl=np.vstack([x.get_ylim() for x in axlist])
    yl=[np.min(yl[:,0]),np.max(yl[:,1])]

    #yl=[-6.1,14] #for comparison with the corr.post
    
    #ysig=min(yl)+.125*np.abs(np.diff(yl))
    ysig=min(yl)+.05*np.abs(np.diff(yl))
    ysigheight=.02*np.abs(np.diff(yl))
    ysigtextgap=.04*np.abs(np.diff(yl))
    for ax in axlist:
        yl0=ax.get_ylim()
        ax.set_ylim(yl)
        yshift=yl[0]-yl0[0]
        ysignums=[int(str(ch.get_label()).replace("txtsig","")) for ch in ax.get_children() if str(ch.get_label()).startswith('txtsig')]
        #ysiggap=.02*np.abs(np.diff(yl))
        ysiggap=.05*np.abs(np.diff(yl))
        for isig,signum in enumerate(ysignums):
            
            
            ysigmin=min([min(ch.get_ydata()) for ch in ax.get_children() if ch.get_label()=='sig%d' % (signum)])
            
            for hsigtext in [ch for ch in ax.get_children() if ch.get_label()=='txtsig%d' % (signum)]:
                #hsigtext.set_y(ysig-ysigtextgap)
                if len(hsigtext.get_text())==0:
                    yextra=0
                    hsigtext.set_visible(False)
                else:
                    yextra=ysigtextgap*1.5
                y=hsigtext.get_position()[1]
                #y=y+yshift+ysiggap*(isig)
                y=ysig+ysiggap*(isig)+ysigtextgap
                hsigtext.set_y(y)

            for hsig in [ch for ch in ax.get_children() if ch.get_label()=='sig%d' % (signum)]:
                y=hsig.get_ydata()
                #y=y+yshift+ysiggap*(isig)+yextra
                y=y-ysigmin+ysig+ysiggap*(isig)+yextra
                hsig.set_ydata(y)
                #hsig.set_ydata([ysig+ysigheight,ysig,ysig,ysig+ysigheight]) #from hcpdev.sib

        
    for ax in axlist:
        ax.set_yticks(np.arange(np.round(yl[0]/2)*2,np.round(yl[1]/2)*2,2))
        
    plt.tight_layout()

    #fig.savefig("/Users/kwj5/Research/krakencoder/demographic_prediction_figures/test.pdf",dpi=300)
    #fig.savefig("/Users/kwj5/Research/krakencoder/demographic_prediction_figures/test.png",dpi=300)
    #_=crop_background("../notes/test.png",outfile="../notes/test.png")
    
    if output_file is not None:
        fig.savefig(output_file,dpi=figdpi)
    
    return fig
    

#################################################################################################
#################################################################################################
def plot_prediction_barplot(df_performance, demo_var_list=None, input_flavors=None, siglist=[], pval_thresh=.01,
                             enc_is_cosinesim=False, enc_is_pc=False, rawdata=False, 
                             fontsize_datavals_override=None, figsize=None,
                             plottitle=None, figdpi=100, closeall=True, 
                             legend_columns=None,legend_location='upper right',legend_bbox_to_anchor=(1, .9),
                             output_file=None):
    """
    Plot the prediction performance as a barplot with significance whiskers for each demographic variable
    
    Parameters:
    df_performance: pd.DataFrame. Dataframe with prediction performance results (output of fit_prediction_model)
    demo_var_list: list of str. Demographic variables to plot (eg Age, Sex, ...)
    input_flavors: list of str. Input flavors to plot (eg 'enc_FCmean')
    siglist: list of str. List of significance tests to plot (output of prediction_comparison_significance_tests)
    pval_thresh: float. Threshold for significance markers (default=.01)
    enc_is_cosinesim: bool. If True, the input flavors are cosine similarities (just affects text labels)
    enc_is_pc: bool. If True, the input flavors are principal components (just affects text labels)
    rawdata: bool. If True, the input flavors are raw data (just affects text labels)
    plottitle: str. Title of the plot (optional)
    figdpi: int. DPI of the output figure (default=100)
    closeall: bool. If True (default), close all previous figures before plotting the new one (useful in jupyter notebooks)
    output_file: str. Filename to save the output figure (optional)
    
    Returns:
    fig. The output figure
    """

    if not plottitle:
        plottitle=''
    input_flavor_colors=[adjust_saturation(list(plt.cm.rainbow(x)),.75) for x in np.linspace(0,1,len([xx for xx in input_flavors if not "ooi22" in xx and not "li19" in xx]))]

    if closeall:
        plt.close('all') #close any previous figures (especially useful in jupyter notebooks)
    
    if figsize is not None:
        fig=plt.figure(figsize=figsize,facecolor='w',dpi=figdpi) #for the Age+Sex+Cog
    elif len(demo_var_list) <= 3:
        fig=plt.figure(figsize=(15,5),facecolor='w',dpi=figdpi) #for the Age+Sex+Cog
        #add dpi=300 for publication?
    else:
        fig=plt.figure(figsize=(20,5),facecolor='w',dpi=figdpi)

    dataname_toplot_list=[None]
    if 'dataname' in df_performance:
        dataname_toplot_list=df_performance['dataname'].unique()
        if len(dataname_toplot_list) == 1:
            dataname_toplot_list=[None]

    xticks=np.array([])
    xticklabels=[]

    #get all the y values so we can set global limits
    y_all_cc=df_performance['holdout_cc'][df_performance['input_name'].isin(input_flavors) & df_performance['output_name'].isin(demo_var_list)]
    #y_all_cc=df_performance['holdout_rmse'][df_performance['input_name'].isin(input_flavors) & df_performance['output_name'].isin(demo_var_list)]
    y_all=df_performance['holdout_score'][df_performance['input_name'].isin(input_flavors) & df_performance['output_name'].isin(demo_var_list)]
    y_all[~y_all_cc.isna()]=y_all_cc[~y_all_cc.isna()]

    yl=[np.min(y_all),np.max(y_all)]
    yl=(yl-np.mean(yl))*1.3 + np.mean(yl) #pad ylim

    #yl=[-.1,1.15]
    yl=[-.15,1.15]

    ytickspacing=.1
    #yticks=np.arange(np.floor(yl[0]/ytickspacing)*ytickspacing, np.ceil(yl[1]/ytickspacing)*ytickspacing, ytickspacing)
    yticks=np.arange(0,1+ytickspacing, ytickspacing)


    do_show_plot_value_text=True
    fontsize_datavals=12 #shrink this based on bar width later
    fontsize=14

    htxt_list=[]
    hbar_list=[]

    hbar_list_info=[]
    
    xstart=0
    for idemo, demo_var_name in enumerate(demo_var_list):
        df_tmp=df_performance[df_performance['output_name']==demo_var_name]
        
        xticks_thisvar=[]
        xtype_thisvar=[]
        
        for ix,xtype in enumerate(input_flavors):
            ix_thisvar=len(xticks_thisvar)
            xticks_thisvar.append(ix_thisvar+xstart)
            xtype_thisvar.append(xtype)
            
            for idata,whichdata in enumerate(dataname_toplot_list):
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
                
                
                #ix_thisvar=len(xticks_thisvar)
                #xticks_thisvar.append(ix_thisvar+xstart)
                #xtype_thisvar.append(xtype)
                
                thiscolor=np.array(input_flavor_colors[ix])
                thiscolor_alphawhite=np.clip(thiscolor*.1+np.ones(4)*.9 + np.array([0,0,0,1]),0,1)
                thiscolor_alphablack=np.clip(thiscolor*.25+np.zeros(4)*.75 + np.array([0,0,0,1]),0,1)
                
                barparams=dict(color=thiscolor)
                if "ooi22" in xtype or "li19" in xtype:
                    #yeo values
                    #plt.plot(ix_thisvar+xstart+np.zeros(2),np.array([0,y]),'k-',color='gray')
                    #plt.plot(ix_thisvar+xstart,y,'o',markersize=10,color='gray',markerfacecolor='white')
                    barparams['color']='lightgray'
                    
                    txtcolor='gray'
                else:
                    #plt.plot(ix_thisvar+xstart+np.zeros(2),np.array([0,y]),'k-')
                    #plt.plot(ix_thisvar+xstart,y,'o',markersize=10)
                    txtcolor='black'
                
                if len(dataname_toplot_list)==1:
                    barwidth=.8
                    barwidth_thisvar=barwidth/len(dataname_toplot_list)
                    barcenter_thisvar=ix_thisvar+xstart+idata*barwidth_thisvar-barwidth_thisvar/2
                else:    
                    barwidth=.8
                    barwidth_thisvar=barwidth/len(dataname_toplot_list)
                    #barcenter_thisvar=ix_thisvar+xstart+idata*barwidth_thisvar-barwidth_thisvar/2
                    barcenter_thisvar=ix_thisvar+xstart+idata*barwidth_thisvar-barwidth_thisvar/2
                    barwidth_thisvar=barwidth_thisvar*(.9/.8)
                    barparams['edgecolor']=thiscolor_alphablack
                    barparams['linewidth']=1
                    if whichdata=='enc':
                        pass
                    elif whichdata.startswith('raw'):
                        #barparams['edgecolor']='k'
                        barparams['color']=np.clip(thiscolor*.25+np.ones(4)*.75 + np.array([0,0,0,1]),0,1)
                        barparams['edgecolor']=thiscolor_alphablack
                    elif whichdata.startswith('encsplit'):
                        #barparams['edgecolor']='k'
                        barparams['color']=np.clip(thiscolor*.5+np.ones(4)*.5 + np.array([0,0,0,1]),0,1)
                        barparams['edgecolor']=thiscolor_alphablack
                
                hb=plt.bar(barcenter_thisvar,y,width=barwidth_thisvar,**barparams,label=whichdata,zorder=1)
                hbar_list.append(hb)
                
                hbar_list_info.append(dict(xtype=xtype,demo_var_name=demo_var_name,dataname=whichdata,x=barcenter_thisvar))
    
                #instead of a bar plot, lets plot a violin plot
                #plottype='violin'
                #plottype='bar'
                plottype='barwhisker'
                
                #qmin,q25,q50,q75,qmax=np.percentile(y_list,[0,25,50,75,100])
                qmin,q25,q50,q75,qmax=np.percentile(y_list,[5,25,50,75,95])
                
                if plottype == 'violin':
                    #plt.vlines(ix_thisvar+xstart,0,y,color=thiscolor*np.array([1,1,1,.5]),linewidth=1,linestyles=':',zorder=0)
                    
                    hviol=plt.violinplot(positions=[ix_thisvar+xstart],dataset=y_list.astype(float),showmeans=False,showmedians=False,showextrema=False)
                    plt.vlines(ix_thisvar+xstart,np.percentile(y_list,25),np.percentile(y_list,75),color=thiscolor_alphablack,linewidth=5,zorder=1)
                    plt.plot(ix_thisvar+xstart,np.mean(y_list),'o',markerfacecolor=thiscolor_alphawhite,markeredgewidth=0,markersize=5,zorder=2)
                    hviol['bodies'][0].set_facecolor(thiscolor)
                    hviol['bodies'][0].set_alpha(1)
                    #hviol['cmeans'].set_color('black')
                    [x.set_alpha(0) for x in hb]
                    
                elif plottype == 'barwhisker':
                    #plt.vlines(ix_thisvar+xstart,qmin,qmax,color='k',linewidth=.75,zorder=1)
                    #plt.vlines(ix_thisvar+xstart,q25,q75,color='k',linewidth=3,zorder=2)
                    plt.vlines(barcenter_thisvar+np.zeros(2),[qmin,q75],[q25,qmax],color=[0,0,0,.5],linewidth=.75,zorder=1)
                    plt.vlines(barcenter_thisvar,q25,q75,color=[0,0,0,.5],linewidth=3,zorder=2)
                            
                #plot individual resamples
                #plt.plot(ix_thisvar+xstart+.1*(np.random.random(len(y_list))-.5),y_list,'o',markersize=3,color='k',alpha=.5)
                
                if do_show_plot_value_text:
                    #ystr="%.3f" % (y)
                    ystr="%.2f" % (y)
                    #txty=y-.05
                    #va='top'
                    txty=y+.01
                    va='bottom'
                    if y<.1:
                        txty=y+.05
                        va='bottom'
                        
                    #txty=0
                    #va='top'
                    txty=-.05
                    va='center'
                    #if qmax > .8:
                    #txty=max(qmin-.01,0)
                    #va='top'
                    #else:
                    #    txty=max(qmax+.01,0)
                    #    va='bottom'
                    
                    txtrotation=0
                    txtbbox=dict(facecolor=[1,1,1,.7],edgecolor='none',pad=1)
                    if len(dataname_toplot_list)>1:
                        txtrotation=45
                        #txtrotation=0
                        txtbbox=None
                        #txty=0
                        #va='bottom'
                        txty=0
                        va='top'
                        
                    ht=plt.text(barcenter_thisvar,txty,ystr,
                            fontsize=fontsize_datavals,horizontalalignment='center',verticalalignment=va,rotation=txtrotation,
                            backgroundcolor=[1,1,1,.5],color=txtcolor,bbox=txtbbox,zorder=200)
                    htxt_list.append(ht)
                    #width_data_units = ax.transData.inverted().transform((bbox.width, 0))[0]
                    #print(ht.get_tightbbox(),ht.get_tightbbox().width)
                    #ht_width=plt.gca().transData.inverted().transform(([ht.get_tightbbox().x0,ht.get_tightbbox().x1],[0,0]))[0,:]
                    #ht_width=np.diff(ht_width)
                    #print(ht_width)
                    #hb_width=hb.get_width()
                    #if ht_width>hb_width:
                    #    print(fontsize_datavals*hb_width/ht_width)
                    #    ht.set_fontsize(fontsize_datavals*hb_width/ht_width)
                    
    #tperm_t=np.array([x[0] for x in tperm])
    #tperm_p=np.array([x[1] for x in tperm])

                
        xticks=np.concatenate((xticks,xticks_thisvar))
        xticklabels+=[pretty_input_name(x, enc_is_cosinesim=enc_is_cosinesim,
                                        enc_is_pc=enc_is_pc,
                                        rawdata=rawdata) for x in xtype_thisvar]
        
        demo_titlestr=pretty_demo_name(demo_var_name)
        
        demo_title_sep=' '
        if False and len(demo_var_list) > 5:
            demo_title_sep='\n'
            
        if df_tmp['metric'].to_numpy()[0] == 'balanced_accuracy':
            demo_titlestr+=demo_title_sep+'(Bal. Acc)'
        elif df_tmp['metric'].to_numpy()[0] == 'accuracy':
            demo_titlestr+=demo_title_sep+'(Acc)'
        else:
            #demo_titlestr+=demo_title_sep+'(corrcoef)'
            demo_titlestr+=demo_title_sep+r'($r$)'

        plt.gca().text(xstart+len(xtype_thisvar)/2-.5,yl[1]*.97,demo_titlestr,
                    fontsize=fontsize,horizontalalignment='center',verticalalignment='top',backgroundcolor=[1,1,1,.75],zorder=100)
        
        xstart+=len(xtype_thisvar)+.5
        
    plt.xticks(xticks,xticklabels,horizontalalignment='center',rotation=0)

    for xtl in plt.gca().get_xticklabels():
        if 'ooi22' in xtl.get_text() or 'li19' in xtl.get_text():
            xtl.set_color('gray')
            
    plt.ylabel('Bal.Acc or Pearson\'s $r$')
    plt.grid(axis='y',linewidth=.25)
    plt.gca().set_axisbelow(True)
    plt.yticks(yticks)
    plt.ylim(yl)
    plt.xlim([np.min(xticks)-1,np.max(xticks)+1])

    titlefontsize=fontsize
    plt.title(plottitle+"\n",fontsize=titlefontsize)

    ax=fig.gca()


    for item in ([ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
        
    #shrink data label fonts to fit bar width
    new_fontsize_datavals=fontsize_datavals
    for i in range(len(hbar_list)):
        ht_width=htxt_list[i].get_tightbbox(renderer=fig.canvas.get_renderer()).width
        hb_width=hbar_list[i].get_children()[0].get_tightbbox(renderer=fig.canvas.get_renderer()).width
        if ht_width>hb_width:
            new_fontsize_datavals=min(new_fontsize_datavals,htxt_list[i].get_fontsize()*hb_width/ht_width)

    new_fontsize_datavals=12
    if len(dataname_toplot_list) > 3:
        new_fontsize_datavals=8
    if fontsize_datavals_override:
        new_fontsize_datavals=fontsize_datavals_override
    for ht in htxt_list:
        ht.set_fontsize(new_fontsize_datavals)
    
    for isig, sig in enumerate(siglist):
        if sig['p']>=pval_thresh:
            print('ns','%.8f'% (sig['p']),'%.8f' % (sig['p_corrected']), sig['sig1']['dataname'],sig['sig1']['demo_var_name'],sig['sig1']['xtype'],sig['sig2']['xtype'])
            continue
        else:
            print('  ','%.8f'% (sig['p']),'%.8f' % (sig['p_corrected']), sig['sig1']['dataname'],sig['sig1']['demo_var_name'],sig['sig1']['xtype'],sig['sig2']['xtype'])
        
        hbar_info_sig1=[x for x in hbar_list_info if x['xtype']==sig['sig1']['xtype'] and x['demo_var_name']==sig['sig1']['demo_var_name'] and x['dataname']==sig['sig1']['dataname']][0]
        hbar_info_sig2=[x for x in hbar_list_info if x['xtype']==sig['sig2']['xtype'] and x['demo_var_name']==sig['sig2']['demo_var_name'] and x['dataname']==sig['sig2']['dataname']][0]
        
        sig['sig1']['x']=hbar_info_sig1['x']
        sig['sig2']['x']=hbar_info_sig2['x']
        
        #continue #skip all significance markers
        sigcolor='r'
        sigtxt=r'$\ast$'
        if sig['sig1']['dataname']==sig['sig2']['dataname']:
            #enc vs enc, raw vs raw
            if sig['sig1']['dataname']=='raw':
                sigtxt='r'
                sigcolor='b'
            sigbracket=True
        else:
            #enc vs raw
            sigtxt='*'
            if sig['statval']>0:
                #latent > raw
                sigcolor='k'
            else:
                #latent < raw
                #sigcolor='r'
                sigcolor='k'
            sigbracket=False
        #print(sig['p'],sig['sig1']['dataname'],sig['sig1']['demo_var_name'],sig['sig1']['xtype'],sig['sig2']['xtype'])
        ysig=max([sig['sig1']['y'],sig['sig2']['y']])
        ysig=ysig+.05
        xsig=[sig['sig1']['x'],sig['sig2']['x']]
        
        #sigtxt_size=fontsize*2
        sigtxt_size=fontsize
        #sigtxt=''
        if sigtxt != '':
            hsigtxt=plt.text(np.mean(xsig),ysig,sigtxt,color=sigcolor,fontsize=sigtxt_size,horizontalalignment='center',verticalalignment='center')
            fig.canvas.draw()                
            hsigbbox=hsigtxt.get_window_extent() #get_tightbbox(renderer=fig.canvas.get_renderer())
            axtransform=plt.gca().transData.inverted()
            hsigbbox=axtransform.transform(hsigbbox)
            
        if sigbracket:
            ysigtick=(max(yl)-min(yl))*.02
            #plt.plot([xsig[0],xsig[0],hsigbbox[0][0]],[ysig-ysigtick, ysig, ysig],'-',color=sigcolor,linewidth=1)
            #plt.plot([hsigbbox[1][0],xsig[1],xsig[1]],[ysig,ysig,ysig-ysigtick],'-',color=sigcolor,linewidth=1)
            plt.plot([xsig[0],xsig[0],xsig[1],xsig[1]],[ysig-ysigtick,ysig,ysig,ysig-ysigtick],'-',color=sigcolor,linewidth=1)
            
    # add legend
    if len(dataname_toplot_list)>1:
        hbar_legend_list=[]
        hbar_legend_names=[]
        barparams=dict(edgecolor='k')
        for idata,whichdata in enumerate(dataname_toplot_list):
            if whichdata == 'encsplit':
                thiscolor=[.5,.5,.5,1]
                thisname='latent split'
                
            elif whichdata.startswith('enc'):
                thiscolor=[.25,.25,.25,1]
                if(len(whichdata)==len('enc')):
                    thisname='latent'
                else:
                    thisname=whichdata.replace("enc","latent ")
                
            elif whichdata.startswith('raw'):
                thiscolor=[.85,.85,.85,1]
                if(len(whichdata)==len('raw')):
                    #thisname='raw'
                    thisname='observed'
                else:
                    #thisname=whichdata.replace("raw","raw ")
                    thisname=whichdata.replace("raw","observed ")
            
            hbar_legend_list.append(plt.bar(0,np.nan,color=thiscolor,**barparams))
            hbar_legend_names.append(thisname)    
        if len(hbar_legend_names)>3:
            legend_ncol=len(hbar_legend_names)
            #legend_ncol=len(hbar_legend_names)//2
        else:
            legend_ncol=1
        if legend_columns is not None:
            legend_ncol=legend_columns
        hl=plt.legend(hbar_legend_list,hbar_legend_names,loc=legend_location,bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize, title='Prediction input',ncol=legend_ncol)
        hl.get_title().set_fontsize(fontsize)

    #output_dir=os.path.join(krakendir(),'demographic_prediction_figures')
    #output_filename="performance_%s_resample%d_%%T" % (whichstudy,max(df_performance['resample'])+1)
    #output_filename="performance_%s_resample%d_%%T" % (whichstudy,len(y_list))
    
    if output_file is not None:
        fig.savefig(output_file,dpi=figdpi)
    
    return fig
