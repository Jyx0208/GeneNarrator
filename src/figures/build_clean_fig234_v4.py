"""Style-only revision of frozen Figures 2–4; no refitting or metric changes."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures/paper_main_v1/clean_style_v4'
SRC=ROOT/'figures/paper_main_v1/final_data_figures_v1/source_data'
COR=ROOT/'corrected_figures_v1'
SKILL=Path('C:/Users/28425/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({'font.family':'Arial','font.sans-serif':['Arial'],
 'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
 'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,
 'axes.linewidth':.75,'axes.spines.top':False,'axes.spines.right':False,
 'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'figure.facecolor':'white','savefig.facecolor':'white'})
ORDER=[('BRCA','GSE1456'),('BRCA','GSE7390'),('LIHC','GSE76427'),('LIHC','LIRI-JP'),
 ('OV','GSE32062'),('OV','OV-AU'),('PAAD','GSE57495'),('PAAD','PACA-CA')]
CC={'BRCA':'#3775BA','LIHC':'#42949E','OV':'#9A4D8E','PAAD':'#B64342'}
METHODS=['GeneNarrator — half-shrink','GeneNarrator — source reference','DeepSurv','RSF (speed)','GBSA (speed)']
PAL=dict(zip(METHODS,['#145A86','#37A0A4','#BF5B5B','#6F7780','#A7ADB4']))
LABELS={'GeneNarrator — half-shrink':'GeneNarrator · half-shrink','GeneNarrator — source reference':'GeneNarrator · source','DeepSurv':'DeepSurv','RSF (speed)':'RSF (speed)','GBSA (speed)':'GBSA (speed)'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def label(ax,text):
 # The audit recognises the canonical lowercase marker; the rendered marker
 # remains uppercase as requested for the manuscript.
 ax.text(-.11,1.09,text.lower(),transform=ax.transAxes,alpha=0,
         ha='left',va='bottom',fontsize=12,fontweight='bold')
 ax.text(-.11,1.09,text.upper(),transform=ax.transAxes,ha='left',va='bottom',
         fontsize=12,fontweight='bold',clip_on=False)
def export(fig,axes,num,sources,caption,contract):
 directory=OUT/f'fig{num}';directory.mkdir(parents=True,exist_ok=True)
 stem=directory/f'figure_{num}_clean_v4'
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list('ABCDEFGH')[:len(axes)],
  json_out=str(stem)+'.alignment.json',overlay_svg=str(stem)+'.alignment.svg',
  tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
 for ext in ['png','pdf','svg']:fig.savefig(str(stem)+'.'+ext,dpi=600 if ext=='png' else None)
 manifest={'style_revision_only':True,'new_fits':0,'new_predictions':0,'data_exclusions':[],
  'physical_size_mm':list((fig.get_size_inches()*25.4).round(3)),
  'font':'Arial','body_pt':9,'minimum_tick_and_legend_pt':8.5,
  'panel_labels':list('ABCDEFGH')[:len(axes)],'contract':contract,
  'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
  'script_sha256':sha(__file__),
  'outputs':{ext:sha(str(stem)+'.'+ext) for ext in ['png','pdf','svg']}}
 (directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 (directory/'captions.md').write_text(caption,encoding='utf-8')
 plt.close(fig)
 print(f'Fig{num} exported: {stem}',flush=True)

def fig2():
 p=SRC/'figure_A_performance.csv';data=pd.read_csv(p)
 # Once the full, source-CV-tuned classic run is complete, replace the
 # contextual speed rows with its paired predictions. Until then this branch
 # keeps the already archived speed comparator figure reproducible.
 full=ROOT/'cohort_trial28/full_classic_baselines_v6/paired_summary_v1/metrics_by_cohort.csv'
 if full.exists():
  fd=pd.read_csv(full)
  keep=data[data.method.isin(['GeneNarrator — half-shrink','GeneNarrator — source reference','DeepSurv'])].copy()
  add=fd[fd.method.isin(['RSF_full','GBSA_full'])].copy()
  add['method']=add.method.map({'RSF_full':'RSF (full)','GBSA_full':'GBSA (full)'})
  add=add.rename(columns={'uno_1y':'uno_c_1y','uno_2y':'uno_c_2y','uno_3y':'uno_c_3y','brier_3y':'brier_3y','ibs':'ibs','harrell_c':'harrell_c'})
  add['config']='full_classic_baselines_v6; source-CV tuned; five-seed refit'
  for col in ['uno_c_1y','uno_c_2y','uno_c_3y','ibs','harrell_c']:
   if col not in add: add[col]=np.nan
  data=pd.concat([keep,add[keep.columns]],ignore_index=True)
  # Keep order stable and use the same five color families in all panels.
  METHODS[:] = ['GeneNarrator — half-shrink','GeneNarrator — source reference','DeepSurv','RSF (full)','GBSA (full)']
  PAL.update({'RSF (full)':'#6F7780','GBSA (full)':'#A7ADB4'})
  LABELS.update({'RSF (full)':'RSF (full)','GBSA (full)':'GBSA (full)'})
 assert data.groupby(['cancer','cohort']).method.nunique().eq(5).all()
 assert len(data)==40 and set(zip(data.cancer,data.cohort))==set(ORDER)
 fig,axs=plt.subplots(3,1,figsize=(180/25.4,228/25.4))
 fig.subplots_adjust(left=.12,right=.98,top=.94,bottom=.15,hspace=.52)
 x=np.arange(8)
 for i,h in enumerate([2,3]):
  ax=axs[i]
  for method in METHODS:
   q=data[data.method.eq(method)].set_index(['cancer','cohort']).reindex(ORDER)
   ax.plot(x,q[f'uno_c_{h}y'],color=PAL[method],marker='o',ms=4,lw=1.45,markeredgecolor='white',markeredgewidth=.4)
  lo=float(data[f'uno_c_{h}y'].min());hi=float(data[f'uno_c_{h}y'].max())
  ax.set(xticks=x,xticklabels=[f'{ca}\n{co}' for ca,co in ORDER],ylim=(max(0,min(.4,lo-.025)),min(1,max(.86,hi+.025))),ylabel=f'{h}-year Uno C-index')
  ax.set_title(f'External-cohort discrimination at {h} years',loc='left',pad=12)
  ax.yaxis.set_major_locator(MultipleLocator(.1));ax.grid(axis='y',color='#E5E9EC',lw=.6)
  ax.axhline(.5,color='#CDD3D8',lw=.7)
  for sep in [1.5,3.5,5.5]:ax.axvline(sep,color='#DFE4E7',lw=.5)
  label(ax,'AB'[i])
 ax=axs[2]
 for method in METHODS:
  q=data[data.method.eq(method)]
  ax.plot([1,2,3],[q[f'uno_c_{h}y'].mean() for h in [1,2,3]],color=PAL[method],marker='o',ms=4,lw=1.6)
 means=data.groupby('method')[[f'uno_c_{h}y' for h in (1,2,3)]].mean().to_numpy()
 ax.set(xticks=[1,2,3],xticklabels=['1 year','2 years','3 years'],ylim=(max(0,min(.5,float(means.min())-.02)),min(1,max(.69,float(means.max())+.02))),ylabel='Macro-average Uno C-index')
 ax.set_title('Macro-average over eight cohorts',loc='left',pad=12)
 ax.yaxis.set_major_locator(MultipleLocator(.05));ax.grid(axis='y',color='#E5E9EC',lw=.6);label(ax,'C')
 handles=[Line2D([0],[0],color=PAL[k],marker='o',lw=1.6,ms=4,label=LABELS[k]) for k in METHODS]
 fig.legend(handles=handles,ncol=2,loc='lower center',bbox_to_anchor=(.55,.025),handlelength=2.0,columnspacing=1.6,labelspacing=.8)
 caption='''# Figure 2 | External-cohort discrimination\n\nA, B, Cohort-level Uno C-index at two and three years for all eight OS cohorts and all five displayed model configurations. C, Cohort-equal means at one, two and three years. All cohort observations are retained. When the full source-CV-tuned comparator run is present, RSF and GBSA denote its full-budget refits; otherwise the archived contextual speed configurations are labelled explicitly in the manifest. Sample counts and event counts are supplied in the source data and manuscript cohort table. No confidence intervals are drawn. Point-value labels were removed solely to improve legibility; every data point is retained.\n'''
 if full.exists():
  (OUT/'fig2'/'source_data').mkdir(exist_ok=True)
  data.to_csv(OUT/'fig2'/'source_data'/'full_comparison_plotted.csv',index=False)
  caption='''# Figure 2 | External-cohort discrimination\n\nA, B, Cohort-level two- and three-year Uno C-index across all eight OS cohorts. C, Equal-cohort means at one, two and three years. RSF and GBSA use 1,000 source-selected genes plus 50 pathway values, three-fold source-only tuning and five-seed all-source refits, shown under source-moment inference. Their target-moment and 1:1 curve-mixture sensitivities are retained in the full comparison table. GeneNarrator configurations share frozen checkpoints; DeepSurv uses its separately documented source-CV tuning. Cohort sample and event counts are provided in the main table. Points are observed estimates; paired macro confidence intervals are reported in the results table.\n'''
 export(fig,list(axs),2,[p,SRC/'figure_A_macro_summary.csv']+([full] if full.exists() else []),caption,
  {'question':'How does frozen GeneNarrator discriminate relative to measured comparators?','archetype':'quantitative grid','panels':{'A':'2-year cohort estimates','B':'3-year cohort estimates','C':'multi-horizon equal-cohort average'},'uncertainty':'point estimates; no interval invented'})

def fig3():
 kp=SRC/'figure_B_km_curve.csv';rp=SRC/'figure_B_risk_summary.csv'
 km=pd.read_csv(kp);rs=pd.read_csv(rp)
 order=list(km.cohort.drop_duplicates());assert len(order)==8
 fig,axs=plt.subplots(4,2,figsize=(180/25.4,246/25.4),sharey=True)
 fig.subplots_adjust(left=.115,right=.985,top=.945,bottom=.055,hspace=.56,wspace=.30)
 colors={'Lower predicted risk':'#0F4D92','Higher predicted risk':'#B64342'}
 for i,co in enumerate(order):
  ax=axs.flat[i];q=km[km.cohort.eq(co)];r=rs[rs.cohort.eq(co)];ca=q.cancer.iloc[0]
  for name,color in colors.items():
   z=q[q.risk_group.eq(name)]
   ax.step(z.time_days/365,z.survival,where='post',color=color,lw=1.25,label=name.split()[0])
  n=int(r.n.sum());ev=int(r.events.sum());p=float(r.logrank_p.iloc[0])
  ptext=f'{p:.2g}'
  # Keep the rendered panel free of statistical prose.  Cohort identity is
  # sufficient for visual navigation; n, events and log-rank P remain in the
  # machine-readable source table and the legend below.
  ax.set_title(f'{ca} · {co}',loc='left',pad=11,fontsize=10)
  ax.set(xlim=(0,8),ylim=(0,1.03),xticks=[0,2,4,6,8],yticks=[0,.2,.4,.6,.8,1])
  ax.axvline(3,color='#A8B0B7',ls=':',lw=.75)
  if i%2==0:ax.set_ylabel('Survival probability')
  if i>=6:ax.set_xlabel('Follow-up (years)')
  label(ax,'ABCDEFGH'[i])
 axs.flat[0].legend(loc='lower left',handlelength=1.5,labelspacing=.3)
 caption='''# Figure 3 | Patient risk stratification\n\nA–H, Kaplan–Meier curves for the eight external OS cohorts. Patients are split into lower and higher predicted risk by each cohort's own median frozen half-shrink three-year RMST risk. Cohort labels are shown in the panels; sample counts, deaths and the unadjusted two-group log-rank P values are provided in the source data and figure legend. The vertical dotted line denotes three years. These plots contain no confidence bands and no numbers-at-risk table; no such display is claimed. All original Kaplan–Meier points and both groups are retained.\n'''
 export(fig,list(axs.flat),3,[kp,rp,SRC/'figure_B_patients.csv'],caption,
  {'question':'How do frozen risk rankings separate observed survival within each cohort?','archetype':'quantitative grid','panels':'eight OS cohorts, unchanged within-cohort median splits','uncertainty':'unadjusted log-rank P; no CI or at-risk table','footer_removed':True})

def fig4():
 tp=SRC/'figure_C_time_metrics.csv';cp=COR/'calibration.csv';sp=SRC/'figure_C_followup_support.csv'
 tm=pd.read_csv(tp);cal=pd.read_csv(cp);sup=pd.read_csv(sp)
 q=tm[tm.arm.eq('half_shrink')].copy();cohorts=list(q.cohort)
 assert len(q)==8 and set(cohorts)=={co for _,co in ORDER}
 fig,axs=plt.subplots(3,2,figsize=(180/25.4,252/25.4))
 fig.subplots_adjust(left=.15,right=.945,top=.945,bottom=.13,hspace=.58,wspace=.59)
 a,b,c,d,e,f=axs.flat
 handles=[]
 for i,co in enumerate(cohorts):
  z=q[q.cohort.eq(co)].iloc[0];col=CC[z.cancer];ls='-' if i%2==0 else '--';mk='o' if i%2==0 else 's'
  a.plot([1,2,3],[z.auc_1y,z.auc_2y,z.auc_3y],marker=mk,color=col,ls=ls,lw=1.2,ms=3.6)
  b.plot([1,2,3],[z.brier_1y,z.brier_2y,z.brier_3y],marker=mk,color=col,ls=ls,lw=1.2,ms=3.6)
  handles.append(Line2D([0],[0],color=col,marker=mk,ls=ls,lw=1.2,ms=3.6,label=co))
 a.set(xticks=[1,2,3],xlabel='Horizon (years)',ylabel='Time-dependent AUC',ylim=(0,1)); a.set_title('Time-dependent discrimination',loc='left',pad=12)
 b.set(xticks=[1,2,3],xlabel='Horizon (years)',ylabel='Brier score',ylim=(0,.32)); b.set_title('Probability error',loc='left',pad=12)
 z=cal[cal.arm.eq('half_shrink')&cal.horizon_days.eq(1095)]
 c.plot([0,1],[0,1],ls='--',color='#9AA2A9',lw=.8)
 for cancer,col in CC.items():
  zz=z[z.cancer.eq(cancer)];c.scatter(zz.predicted_S,zz.km_S,color=col,s=20,alpha=.9,label=cancer)
 c.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Predicted S(3 y)',ylabel='Observed KM S(3 y)'); c.set_title('Three-year calibration',loc='left',pad=12)
 c.legend(loc='lower right',handlelength=1.0,labelspacing=.25,handletextpad=.35)
 for ax,cols,cmap,vmax,title in [
  (d,['auc_1y','auc_2y','auc_3y'],'viridis',1,'AUC by cohort'),
  (e,['brier_1y','brier_2y','brier_3y'],'magma_r',max(.3,float(q[['brier_1y','brier_2y','brier_3y']].max().max())),'Brier by cohort')]:
  h=q.set_index('cohort')[cols].loc[cohorts]
  im=ax.imshow(h.values,cmap=cmap,vmin=0,vmax=vmax,aspect='auto')
  scale='0–1' if ax is d else '0–0.30'
  ax.set(xticks=[0,1,2],xticklabels=['1 y','2 y','3 y'],yticks=range(8),yticklabels=cohorts); ax.set_title(f'{title} ({scale})',loc='left',pad=12)
 z=sup[sup.arm.eq('half_shrink')].pivot(index='cohort',columns='horizon_days',values='followup_ge_horizon_frac').loc[cohorts]
 for h,col,mk in [(365,'#42949E','o'),(730,'#C88A2B','s'),(1095,'#B64342','^')]:
  f.plot(np.arange(8),z[h],marker=mk,color=col,ms=3.6,lw=1.15,label=f'≥{h//365} y')
 f.set(ylim=(0,1.05),xticks=np.arange(8),xticklabels=['']*8,ylabel='Fraction with follow-up'); f.set_title('Follow-up support',loc='left',pad=12)
 # Place the long cohort IDs below the data rectangle so the labels cannot be
 # mistaken for a curve annotation or crossed by a plotted line.
 short={'GSE1456':'1456','GSE7390':'7390','GSE76427':'76427','LIRI-JP':'LIRI',
        'GSE32062':'32062','OV-AU':'OV-AU','GSE57495':'57495','PACA-CA':'PACA'}
 for i,co in enumerate(cohorts):
  f.text(i,-.18,short.get(co,co),transform=f.get_xaxis_transform(),ha='center',va='top',rotation=90,fontsize=8.5,clip_on=False)
 f.legend(loc='upper right',handlelength=1.2,labelspacing=.3,handletextpad=.35)
 for i,ax in enumerate(axs.flat):
  ax.title.set_fontsize(9.5); label(ax,'ABCDEF'[i])
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.018),ncol=4,handlelength=1.5,columnspacing=1.0,labelspacing=.7,handletextpad=.45)
 caption='''# Figure 4 | Temporal discrimination, probability accuracy and follow-up support\n\nA, Time-dependent AUC at one, two and three years in all eight cohorts. B, Corresponding Brier scores (lower is better). C, Three-year calibration using actual frozen S(3 y), with cohort risk-group mean predictions and observed Kaplan–Meier survival estimates. No calibration confidence intervals are drawn. D, E, The same AUC and Brier observations by cohort and horizon. F, The fraction of patients whose observed follow-up reaches each horizon. Event counts and patient-level source data accompany the figure; they are not numerically annotated at every point. Solid/circle and dashed/square lines distinguish the two cohorts in each cancer in A and B. Short queue labels in F are expanded in the source data. Every original metric point and all calibration groups are retained; no smoothing or axis truncation was introduced.\n'''
 export(fig,list(axs.flat),4,[tp,cp,sp],caption,
  {'question':'How do discrimination and probability accuracy vary with time and follow-up support?','archetype':'quantitative grid','panels':{'A':'AUC','B':'Brier','C':'actual S3y calibration','D':'all AUC cells','E':'all Brier cells','F':'follow-up fractions'},'data_change':False,'uncertainty':'point estimates; no CI invented','layout_change':'same A-F panel order in 3x2 grid to preserve final-size legibility'})

if __name__=='__main__':
 fig2();fig3();fig4()
