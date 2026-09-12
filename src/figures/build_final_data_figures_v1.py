from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'cohort_trial28'/'unified_multihorizon_benchmark_v1'
SPEED=ROOT/'cohort_trial28'/'benchmark_classic_survival_v5_speed_aggregate'
RISK=ROOT/'cohort_trial28'/'unified_base_audit_v1'/'risk_stratification_v1'
TCAL=ROOT/'cohort_trial28'/'paper_time_calibration_data_v1'
OUT=ROOT/'figures'/'paper_main_v1'/'final_data_figures_v1'
SRC=OUT/'source_data'; SRC.mkdir(parents=True,exist_ok=True)

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':9,'axes.titlesize':11,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
    'legend.fontsize':8,'axes.linewidth':0.9,'axes.spines.top':False,'axes.spines.right':False,
    'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
    'figure.facecolor':'white','savefig.facecolor':'white'})
PAL={'GeneNarrator — half-shrink':'#145A86','GeneNarrator — source reference':'#37A0A4','DeepSurv':'#BF5B5B','RSF (speed)':'#6F7780','GBSA (speed)':'#A7ADB4'}
METHODS=list(PAL)
COHORTS=[('BRCA','GSE1456'),('BRCA','GSE7390'),('LIHC','GSE76427'),('LIHC','LIRI-JP'),('OV','GSE32062'),('OV','OV-AU'),('PAAD','GSE57495'),('PAAD','PACA-CA')]
# Load and validate benchmark inputs
m=pd.read_csv(BENCH/'metrics_long.csv')
s=pd.read_csv(SPEED/'metrics.csv')
assert set(zip(m.cancer,m.cohort))==set(COHORTS)
assert set(m.method.unique()) >= {'GeneNarrator','GeneNarrator+Adapt','DeepSurv'}
assert set(zip(s.cancer,s.cohort))==set(COHORTS)
assert set(s.method.unique())=={'RSF','GBSA'}
# Build canonical long table. Speed baseline arm is explicit and not conflated with full benchmark budget.
rows=[]
for _,r in m.iterrows():
    if r.method not in {'GeneNarrator','GeneNarrator+Adapt','DeepSurv'}: continue
    rows.append({'cancer':r.cancer,'cohort':r.cohort,'method':{'GeneNarrator':'GeneNarrator — source reference','GeneNarrator+Adapt':'GeneNarrator — half-shrink','DeepSurv':'DeepSurv'}[r.method], 'n':r.n,'events':r.events, 'uno_c_1y':r.uno_c_1y,'uno_c_2y':r.uno_c_2y,'uno_c_3y':r.uno_c_3y,'harrell_c':r.harrell_c,'ibs':r.ibs,'config':'unified_multihorizon_benchmark_v1'})
for _,r in s.iterrows():
    if r.method not in {'RSF','GBSA'}: continue
    # use source_z only; target_z is the paired alternate preprocessing arm and is not a separate model
    if r.arm!='source_z': continue
    rows.append({'cancer':r.cancer,'cohort':r.cohort,'method':f'{r.method} (speed)','n':r.n,'events':r.events, 'uno_c_1y':r.uno_c_1y,'uno_c_2y':r.uno_c_2y,'uno_c_3y':r.uno_c_3y,'harrell_c':r.harrell_c,'ibs':r.ibs,'config':'benchmark_classic_survival_v5_speed_aggregate; source_z; reduced trees/iterations'})
canon=pd.DataFrame(rows)
assert canon.groupby(['cancer','cohort']).method.nunique().eq(5).all()
canon.to_csv(SRC/'figure_A_performance.csv',index=False,float_format='%.8f')
# Figure A: two horizon dot/line panels + macro summary
fig=plt.figure(figsize=(13.0,8.1),constrained_layout=False)
gs=fig.add_gridspec(3,2,height_ratios=[1.16,1.16,0.9],hspace=0.48,wspace=0.25)
for pi,h in enumerate(['uno_c_2y','uno_c_3y']):
    ax=fig.add_subplot(gs[pi,:]); x=np.arange(len(COHORTS));
    for meth in METHODS:
        sub=canon[canon.method.eq(meth)].set_index(['cancer','cohort']).reindex(COHORTS)
        ax.plot(x,sub[h].to_numpy(float),'-o',lw=1.9,ms=5,color=PAL[meth],mec='white',mew=.65,label=meth,zorder=3)
    ax.set_xticks(x,[f'{c}\n{co}' for c,co in COHORTS]); ax.tick_params(axis='x',pad=3)
    ax.set_ylabel(f'{h[6:]}-year Uno C-index'); ax.set_ylim(0.40,0.86); ax.yaxis.set_major_locator(MultipleLocator(.1)); ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f')); ax.grid(axis='y',color='#E8EDF0',lw=.7); ax.axhline(.5,color='#C9CFD4',lw=.8,zorder=0)
    ax.set_title(f"{'a' if pi==0 else 'b'}  External-cohort discrimination at {h[6:]} years",loc='left',fontweight='bold',pad=7)
    # cancer separators
    for sep in [1.5,3.5,5.5]: ax.axvline(sep,color='#D9DEE2',lw=.6,zorder=0)
# macro horizon panel
ax=fig.add_subplot(gs[2,:]);
for meth in METHODS:
    sub=canon[canon.method.eq(meth)]
    y=[sub[f'uno_c_{k}y'].mean() for k in [1,2,3]]
    ax.plot([1,2,3],y,'-o',lw=2,ms=5,color=PAL[meth],mec='white',mew=.6,label=meth)
    for xx,yy in zip([1,2,3],y): ax.text(xx,yy+(.008 if meth in ['GeneNarrator — half-shrink','DeepSurv'] else -.012),f'{yy:.3f}',ha='center',va='center',fontsize=6.8,color=PAL[meth])
ax.set_xticks([1,2,3],['1 year','2 years','3 years']); ax.set_ylabel('Macro-average Uno C-index'); ax.set_ylim(.50,.69); ax.yaxis.set_major_locator(MultipleLocator(.05)); ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')); ax.grid(axis='y',color='#E8EDF0',lw=.7); ax.set_title('c  Macro-average trajectory across eight cohorts',loc='left',fontweight='bold',pad=7)
handles=[Line2D([0],[0],color=PAL[k],marker='o',lw=2,ms=5,label=k) for k in METHODS]
ax.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.5,-.28),frameon=False)
fig.suptitle('GeneNarrator external survival benchmark',fontsize=14,fontweight='bold',y=.995)
fig.text(.01,.005,'All points are cohort-level estimates. RSF/GBSA are explicitly labelled speed baselines (source_z; reduced tree/iteration budget) and are shown for contextual comparison.',fontsize=7.3,color='#555B60')
fig.subplots_adjust(left=.06,right=.99,top=.95,bottom=.13)
for ext in ['png','pdf','svg']: fig.savefig(OUT/f'figure_A_performance.{ext}',dpi=600 if ext=='png' else None,bbox_inches='tight')
plt.close(fig)
# Aggregate summary source
agg=canon.groupby('method',as_index=False)[['uno_c_1y','uno_c_2y','uno_c_3y','harrell_c','ibs']].mean(); agg.to_csv(SRC/'figure_A_macro_summary.csv',index=False,float_format='%.8f')

# Figure B: KM risk stratification, use frozen half_shrink patient-level predictions
patients=pd.read_csv(RISK/'patients.csv'); curves=pd.read_csv(RISK/'km_curve.csv'); summ=pd.read_csv(RISK/'summary.csv')
assert set(zip(patients.cancer,patients.cohort))==set(COHORTS)
assert patients['sample'].is_unique is False or patients['sample'].notna().all()
assert set(curves.risk_group.unique())=={'Lower predicted risk','Higher predicted risk'}
patients.to_csv(SRC/'figure_B_patients.csv',index=False); curves.to_csv(SRC/'figure_B_km_curve.csv',index=False); summ.to_csv(SRC/'figure_B_risk_summary.csv',index=False)
fig,axs=plt.subplots(2,4,figsize=(13.4,6.9),sharex=True,sharey=True)
for ax,(ca,co) in zip(axs.flat,COHORTS):
    sub=curves[(curves.cancer==ca)&(curves.cohort==co)]
    sm=summ[(summ.cancer==ca)&(summ.cohort==co)]
    for grp,col in [('Lower predicted risk','#37A0A4'),('Higher predicted risk','#145A86')]:
        z=sub[sub.risk_group==grp].sort_values('time_days'); ax.step(z.time_days/365.25,z.survival,where='post',color=col,lw=1.7,label=grp.replace(' predicted risk',''))
    p=sm.logrank_p.iloc[0] if len(sm) else np.nan; ax.text(.98,.08,f'log-rank p={p:.2g}',ha='right',transform=ax.transAxes,fontsize=7,color='#4C555B')
    ax.set_title(f'{ca} · {co}',loc='left',fontweight='bold',fontsize=9,pad=5); ax.set_xlim(0,8); ax.set_ylim(0,1.03); ax.grid(axis='y',color='#EDF0F2',lw=.6)
    ax.set_xlabel('Years'); ax.set_ylabel('Survival' if ax in axs[:,0] else '')
axs[0,0].legend(loc='lower left',frameon=False,fontsize=7)
fig.suptitle('Patient risk stratification by GeneNarrator half-shrink predictions',fontsize=13.5,fontweight='bold',y=.995)
fig.text(.01,.005,'Each cohort is split at its prespecified within-cohort median 3-year RMST risk; curves and p-values are retained for all eight cohorts.',fontsize=7.3,color='#555B60')
fig.subplots_adjust(left=.06,right=.995,top=.93,bottom=.11,wspace=.22,hspace=.36)
for ext in ['png','pdf','svg']: fig.savefig(OUT/f'figure_B_KM_risk_stratification.{ext}',dpi=600 if ext=='png' else None,bbox_inches='tight')
plt.close(fig)

# Figure C: AUC, Brier, and 3-year calibration scatter
metrics=pd.read_csv(TCAL/'time_metrics.csv'); kmc=pd.read_csv(TCAL/'km_calibration.csv')
metrics.to_csv(SRC/'figure_C_time_metrics.csv',index=False); kmc.to_csv(SRC/'figure_C_km_calibration.csv',index=False); pd.read_csv(TCAL/'followup_support.csv').to_csv(SRC/'figure_C_followup_support.csv',index=False)
fig,axs=plt.subplots(1,3,figsize=(13.2,4.2),gridspec_kw={'width_ratios':[1.2,1.2,1]})
# AUC and Brier by horizon, one line per cohort
for ca,co in COHORTS:
    z=metrics[(metrics.cancer==ca)&(metrics.cohort==co)&(metrics.arm=='half_shrink')].iloc[0]
    axs[0].plot([1,2,3],[z.auc_1y,z.auc_2y,z.auc_3y],'-o',lw=1.25,ms=3.5,color={'BRCA':'#145A86','LIHC':'#1B8A7A','OV':'#C27A28','PAAD':'#9B4B75'}[ca],alpha=.82)
    axs[1].plot([1,2,3],[z.brier_1y,z.brier_2y,z.brier_3y],'-o',lw=1.25,ms=3.5,color={'BRCA':'#145A86','LIHC':'#1B8A7A','OV':'#C27A28','PAAD':'#9B4B75'}[ca],alpha=.82)
axs[0].set_title('a  Time-dependent AUC',loc='left',fontweight='bold'); axs[0].set_ylabel('AUC'); axs[0].set_xticks([1,2,3],['1 y','2 y','3 y']); axs[0].set_ylim(.35,.9); axs[0].grid(axis='y',color='#E8EDF0',lw=.7)
axs[1].set_title('b  Brier score',loc='left',fontweight='bold'); axs[1].set_ylabel('Brier (lower is better)'); axs[1].set_xticks([1,2,3],['1 y','2 y','3 y']); axs[1].set_ylim(0,.34); axs[1].grid(axis='y',color='#E8EDF0',lw=.7)
# calibration at 3 years: x=predicted survival = 1 - 3y RMST risk, y=KM survival at horizon 1095
z=kmc[(kmc.arm=='half_shrink')&(kmc.horizon_days==1095)].copy(); z['pred_survival']=1-z.risk_mean_3y_rmst
for ca,co in COHORTS:
    q=z[(z.cancer==ca)&(z.cohort==co)].sort_values('pred_survival'); axs[2].plot(q.pred_survival,q.km_survival,'o-',lw=1.0,ms=3.3,color={'BRCA':'#145A86','LIHC':'#1B8A7A','OV':'#C27A28','PAAD':'#9B4B75'}[ca],alpha=.8)
axs[2].plot([0,1],[0,1],'--',color='#7B858C',lw=1); axs[2].set_xlim(.65,1.01); axs[2].set_ylim(.55,1.03); axs[2].set_xlabel('Predicted survival at 3 y\n(1 − 3-y RMST risk)'); axs[2].set_ylabel('Observed KM survival at 3 y'); axs[2].set_title('c  Three-year calibration by risk group',loc='left',fontweight='bold'); axs[2].grid(color='#E8EDF0',lw=.7)
leg=[Line2D([0],[0],color={'BRCA':'#145A86','LIHC':'#1B8A7A','OV':'#C27A28','PAAD':'#9B4B75'}[ca],marker='o',lw=1.5,label=ca) for ca in ['BRCA','LIHC','OV','PAAD']]
axs[1].legend(handles=leg,loc='upper right',frameon=False,fontsize=7)
fig.suptitle('Temporal discrimination and calibration of GeneNarrator',fontsize=13.5,fontweight='bold',y=1.02)
fig.text(.01,-.025,'Metrics use the frozen half-shrink arm and cohort-specific censoring estimator. Calibration points are the three prespecified within-cohort risk groups; all eight cohorts are retained.',fontsize=7.2,color='#555B60')
fig.subplots_adjust(left=.06,right=.99,top=.87,bottom=.2,wspace=.3)
for ext in ['png','pdf','svg']: fig.savefig(OUT/f'figure_C_time_calibration.{ext}',dpi=600 if ext=='png' else None,bbox_inches='tight')
plt.close(fig)

# Captions, audit manifest
(OUT/'captions.md').write_text('''# Final data figures v1\n\n**Figure A.** External-cohort discrimination for GeneNarrator, source reference and half-shrink, DeepSurv, and contextual RSF/GBSA speed baselines. RSF/GBSA use the source_z arm and reduced tree/iteration budget from `benchmark_classic_survival_v5_speed_aggregate`; they are labelled accordingly and are exploratory context rather than a matched compute-budget comparison.\n\n**Figure B.** Kaplan–Meier risk stratification for all eight external OS cohorts using frozen GeneNarrator half-shrink predictions. Splits use the within-cohort median 3-year RMST risk.\n\n**Figure C.** Time-dependent AUC, Brier score, and three-year risk-group calibration from the frozen half-shrink arm.\n''',encoding='utf-8')
manifest={'created_at_utc':pd.Timestamp.utcnow().isoformat(),'status':'READY_FOR_REVIEW','inputs':{},'outputs':{}}
for p in [BENCH/'metrics_long.csv',SPEED/'metrics.csv',RISK/'patients.csv',RISK/'km_curve.csv',RISK/'summary.csv',TCAL/'time_metrics.csv',TCAL/'km_calibration.csv',TCAL/'followup_support.csv']:
    manifest['inputs'][str(p.relative_to(ROOT))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
for p in list(OUT.glob('figure_*.*'))+[SRC/'figure_A_performance.csv',SRC/'figure_A_macro_summary.csv',SRC/'figure_B_patients.csv',SRC/'figure_B_km_curve.csv',SRC/'figure_B_risk_summary.csv',SRC/'figure_C_time_metrics.csv',SRC/'figure_C_km_calibration.csv',SRC/'figure_C_followup_support.csv']:
    if p.is_file(): manifest['outputs'][str(p.relative_to(ROOT))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'out':str(OUT),'performance_rows':len(canon),'risk_patients':len(patients),'km_rows':len(curves),'time_rows':len(metrics),'calibration_rows':len(kmc),'macro':agg.to_dict('records')},indent=2,default=str))
