import os
from pathlib import Path
from datetime import datetime
import hashlib, os, warnings

os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['NUMEXPR_NUM_THREADS']='1'

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, theilslopes, linregress
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(os.environ.get("NISAR_PROJECT_ROOT", r"D:\NISAR_Mangrove_Paper"))
INPUT_TABLE = PROJECT_ROOT/'05_ML_DL'/'Spatial_CV'/'GEDI_NISAR_modeling_table_with_spatial_folds.csv'
OUT_DIR = PROJECT_ROOT/'06_Results'/'Saturation_Analysis'
TABLE_DIR = OUT_DIR/'Tables'; FIG_DIR = OUT_DIR/'Figures'
LOG_PATH = PROJECT_ROOT/'00_Documentation'/'analysis_log.txt'
PREDICTORS = ['HH_dB','HV_dB','HH_minus_HV','HV_SD']
BIN_EDGES=[0,25,50,75,100,125,150,np.inf]
BIN_LABELS=['0-<25','25-<50','50-<75','75-<100','100-<125','125-<150','>=150']
RANGE_DEFS=[('0-<50',0,50),('50-<100',50,100),('>=100',100,np.inf)]
THRESHOLDS=[25,50,75,100,125,150,200]
FOCUS_MODELS=['Linear_baseline_3P','ET_3P_SENSITIVITY','MLP_3P_SENSITIVITY']
TABLE_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True)

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def append_log(lines):
    stamp=datetime.now().astimezone().isoformat(timespec='seconds')
    with LOG_PATH.open('a',encoding='utf-8') as f:
        f.write(f'\n[{stamp}] PHASE 8A — AGBD SATURATION / BIOMASS-RANGE DIAGNOSTICS\n')
        for line in lines: f.write(line+'\n')
        f.write('-'*72+'\n')

def add_agbd_bin(df,col='agbd'):
    out=df.copy(); out['agbd_bin']=pd.cut(out[col],bins=BIN_EDGES,labels=BIN_LABELS,right=False,include_lowest=True,ordered=True); return out

def calc_basic_errors(y,p):
    return {'RMSE_Mg_ha':float(np.sqrt(mean_squared_error(y,p))), 'MAE_Mg_ha':float(mean_absolute_error(y,p)), 'Bias_pred_minus_obs_Mg_ha':float(np.mean(p-y)), 'MedianAE_Mg_ha':float(np.median(np.abs(p-y)))}

def load_oof_predictions():
    parts=[]
    f=PROJECT_ROOT/'06_Results'/'Baseline'/'Tables'/'baseline_OOF_predictions.csv'
    if f.exists():
        d=pd.read_csv(f)
        if {'observed_agbd','predicted_agbd_baseline'}.issubset(d.columns):
            keep=[c for c in ['shot_number','outer_fold','observed_agbd','predicted_agbd_baseline'] if c in d.columns]
            x=d[keep].copy(); x['model']='Linear_baseline_3P'; x=x.rename(columns={'predicted_agbd_baseline':'predicted_agbd'}); parts.append(x)
    sources=[
        PROJECT_ROOT/'06_Results'/'Random_Forest'/'Tables'/'RF_OOF_predictions.csv',
        PROJECT_ROOT/'06_Results'/'XGBoost'/'Tables'/'XGB_OOF_predictions.csv',
        PROJECT_ROOT/'06_Results'/'SVR'/'Tables'/'SVR_OOF_predictions.csv',
        PROJECT_ROOT/'06_Results'/'Extra_Trees'/'Tables'/'ET_OOF_predictions.csv',
        PROJECT_ROOT/'06_Results'/'DNN_MLP'/'Tables'/'MLP_OOF_predictions.csv',
        PROJECT_ROOT/'06_Results'/'GAM'/'Tables'/'GAM_OOF_predictions.csv']
    for f in sources:
        if not f.exists(): continue
        d=pd.read_csv(f)
        if not {'model','observed_agbd','predicted_agbd'}.issubset(d.columns): continue
        keep=[c for c in ['shot_number','outer_fold','model','observed_agbd','predicted_agbd'] if c in d.columns]
        parts.append(d[keep].copy())
    if not parts: raise RuntimeError('No out-of-fold prediction tables were found.')
    out=pd.concat(parts,ignore_index=True,sort=False)
    keys=['model']+(['shot_number'] if 'shot_number' in out.columns else ['outer_fold','observed_agbd','predicted_agbd'])
    dup=int(out.duplicated(subset=keys).sum())
    if dup: raise RuntimeError(f'Duplicate OOF prediction records detected: {dup}')
    return out

def predictor_bin_summary(df):
    rows=[]; b=add_agbd_bin(df)
    for pred in PREDICTORS:
        for label in BIN_LABELS:
            g=b[b['agbd_bin'].astype(str)==label]
            if len(g)==0: continue
            rows.append({'predictor':pred,'agbd_bin':label,'n':len(g),'agbd_mean':float(g.agbd.mean()),'agbd_median':float(g.agbd.median()),'predictor_mean':float(g[pred].mean()),'predictor_median':float(g[pred].median()),'predictor_q25':float(g[pred].quantile(.25)),'predictor_q75':float(g[pred].quantile(.75)),'predictor_sd':float(g[pred].std(ddof=1))})
    return pd.DataFrame(rows)

def sensitivity_by_biomass_range(df):
    rows=[]
    for pred in PREDICTORS:
        for name,lo,hi in RANGE_DEFS:
            g=df[(df.agbd>=lo)&(df.agbd<hi)] if np.isfinite(hi) else df[df.agbd>=lo]
            if len(g)<10: continue
            x=g.agbd.to_numpy(float); y=g[pred].to_numpy(float); ts=theilslopes(y,x,alpha=.95); sr=spearmanr(x,y)
            rows.append({'predictor':pred,'biomass_range':name,'n':len(g),'theil_sen_slope_per_10Mg_ha':float(ts.slope*10),'slope_CI_low_per_10Mg_ha':float(ts.low_slope*10),'slope_CI_high_per_10Mg_ha':float(ts.high_slope*10),'spearman_rho':float(sr.statistic)})
    out=pd.DataFrame(rows); rr=[]
    for pred in PREDICTORS:
        sub=out[out.predictor==pred]; low=sub[sub.biomass_range=='0-<50']; high=sub[sub.biomass_range=='>=100']
        if len(low)==1 and len(high)==1:
            ls=abs(float(low.iloc[0].theil_sen_slope_per_10Mg_ha)); hs=abs(float(high.iloc[0].theil_sen_slope_per_10Mg_ha))
            rr.append({'predictor':pred,'abs_low_range_slope_per_10Mg_ha':ls,'abs_high_range_slope_per_10Mg_ha':hs,'high_to_low_abs_slope_ratio':hs/ls if ls>0 else np.nan})
    return out,pd.DataFrame(rr)

def cumulative_correlations(df):
    rows=[]
    for t in THRESHOLDS:
        g=df[df.agbd<=t]
        for pred in PREDICTORS:
            if len(g)<10: continue
            sp=spearmanr(g.agbd,g[pred]); pe=pearsonr(g.agbd,g[pred])
            rows.append({'agbd_ceiling_Mg_ha':t,'predictor':pred,'n':len(g),'spearman_rho':float(sp.statistic),'pearson_r':float(pe.statistic)})
    return pd.DataFrame(rows)

def model_performance_by_bin(oof):
    o=oof.copy(); o['agbd_bin']=pd.cut(o.observed_agbd,bins=BIN_EDGES,labels=BIN_LABELS,right=False,include_lowest=True,ordered=True); rows=[]
    for model,gm in o.groupby('model'):
        for label in BIN_LABELS:
            g=gm[gm.agbd_bin.astype(str)==label]
            if len(g)==0: continue
            e=calc_basic_errors(g.observed_agbd.to_numpy(float),g.predicted_agbd.to_numpy(float))
            rows.append({'model':model,'agbd_bin':label,'n':len(g),'observed_mean_Mg_ha':float(g.observed_agbd.mean()),'observed_median_Mg_ha':float(g.observed_agbd.median()),'predicted_mean_Mg_ha':float(g.predicted_agbd.mean()),'predicted_median_Mg_ha':float(g.predicted_agbd.median()),**e})
    return pd.DataFrame(rows)

def model_performance_by_regime(oof):
    regimes=[('<50',0,50),('50-<100',50,100),('>=100',100,np.inf)]; rows=[]
    for model,gm in oof.groupby('model'):
        for name,lo,hi in regimes:
            g=gm[(gm.observed_agbd>=lo)&(gm.observed_agbd<hi)] if np.isfinite(hi) else gm[gm.observed_agbd>=lo]
            if len(g)==0: continue
            e=calc_basic_errors(g.observed_agbd.to_numpy(float),g.predicted_agbd.to_numpy(float))
            rows.append({'model':model,'biomass_regime':name,'n':len(g),'observed_mean_Mg_ha':float(g.observed_agbd.mean()),'predicted_mean_Mg_ha':float(g.predicted_agbd.mean()),**e})
    return pd.DataFrame(rows)

def prediction_compression_summary(oof):
    rows=[]
    for model,g in oof.groupby('model'):
        y=g.observed_agbd.to_numpy(float); p=g.predicted_agbd.to_numpy(float); lr=linregress(y,p)
        rows.append({'model':model,'n':len(g),'observed_mean':float(y.mean()),'predicted_mean':float(p.mean()),'observed_sd':float(y.std(ddof=1)),'predicted_sd':float(p.std(ddof=1)),'predicted_to_observed_SD_ratio':float(p.std(ddof=1)/y.std(ddof=1)),'observed_range':float(y.max()-y.min()),'predicted_range':float(p.max()-p.min()),'predicted_to_observed_range_ratio':float((p.max()-p.min())/(y.max()-y.min())),'OOF_prediction_vs_observed_slope':float(lr.slope),'OOF_prediction_vs_observed_intercept':float(lr.intercept),'Pearson_r':float(lr.rvalue)})
    return pd.DataFrame(rows).sort_values('predicted_to_observed_SD_ratio',ascending=False)

def plot_predictor_binned(summary,pred):
    g=summary[summary.predictor==pred].copy(); g['agbd_bin']=pd.Categorical(g.agbd_bin,categories=BIN_LABELS,ordered=True); g=g.sort_values('agbd_bin')
    x=g.agbd_median.to_numpy(float); y=g.predictor_median.to_numpy(float); lo=y-g.predictor_q25.to_numpy(float); hi=g.predictor_q75.to_numpy(float)-y
    fig,ax=plt.subplots(figsize=(7.2,5.4)); ax.errorbar(x,y,yerr=np.vstack([lo,hi]),marker='o',capsize=3,linewidth=1.4); ax.set_xlabel('Median GEDI AGBD within bin (Mg/ha)'); ax.set_ylabel(pred); ax.set_title(f'{pred} response across AGBD bins',pad=12); fig.tight_layout(); fig.savefig(FIG_DIR/f'{pred}_vs_AGBD_binned.png',dpi=450,bbox_inches='tight'); fig.savefig(FIG_DIR/f'{pred}_vs_AGBD_binned.pdf',bbox_inches='tight'); plt.close(fig)

def plot_focus_metric(perf,col,ylabel,filename,title):
    g=perf[perf.model.isin(FOCUS_MODELS)].copy(); g['agbd_bin']=pd.Categorical(g.agbd_bin,categories=BIN_LABELS,ordered=True); pivot=g.pivot(index='agbd_bin',columns='model',values=col).reindex(BIN_LABELS)
    fig,ax=plt.subplots(figsize=(9.2,5.5)); x=np.arange(len(pivot.index)); cols=[c for c in FOCUS_MODELS if c in pivot.columns]; width=.24
    for i,c in enumerate(cols): ax.bar(x+(i-(len(cols)-1)/2)*width,pivot[c].values,width=width,label=c)
    ax.set_xticks(x); ax.set_xticklabels(BIN_LABELS); ax.set_xlabel('Observed AGBD bin (Mg/ha)'); ax.set_ylabel(ylabel); ax.set_title(title,pad=12); ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.0),frameon=False); fig.tight_layout(); fig.savefig(FIG_DIR/f'{filename}.png',dpi=450,bbox_inches='tight'); fig.savefig(FIG_DIR/f'{filename}.pdf',bbox_inches='tight'); plt.close(fig)

def plot_focus_mean_prediction(perf):
    g=perf[perf.model.isin(FOCUS_MODELS)].copy(); g['agbd_bin']=pd.Categorical(g.agbd_bin,categories=BIN_LABELS,ordered=True); g=g.sort_values(['model','agbd_bin']); fig,ax=plt.subplots(figsize=(7.6,6.0)); xx=[]; yy=[]
    for model in FOCUS_MODELS:
        s=g[g.model==model]
        if len(s)==0: continue
        x=s.observed_mean_Mg_ha.to_numpy(float); y=s.predicted_mean_Mg_ha.to_numpy(float); xx.extend(x); yy.extend(y); ax.plot(x,y,marker='o',linewidth=1.4,label=model)
    hi=max(max(xx),max(yy)) if xx and yy else 200.; ax.plot([0,hi],[0,hi],linestyle='--',linewidth=1.1,label='1:1'); ax.set_xlabel('Mean observed AGBD in bin (Mg/ha)'); ax.set_ylabel('Mean predicted AGBD in bin (Mg/ha)'); ax.set_title('Prediction compression across biomass ranges',pad=12); ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.0),frameon=False); fig.tight_layout(); fig.savefig(FIG_DIR/'focus_models_mean_prediction_by_AGBD_bin.png',dpi=450,bbox_inches='tight'); fig.savefig(FIG_DIR/'focus_models_mean_prediction_by_AGBD_bin.pdf',bbox_inches='tight'); plt.close(fig)

def plot_cumulative(corr):
    fig,ax=plt.subplots(figsize=(8.0,5.5))
    for pred in PREDICTORS:
        g=corr[corr.predictor==pred].sort_values('agbd_ceiling_Mg_ha')
        if len(g): ax.plot(g.agbd_ceiling_Mg_ha,g.spearman_rho,marker='o',linewidth=1.4,label=pred)
    ax.axhline(0,linestyle='--',linewidth=1.0); ax.set_xlabel('Maximum AGBD included (Mg/ha)'); ax.set_ylabel('Spearman correlation with AGBD'); ax.set_title('Cumulative radar–AGBD association by biomass ceiling',pad=12); ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.0),frameon=False); fig.tight_layout(); fig.savefig(FIG_DIR/'cumulative_spearman_by_AGBD_threshold.png',dpi=450,bbox_inches='tight'); fig.savefig(FIG_DIR/'cumulative_spearman_by_AGBD_threshold.pdf',bbox_inches='tight'); plt.close(fig)

def plot_compression(comp):
    g=comp.sort_values('predicted_to_observed_SD_ratio',ascending=False); fig,ax=plt.subplots(figsize=(10.2,5.8)); ax.bar(g.model,g.predicted_to_observed_SD_ratio); ax.axhline(1,linestyle='--',linewidth=1.0); ax.set_ylabel('Predicted SD / observed SD'); ax.set_title('Out-of-fold prediction dynamic-range compression',pad=12); ax.tick_params(axis='x',rotation=30); fig.tight_layout(); fig.savefig(FIG_DIR/'prediction_SD_compression_all_models.png',dpi=450,bbox_inches='tight'); fig.savefig(FIG_DIR/'prediction_SD_compression_all_models.pdf',bbox_inches='tight'); plt.close(fig)

def main():
    print('='*72); print('PHASE 8A — AGBD SATURATION / BIOMASS-RANGE DIAGNOSTICS'); print('='*72); print('Diagnostic only: no model refitting, no retuning, no CV changes.')
    if not INPUT_TABLE.exists(): raise FileNotFoundError(f'Missing frozen modelling table:\n{INPUT_TABLE}')
    df=pd.read_csv(INPUT_TABLE); missing=sorted(({'agbd'}|set(PREDICTORS))-set(df.columns))
    if missing: raise RuntimeError(f'Missing required columns: {missing}')
    if not np.isfinite(df[['agbd']+PREDICTORS].to_numpy(float)).all(): raise RuntimeError('Non-finite AGBD/predictor values found.')
    print(f'Frozen modelling records: {len(df):,}'); print(f'Input SHA256: {sha256_file(INPUT_TABLE)}'); print(f'AGBD range: {df.agbd.min():.2f} to {df.agbd.max():.2f} Mg/ha')
    pred_bins=predictor_bin_summary(df); pred_bins.to_csv(TABLE_DIR/'predictor_summary_by_AGBD_bin.csv',index=False)
    sens,ratios=sensitivity_by_biomass_range(df); sens.to_csv(TABLE_DIR/'radar_sensitivity_by_biomass_range.csv',index=False); ratios.to_csv(TABLE_DIR/'high_vs_low_biomass_slope_ratios.csv',index=False)
    corr=cumulative_correlations(df); corr.to_csv(TABLE_DIR/'cumulative_predictor_correlations_by_AGBD_threshold.csv',index=False)
    oof=load_oof_predictions(); print(f'OOF prediction rows across model variants: {len(oof):,}'); print(f'Model variants found: {oof.model.nunique()}')
    perf=model_performance_by_bin(oof); perf.to_csv(TABLE_DIR/'OOF_model_performance_by_AGBD_bin.csv',index=False)
    regimes=model_performance_by_regime(oof); regimes.to_csv(TABLE_DIR/'OOF_model_performance_by_biomass_regime.csv',index=False)
    comp=prediction_compression_summary(oof); comp.to_csv(TABLE_DIR/'OOF_prediction_compression_summary.csv',index=False)
    for pred in PREDICTORS: plot_predictor_binned(pred_bins,pred)
    plot_focus_metric(perf,'RMSE_Mg_ha','RMSE (Mg/ha)','focus_models_RMSE_by_AGBD_bin','Spatial-CV RMSE across observed biomass ranges')
    plot_focus_metric(perf,'Bias_pred_minus_obs_Mg_ha','Bias: predicted − observed (Mg/ha)','focus_models_bias_by_AGBD_bin','Spatial-CV bias across observed biomass ranges')
    plot_focus_mean_prediction(perf); plot_cumulative(corr); plot_compression(comp)
    counts=add_agbd_bin(df).agbd_bin.value_counts(sort=False)
    report=['PHASE 8A — AGBD SATURATION / BIOMASS-RANGE DIAGNOSTICS','='*72,f'Frozen modelling table: {INPUT_TABLE}',f'Input SHA256: {sha256_file(INPUT_TABLE)}',f'Records: {len(df):,}',f'AGBD range: {df.agbd.min():.3f} to {df.agbd.max():.3f} Mg/ha','','Purpose:','Assess whether radar predictor sensitivity and spatial-CV model performance weaken as observed GEDI AGBD increases. This is diagnostic, not model tuning.','','Fixed AGBD-bin counts:',counts.to_string(),'','Robust radar sensitivity by broad biomass range:',sens.to_string(index=False),'','High-to-low absolute Theil-Sen slope ratios:',ratios.to_string(index=False),'','Prediction compression summary:',comp.to_string(index=False),'','Interpretation rules:','- Do NOT infer a saturation threshold from one plot alone.','- Evidence is stronger when multiple diagnostics agree: flattening predictor response, weaker high-biomass slope/correlation, increasing error/negative bias, and strong prediction-range compression.','- Theil-Sen slopes and correlations are descriptive because observations are spatially clustered.','- Model error diagnostics use only out-of-fold predictions from the frozen spatial CV.','- No model was refit, tuned, or selected in Phase 8A.','','Next decision after review:','If high-biomass sensitivity clearly weakens, quantify/report saturation behavior and proceed to Phase 8B temporal-mismatch sensitivity. If not, temporal mismatch / missing predictor information becomes the more likely limitation.']
    (OUT_DIR/'SATURATION_DIAGNOSTIC_REPORT.txt').write_text('\n'.join(report),encoding='utf-8')
    append_log(['Script: phase8a_saturation_analysis.py',f'Input modelling table SHA256: {sha256_file(INPUT_TABLE)}',f'Records: {len(df)}','Fixed bins: 0-25, 25-50, 50-75, 75-100, 100-125, 125-150, >=150 Mg/ha','Broad sensitivity ranges: <50, 50-100, >=100 Mg/ha',f'OOF model variants included: {oof.model.nunique()}','Phase 8A diagnostic only; no model tuning/refitting.'])
    print('\n'+'='*72); print('PHASE 8A COMPLETE — SATURATION REVIEW REQUIRED'); print('='*72); print('\nAGBD-bin counts:'); print(counts.to_string()); print('\nHigh-vs-low absolute radar slope ratios:'); print(ratios.to_string(index=False)); print(f'\nOutputs: {OUT_DIR}'); print('\nIMPORTANT: Upload the report, key tables, and figures. Do not assign a final saturation threshold yet.'); None  # interactive pause removed for portable execution

if __name__=='__main__': main()
