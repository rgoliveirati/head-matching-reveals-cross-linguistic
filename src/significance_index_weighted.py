# save as: src/significance_index_weighted.py
import argparse, numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import spearmanr

def wmean(v, w):
    v=np.asarray(v,float); w=np.asarray(w,float)
    m=np.isfinite(v)&np.isfinite(w)&(w>0)
    return float(np.sum(v[m]*w[m])/np.sum(w[m])) if m.any() else np.nan

def vec(df, lang, dep):
    s=df[(df.lang==lang)&(df.deprel==dep)].sort_values(["layer","head"])
    return s["mean_attention"].to_numpy(float), s["n_arcs"].to_numpy(float)

def micro_weighted(df, dep, langs):
    vals, wts = [], []
    for a,b in combinations(langs,2):
        va,na = vec(df,a,dep); vb,nb = vec(df,b,dep)
        if len(va)!=len(vb) or len(va)==0: continue
        r = spearmanr(va,vb).correlation
        if np.isfinite(r):
            vals.append(r); wts.append(min(na.sum(),nb.sum()))
    return wmean(vals,wts)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--direction", default="head_to_dep")
    ap.add_argument("--model_family", default="mono")
    ap.add_argument("--langs", default="de,es,fr,gl,it,pt,ro")
    ap.add_argument("--rels", default="nsubj,obj,case,amod")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--output", default="results/significance_weighted.csv")
    a=ap.parse_args()
    rng=np.random.default_rng(a.seed)
    df=pd.read_csv(a.in_csv)
    df=df[(df.split==a.split)&(df.direction==a.direction)&(df.model_family==a.model_family)].copy()
    langs=a.langs.split(","); rows=[]
    for dep in a.rels.split(","):
        obs=micro_weighted(df,dep,langs)
        # permutation null: shuffle mean_attention within each lang/dep block
        perm=[]
        sub=df[df.deprel==dep].copy()
        for _ in range(a.n_perm):
            p=sub.copy()
            p["mean_attention"]=p.groupby("lang")["mean_attention"].transform(lambda x: rng.permutation(x.values))
            perm.append(micro_weighted(p,dep,langs))
        perm=np.array([x for x in perm if np.isfinite(x)])
        z=(obs-perm.mean())/(perm.std(ddof=1)+1e-12)
        p_val=(np.sum(perm>=obs)+1)/(len(perm)+1)
        # bootstrap CI over language pairs (reuse pair correlations)
        pairvals,pairw=[],[]
        for x,y in combinations(langs,2):
            va,na=vec(df,x,dep); vb,nb=vec(df,y,dep)
            if len(va)==len(vb) and len(va)>0:
                r=spearmanr(va,vb).correlation
                if np.isfinite(r): pairvals.append(r); pairw.append(min(na.sum(),nb.sum()))
        pairvals=np.array(pairvals); pairw=np.array(pairw); n=len(pairvals)
        boot=[wmean(pairvals[idx],pairw[idx]) for idx in (rng.integers(0,n,n) for _ in range(a.n_boot))]
        lo,hi=np.percentile(boot,[2.5,97.5])
        rows.append({"deprel":dep,"rho_micro":round(obs,3),"ci_low":round(lo,3),
                     "ci_high":round(hi,3),"z":round(z,2),"p":p_val})
    out=pd.DataFrame(rows); out.to_csv(a.output,index=False)
    print(out.to_string(index=False))

if __name__=="__main__": main()
