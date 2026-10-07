import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
df=pd.read_csv("hiring_practice.csv"); d=df.dropna()
cat=["gender","department","degree"]; num=["test_score","years_experience"]
X=d[cat+num]; y=d["hired"]
print("n",len(d),"base rate",y.mean())
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,stratify=y,random_state=0)
print("train",len(Xtr),ytr.mean(),"test",len(Xte),yte.mean())
pre=ColumnTransformer([("cat",OneHotEncoder(drop="first"),cat),("num",StandardScaler(),num)])
def report(name,model,thr=0.5):
    p=model.predict_proba(Xte)[:,1]; yh=(p>=thr).astype(int)
    cm=confusion_matrix(yte,yh); 
    r=dict(acc=accuracy_score(yte,yh),prec=precision_score(yte,yh,zero_division=0),rec=recall_score(yte,yh),f1=f1_score(yte,yh),auc=roc_auc_score(yte,p))
    print(name,"thr",thr,"cm",cm.ravel().tolist(),{k:round(v,3) for k,v in r.items()}); return p
dum=DummyClassifier(strategy="most_frequent").fit(Xtr,ytr); report("dummy",dum)
lr=Pipeline([("pre",pre),("clf",LogisticRegression(max_iter=1000))]).fit(Xtr,ytr)
p_lr=report("logit",lr); report("logit",lr,0.3); report("logit",lr,0.4)
names=lr.named_steps["pre"].get_feature_names_out(); coef=lr.named_steps["clf"].coef_[0]
print(pd.DataFrame({"feat":names,"coef":coef.round(3),"OR":np.exp(coef).round(2)}))
print("intercept",lr.named_steps["clf"].intercept_)
print("first5 proba", np.round(p_lr[:5],3), Xte.head(5).to_dict("records"), yte.head(5).tolist())
tree=Pipeline([("pre",ColumnTransformer([("cat",OneHotEncoder(drop="first"),cat)],remainder="passthrough")),("clf",DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,random_state=0))]).fit(Xtr,ytr)
p_tr=report("tree",tree)
rf=Pipeline([("pre",ColumnTransformer([("cat",OneHotEncoder(drop="first"),cat)],remainder="passthrough")),("clf",RandomForestClassifier(n_estimators=300,min_samples_leaf=10,random_state=0))]).fit(Xtr,ytr)
p_rf=report("rf",rf)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
for nm,m in [("logit",lr),("tree",tree),("rf",rf)]:
    sc=cross_val_score(m,X,y,cv=cv,scoring="roc_auc"); print("cv",nm,sc.round(3),sc.mean().round(3),sc.std().round(3))
pi=permutation_importance(lr,Xte,yte,scoring="roc_auc",n_repeats=30,random_state=0); print("perm imp", dict(zip(X.columns,pi.importances_mean.round(3))))
# fairness on test
def fair(p,thr=0.5,label=""):
    t=Xte.copy(); t["y"]=yte.values; t["yh"]=(p>=thr).astype(int)
    out=t.groupby("gender").apply(lambda g: pd.Series({"n":len(g),"base":g.y.mean(),"sel":g.yh.mean(),"TPR":g[g.y==1].yh.mean(),"FPR":g[g.y==0].yh.mean(),"prec":g[g.yh==1].y.mean() if (g.yh==1).any() else np.nan}),include_groups=False)
    print(label,"\n",out.round(3))
fair(p_lr,0.5,"logit with gender")
# without gender
cat2=["department","degree"]
pre2=ColumnTransformer([("cat",OneHotEncoder(drop="first"),cat2),("num",StandardScaler(),num)])
lr2=Pipeline([("pre",pre2),("clf",LogisticRegression(max_iter=1000))]).fit(Xtr,ytr)
p2=lr2.predict_proba(Xte)[:,1]; print("no-gender auc",roc_auc_score(yte,p2),"acc",accuracy_score(yte,(p2>=.5)))
fair(p2,0.5,"logit without gender")
print("cv no gender", cross_val_score(lr2,X,y,cv=cv,scoring="roc_auc").mean().round(3))
# counterfactual flip on test
Xf=Xte.copy(); Xf["gender"]=Xf["gender"].map({"M":"F","F":"M"}); pf=lr.predict_proba(Xf)[:,1]
print("flip: mean change for F->M", (pf-p_lr)[Xte.gender.values=="F"].mean(), "M->F", (pf-p_lr)[Xte.gender.values=="M"].mean(), "decisions changed", ((pf>=.5)!=(p_lr>=.5)).sum(), "of", len(Xte))
# figures
fig,ax=plt.subplots(1,2,figsize=(10,4))
for nm,p in [("logistic",p_lr),("tree",p_tr),("random forest",p_rf)]:
    f,t,_=roc_curve(yte,p); ax[0].plot(f,t,label=f"{nm} (AUC {roc_auc_score(yte,p):.2f})")
ax[0].plot([0,1],[0,1],"k:",label="random (AUC 0.5)"); ax[0].set(xlabel="False positive rate",ylabel="True positive rate (recall)",title="ROC curve (test set)"); ax[0].legend(fontsize=8)
ths=np.linspace(0.05,0.9,60)
ax[1].plot(ths,[precision_score(yte,p_lr>=t,zero_division=0) for t in ths],label="precision")
ax[1].plot(ths,[recall_score(yte,p_lr>=t) for t in ths],label="recall")
ax[1].plot(ths,[accuracy_score(yte,p_lr>=t) for t in ths],label="accuracy")
ax[1].axvline(.5,c="gray",ls=":"); ax[1].set(xlabel="threshold",title="logistic: metrics vs threshold"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("stats-book/fig/clf_roc.png",dpi=150)
