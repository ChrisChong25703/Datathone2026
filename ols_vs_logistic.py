"""hiring_practice.csv에서 OLS(LPM)와 Logistic을 나란히 비교."""
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("hiring_practice.csv")
d = df.dropna()
print(f"전체 {len(df)}행, 결측 제거 후 {len(d)}행\n")

controls = "C(department) + C(degree) + test_score + years_experience"
specs = {
    "m1 성별만":        "hired ~ C(gender)",
    "m2 +통제":         f"hired ~ C(gender) + {controls}",
    "m3 +성별×부서":    f"hired ~ C(gender)*C(department) + {controls.replace('C(department) + ', '')}",
}

for name, f in specs.items():
    ols = smf.ols(f, d).fit(cov_type="HC1")
    lg = smf.logit(f, d).fit(disp=0)
    me = lg.get_margeff().summary_frame()
    print(f"=== {name}:  {f}")
    rows = []
    for term in ols.params.index[1:]:
        rows.append({
            "term": term,
            "OLS coef": ols.params[term], "OLS p": ols.pvalues[term],
            "Logit coef": lg.params[term], "odds ratio": np.exp(lg.params[term]), "Logit p": lg.pvalues[term],
            "Logit AME": me["dy/dx"].get(term, np.nan),
        })
    print(pd.DataFrame(rows).set_index("term").round(3).to_string())
    yhat = ols.fittedvalues
    print(f"OLS 예측값 범위 [{yhat.min():.2f}, {yhat.max():.2f}], [0,1] 밖 {((yhat<0)|(yhat>1)).sum()}개\n")

# 부서별 성별 효과 (m3의 핵심) — 확률 단위로 비교
lg3 = smf.logit(specs["m3 +성별×부서"], d).fit(disp=0)
ols3 = smf.ols(specs["m3 +성별×부서"], d).fit(cov_type="HC1")
print("=== 부서별 '남성이면 합격확률 몇 %p 높은가' (나머지 조건은 각자 실제 값 유지)")
for dept in ["Engineering", "Marketing"]:
    sub = d[d.department == dept]
    pm = lg3.predict(sub.assign(gender="M")).mean()
    pf = lg3.predict(sub.assign(gender="F")).mean()
    om = ols3.predict(sub.assign(gender="M")).mean() - ols3.predict(sub.assign(gender="F")).mean()
    print(f"{dept:12s}  OLS {om:+.3f}   Logistic {pm-pf:+.3f}")
