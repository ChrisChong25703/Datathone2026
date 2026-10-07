"""numpy·pandas만으로 통계를 직접 구현하는 워크북 노트북 2개 생성.

STEPS: (markdown, 정답 코드, 빈칸 코드 또는 None)
"""
import nbformat as nbf

STEPS = [
    ("# numpy·pandas로 직접 만드는 통계 워크북\n"
     "책 `stats-basics-workbook.pdf`와 같은 순서입니다. `___`를 채우고 실행한 뒤 책의 '실행 결과'와 비교하세요.\n"
     "사용하는 라이브러리는 **numpy, pandas, math** 뿐입니다.", None, None),

    ("## Step 0. 준비", """import math
import numpy as np
import pandas as pd

def Phi(z):
    \"\"\"표준정규 누적분포 P(Z <= z). math.erf로 계산.\"\"\"
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

df = pd.read_csv("hiring_practice.csv")
d = df.dropna()
print(df.shape, d.shape)   # (600, 7) (582, 7)""", None),

    ("## Step 1. 평균, 분산, 표준편차를 직접", """x = d["test_score"].to_numpy()
n = len(x)
mean = x.sum() / n
dev = x - mean                         # 편차 (벡터 연산)
sd_pop = math.sqrt((dev ** 2).sum() / n)        # 편차의 RMS (분모 n)
sd_sample = math.sqrt((dev ** 2).sum() / (n - 1))  # 표본 SD (분모 n-1)
print(round(mean, 3), round(sd_pop, 3), round(sd_sample, 3))
print(round(d["test_score"].std(), 3), round(np.std(x), 3))   # pandas(n-1), numpy(n)""",
     """x = d["test_score"].to_numpy()
n = len(x)
mean = x.___() / n
dev = x - ___
sd_pop = math.sqrt((dev ** 2).sum() / ___)        # 분모 n
sd_sample = math.sqrt((dev ** 2).sum() / ___)     # 분모 n-1
print(round(mean, 3), round(sd_pop, 3), round(sd_sample, 3))
print(round(d["test_score"].std(), 3), round(np.std(x), 3))"""),

    ("## Step 2. 중앙값과 사분위수: 정렬로", """s = np.sort(x)                          # 정렬 = order statistics
median = (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]
q1 = s[math.ceil(0.25 * n) - 1]         # 노트 Prop 5.6: x_(ceil(un))
q3 = s[math.ceil(0.75 * n) - 1]
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = s[(s < lo) | (s > hi)]
print(median, q1, q3, iqr, (lo, hi))
print(outliers)
print(np.partition(x, n // 2)[n // 2])   # 정렬 없이 선택(quickselect)""",
     """s = np.___(x)                           # 정렬
median = (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]
q1 = s[math.ceil(___ * n) - 1]
q3 = s[math.ceil(___ * n) - 1]
iqr = q3 - q1
lo, hi = q1 - ___ * iqr, q3 + ___ * iqr
outliers = s[(s < lo) | (s > hi)]
print(median, q1, q3, iqr, (lo, hi))
print(outliers)
print(np.partition(x, n // 2)[n // 2])"""),

    ("## Step 3. 표준화, 정규근사, 백분위", """z = (x - mean) / sd_sample
for k in [1, 2, 3]:
    actual = (np.abs(z) <= k).mean()
    normal = Phi(k) - Phi(-k)
    print(k, round(actual, 3), round(normal, 3))

# 80점 이상 비율: 실제 vs 정규근사
z80 = (80 - mean) / sd_sample
print(round((x >= 80).mean(), 3), round(1 - Phi(z80), 3))

# 85점은 몇 백분위? 정렬된 배열에서 이진 탐색
pos = np.searchsorted(s, 85, side="right")   # 85 이하인 개수
print(pos, round(pos / n, 3), round(Phi((85 - mean) / sd_sample), 3))""",
     """z = (x - ___) / ___
for k in [1, 2, 3]:
    actual = (np.abs(z) <= k).mean()
    normal = Phi(k) - Phi(___)
    print(k, round(actual, 3), round(normal, 3))

z80 = (80 - mean) / sd_sample
print(round((x >= 80).mean(), 3), round(1 - Phi(___), 3))

pos = np.___(s, 85, side="right")            # 이진 탐색
print(pos, round(pos / n, 3), round(Phi((85 - mean) / sd_sample), 3))"""),

    ("## Step 4. 그룹 비율과 신뢰구간", """g = df.groupby("gender")["hired"].agg(["sum", "count"])
g["p"] = g["sum"] / g["count"]
g["se"] = np.sqrt(g["p"] * (1 - g["p"]) / g["count"])
g["lo"] = g["p"] - 1.96 * g["se"]
g["hi"] = g["p"] + 1.96 * g["se"]
print(g.round(3))

diff = g.loc["M", "p"] - g.loc["F", "p"]
se_diff = math.sqrt(g.loc["M", "se"] ** 2 + g.loc["F", "se"] ** 2)
print(round(diff, 3), round(se_diff, 4), (round(diff - 1.96 * se_diff, 3), round(diff + 1.96 * se_diff, 3)))""",
     """g = df.groupby(___)["hired"].agg(["sum", "count"])
g["p"] = g["sum"] / g[___]
g["se"] = np.sqrt(g["p"] * (1 - g["p"]) / g["count"])
g["lo"] = g["p"] - ___ * g["se"]
g["hi"] = g["p"] + ___ * g["se"]
print(g.round(3))

diff = g.loc["M", "p"] - g.loc["F", "p"]
se_diff = math.sqrt(g.loc["M", "se"] ** 2 + g.loc["F", "se"] ** 2)
print(round(diff, 3), round(se_diff, 4), (round(diff - 1.96 * se_diff, 3), round(diff + 1.96 * se_diff, 3)))"""),

    ("## Step 5. 두 비율의 z-검정을 직접", """def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)                       # H0에서 합친 비율
    se0 = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se0
    pval = 2 * (1 - Phi(abs(z)))                    # 양측
    return z, pval

z, pval = two_prop_z(125, 287, 68, 313)
print(round(z, 3), pval)

for dept, sub in df.groupby("department"):
    k = sub.groupby("gender")["hired"].sum()
    m = sub.groupby("gender")["hired"].count()
    zz, pp = two_prop_z(k["M"], m["M"], k["F"], m["F"])
    print(dept, round(zz, 3), round(pp, 4))""",
     """def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (___)                           # H0에서 합친 비율
    se0 = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / ___
    pval = 2 * (1 - Phi(abs(___)))                  # 양측
    return z, pval

z, pval = two_prop_z(125, 287, 68, 313)
print(round(z, 3), pval)

for dept, sub in df.groupby("department"):
    k = sub.groupby("gender")["hired"].sum()
    m = sub.groupby("gender")["hired"].count()
    zz, pp = two_prop_z(k["M"], m["M"], k["F"], m["F"])
    print(dept, round(zz, 3), round(pp, 4))"""),

    ("## Step 6. 순열 검정(permutation test): 공식 없이 p-value", """rng = np.random.default_rng(0)

def perm_test(sub, reps=10_000):
    y = sub["hired"].to_numpy()
    is_m = (sub["gender"] == "M").to_numpy()
    obs = y[is_m].mean() - y[~is_m].mean()
    count = 0
    for _ in range(reps):
        shuffled = rng.permutation(is_m)            # 성별 라벨을 무작위로 섞기
        diff = y[shuffled].mean() - y[~shuffled].mean()
        if abs(diff) >= abs(obs):
            count += 1
    return obs, count / reps

print(perm_test(df))                               # 전체
print(perm_test(df[df["department"] == "Marketing"]))""",
     """rng = np.random.default_rng(0)

def perm_test(sub, reps=10_000):
    y = sub["hired"].to_numpy()
    is_m = (sub["gender"] == "M").to_numpy()
    obs = y[is_m].mean() - y[~is_m].mean()
    count = 0
    for _ in range(reps):
        shuffled = rng.___(is_m)                   # 라벨 섞기
        diff = y[shuffled].mean() - y[~shuffled].mean()
        if abs(diff) >= abs(___):
            count += 1
    return obs, count / ___

print(perm_test(df))
print(perm_test(df[df["department"] == "Marketing"]))"""),

    ("## Step 7. 교차표와 카이제곱을 직접", """O = pd.crosstab(df["gender"], df["hired"]).to_numpy()   # 관측빈도 2x2
row = O.sum(axis=1)
col = O.sum(axis=0)
E = np.outer(row, col) / O.sum()                          # 기대빈도 = 행합*열합/n
chi2 = ((O - E) ** 2 / E).sum()
pval = 2 * (1 - Phi(math.sqrt(chi2)))                    # df=1이면 chi2 = z^2
print(O); print(E.round(1)); print(round(chi2, 2), pval)""",
     """O = pd.crosstab(df["gender"], df["hired"]).to_numpy()
row = O.sum(axis=___)
col = O.sum(axis=___)
E = np.___(row, col) / O.sum()
chi2 = ((O - E) ** 2 / ___).sum()
pval = 2 * (1 - Phi(math.sqrt(chi2)))
print(O); print(E.round(1)); print(round(chi2, 2), pval)"""),

    ("## Step 8. 상관계수를 직접", """def corr(a, b):
    za = (a - a.mean()) / a.std()      # numpy std = 분모 n
    zb = (b - b.mean()) / b.std()
    return (za * zb).mean()

xs = d["test_score"].to_numpy()
ys = d["hired"].to_numpy()
ex = d["years_experience"].to_numpy()
print(round(corr(xs, ys), 4), round(corr(ex, ys), 4), round(corr(xs, ex), 4))
print(d[["test_score", "years_experience", "hired"]].corr().round(4))""",
     """def corr(a, b):
    za = (a - a.mean()) / a.std()
    zb = (b - b.mean()) / b.std()
    return (za * ___).___()

xs = d["test_score"].to_numpy()
ys = d["hired"].to_numpy()
ex = d["years_experience"].to_numpy()
print(round(corr(xs, ys), 4), round(corr(ex, ys), 4), round(corr(xs, ex), 4))
print(d[["test_score", "years_experience", "hired"]].corr().round(4))"""),

    ("## Step 9. 단순회귀: 공식 두 가지로", """r = corr(xs, ys)
slope = r * ys.std() / xs.std()          # 방법 1: r * SDy / SDx
intercept = ys.mean() - slope * xs.mean()

X = np.column_stack([np.ones(len(xs)), xs])   # 방법 2: 정규방정식 (X^T X) b = X^T y
b = np.linalg.solve(X.T @ X, X.T @ ys)

resid = ys - (intercept + slope * xs)
rmse = math.sqrt((resid ** 2).mean())
print(round(slope, 5), round(intercept, 4), b.round(5))
print(round(rmse, 4), round(math.sqrt(1 - r ** 2) * ys.std(), 4))""",
     """r = corr(xs, ys)
slope = r * ys.std() / ___
intercept = ys.mean() - slope * ___

X = np.column_stack([np.ones(len(xs)), xs])
b = np.linalg.solve(X.T @ X, X.T @ ___)

resid = ys - (intercept + slope * xs)
rmse = math.sqrt((resid ** 2).mean())
print(round(slope, 5), round(intercept, 4), b.round(5))
print(round(rmse, 4), round(math.sqrt(1 - r ** 2) * ys.std(), 4))"""),

    ("## Step 10. 다중회귀: 행렬로 직접 (통제 후 성별 효과)", """Xdf = pd.get_dummies(d[["gender", "department", "degree"]], drop_first=True).astype(float)
Xdf["test_score"] = d["test_score"]
Xdf["years_experience"] = d["years_experience"]
Xdf.insert(0, "const", 1.0)
X = Xdf.to_numpy()
y = d["hired"].to_numpy()

XtX = X.T @ X
beta = np.linalg.solve(XtX, X.T @ y)               # (X^T X)^-1 X^T y
resid = y - X @ beta
n, k = X.shape
sigma2 = (resid ** 2).sum() / (n - k)              # 자유도 n - k
se = np.sqrt(np.diag(sigma2 * np.linalg.inv(XtX)))
t = beta / se
out = pd.DataFrame({"coef": beta, "se": se, "t": t, "p": [2 * (1 - Phi(abs(v))) for v in t]},
                   index=Xdf.columns)
print(out.round(4))""",
     """Xdf = pd.get_dummies(d[["gender", "department", "degree"]], drop_first=___).astype(float)
Xdf["test_score"] = d["test_score"]
Xdf["years_experience"] = d["years_experience"]
Xdf.insert(0, "const", 1.0)
X = Xdf.to_numpy()
y = d["hired"].to_numpy()

XtX = X.T @ X
beta = np.linalg.solve(XtX, X.T @ ___)
resid = y - X @ beta
n, k = X.shape
sigma2 = (resid ** 2).sum() / (___)                # 자유도
se = np.sqrt(np.diag(sigma2 * np.linalg.inv(XtX)))
t = beta / ___
out = pd.DataFrame({"coef": beta, "se": se, "t": t, "p": [2 * (1 - Phi(abs(v))) for v in t]},
                   index=Xdf.columns)
print(out.round(4))"""),

    ("## Step 11. 부트스트랩: 재표본으로 신뢰구간", """rng = np.random.default_rng(0)
y_all = df["hired"].to_numpy()
m_all = (df["gender"] == "M").to_numpy()
B, N = 5000, len(df)

idx = rng.integers(0, N, size=(B, N))             # B개의 재표본 인덱스를 한 번에 (벡터화)
yy, mm = y_all[idx], m_all[idx]
gaps = (yy * mm).sum(axis=1) / mm.sum(axis=1) - (yy * ~mm).sum(axis=1) / (~mm).sum(axis=1)
print(round(gaps.std(), 4), np.percentile(gaps, [2.5, 97.5]).round(3))""",
     """rng = np.random.default_rng(0)
y_all = df["hired"].to_numpy()
m_all = (df["gender"] == "M").to_numpy()
B, N = 5000, len(df)

idx = rng.integers(0, ___, size=(B, N))           # 복원추출 인덱스
yy, mm = y_all[idx], m_all[idx]
gaps = (yy * mm).sum(axis=1) / mm.sum(axis=1) - (yy * ~mm).sum(axis=1) / (~mm).sum(axis=1)
print(round(gaps.std(), 4), np.percentile(gaps, [___, ___]).round(3))"""),
]


def build(solution):
    nb = nbf.v4.new_notebook()
    for md, sol, blank in STEPS:
        nb.cells.append(nbf.v4.new_markdown_cell(md))
        code = sol if (solution or blank is None) else blank
        if code:
            nb.cells.append(nbf.v4.new_code_cell(code))
    return nb


if __name__ == "__main__":
    nbf.write(build(False), "../stats_basics_practice.ipynb")
    nbf.write(build(True), "../stats_basics_solution.ipynb")
    print("written")
