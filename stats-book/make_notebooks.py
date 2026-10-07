"""분류 워크북용 노트북 2개 생성: 연습용(빈칸) / 정답용.

각 단계는 (markdown, 정답 코드, 빈칸 코드) 세트. 빈칸 코드가 None이면 두 노트북에 같은 코드.
"""
import nbformat as nbf

STEPS = [
    ("# 분류 워크북: 채용 데이터로 연습하기\n"
     "책 `classification-workbook.pdf`와 같은 순서입니다. `___` 빈칸을 채우고 실행한 뒤, 책의 '실행 결과'와 비교하세요.",
     None, None),

    ("## Step 0. 준비", """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)""", None),

    ("## Step 1. 데이터 준비: X와 y", """df = pd.read_csv("hiring_practice.csv")
d = df.dropna()
cat = ["gender", "department", "degree"]
num = ["test_score", "years_experience"]
X = d[cat + num]
y = d["hired"]
print(X.shape, round(y.mean(), 3))   # (582, 5) 0.325""",
     """df = pd.read_csv("hiring_practice.csv")
d = df.___()                          # 결측 행 제거
cat = ["gender", "department", "degree"]
num = ["test_score", "years_experience"]
X = d[___]                            # 설명변수 5개
y = d[___]                            # 결과변수
print(X.shape, round(y.mean(), 3))   # (582, 5) 0.325"""),

    ("## Step 2. train / test 나누기", """X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)
print(len(X_train), len(X_test), round(y_train.mean(), 3), round(y_test.mean(), 3))
# 436 146 0.326 0.322""",
     """X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=___, stratify=___, random_state=0)
print(len(X_train), len(X_test), round(y_train.mean(), 3), round(y_test.mean(), 3))
# 436 146 0.326 0.322"""),

    ("## Step 3. 기준선(baseline)", """dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print(round(accuracy_score(y_test, dummy.predict(X_test)), 3))   # 0.678""",
     """dummy = DummyClassifier(strategy=___).fit(X_train, y_train)
print(round(accuracy_score(y_test, dummy.predict(X_test)), 3))   # 0.678"""),

    ("## Step 4. 전처리 파이프라인", """pre = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), cat),
    ("num", StandardScaler(), num),
])
pre.fit(X_train)
print(pre.get_feature_names_out())
print(pre.transform(X_train.head(3)).round(2))""",
     """pre = ColumnTransformer([
    ("cat", ___(drop="first"), cat),   # 범주 -> 0/1 열
    ("num", ___(), num),               # 숫자 -> 평균 0, SD 1
])
pre.fit(X_train)
print(pre.get_feature_names_out())
print(pre.transform(X_train.head(3)).round(2))"""),

    ("## Step 5. 로지스틱 회귀 학습과 예측", """logit = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
logit.fit(X_train, y_train)
p = logit.predict_proba(X_test)[:, 1]      # 합격 확률
y_hat = (p >= 0.5).astype(int)             # 0.5 기준으로 0/1
print(p[:5].round(3), y_hat[:5])           # [0.502 0.286 0.535 0.091 0.427] [1 0 1 0 0]""",
     """logit = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
logit.___(X_train, y_train)
p = logit.___(X_test)[:, 1]                # 합격 확률 (두 번째 열)
y_hat = (p >= ___).astype(int)
print(p[:5].round(3), y_hat[:5])           # [0.502 0.286 0.535 0.091 0.427] [1 0 1 0 0]"""),

    ("## Step 6. 평가: 혼동행렬과 지표", """print(confusion_matrix(y_test, y_hat))       # [[95  4] [31 16]]
print("accuracy ", round(accuracy_score(y_test, y_hat), 3))    # 0.760
print("precision", round(precision_score(y_test, y_hat), 3))   # 0.800
print("recall   ", round(recall_score(y_test, y_hat), 3))      # 0.340
print("f1       ", round(f1_score(y_test, y_hat), 3))          # 0.478
print("AUC      ", round(roc_auc_score(y_test, p), 3))         # 0.796""",
     """print(confusion_matrix(y_test, y_hat))       # [[95  4] [31 16]]
tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
print("accuracy ", (___) / len(y_test))      # 0.760  직접 계산
print("precision", tp / (___))               # 0.800
print("recall   ", tp / (___))               # 0.340
print("AUC      ", round(roc_auc_score(y_test, ___), 3))   # 0.796 (확률을 넣는다)"""),

    ("## Step 7. 임계값(threshold) 바꾸기", """for thr in [0.3, 0.4, 0.5]:
    yh = (p >= thr).astype(int)
    print(thr, confusion_matrix(y_test, yh).ravel(),
          "prec", round(precision_score(y_test, yh), 3),
          "rec", round(recall_score(y_test, yh), 3))
# 0.3 [71 28 16 31] prec 0.525 rec 0.66
# 0.4 [83 16 19 28] prec 0.636 rec 0.596
# 0.5 [95  4 31 16] prec 0.8   rec 0.34""",
     """for thr in [0.3, 0.4, 0.5]:
    yh = (p >= ___).astype(int)
    print(thr, confusion_matrix(y_test, yh).ravel(),
          "prec", round(precision_score(y_test, yh), 3),
          "rec", round(recall_score(y_test, yh), 3))"""),

    ("## Step 8. 다른 모델과 교차검증", """pre_tree = ColumnTransformer([("cat", OneHotEncoder(drop="first"), cat)], remainder="passthrough")
tree = Pipeline([("pre", pre_tree),
                 ("clf", DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0))])
forest = Pipeline([("pre", pre_tree),
                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=0))])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, m in [("logit", logit), ("tree", tree), ("forest", forest)]:
    s = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
    print(name, s.round(3), "mean", s.mean().round(3))
# logit  mean 0.758 / tree mean 0.714 / forest mean 0.763""",
     """pre_tree = ColumnTransformer([("cat", OneHotEncoder(drop="first"), cat)], remainder="passthrough")
tree = Pipeline([("pre", pre_tree),
                 ("clf", DecisionTreeClassifier(max_depth=___, min_samples_leaf=20, random_state=0))])
forest = Pipeline([("pre", pre_tree),
                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=0))])
cv = StratifiedKFold(___, shuffle=True, random_state=0)
for name, m in [("logit", logit), ("tree", tree), ("forest", forest)]:
    s = cross_val_score(m, ___, ___, cv=cv, scoring="roc_auc")   # 전체 X, y
    print(name, s.round(3), "mean", s.mean().round(3))"""),

    ("## Step 9. 해석: 계수와 중요도", """names = logit.named_steps["pre"].get_feature_names_out()
coef = logit.named_steps["clf"].coef_[0]
print(pd.DataFrame({"feature": names, "coef": coef.round(3), "odds_ratio": np.exp(coef).round(2)}))

from sklearn.inspection import permutation_importance
pi = permutation_importance(logit, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=0)
print(pd.Series(pi.importances_mean, index=X.columns).round(3).sort_values(ascending=False))""",
     """names = logit.named_steps["pre"].get_feature_names_out()
coef = logit.named_steps["clf"].coef_[0]
print(pd.DataFrame({"feature": names, "coef": coef.round(3), "odds_ratio": np.___(coef).round(2)}))

from sklearn.inspection import permutation_importance
pi = permutation_importance(logit, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=0)
print(pd.Series(pi.importances_mean, index=X.columns).round(3).sort_values(ascending=False))"""),

    ("## Step 10. 공정성 감사(audit)", """def group_report(p, thr=0.5):
    t = X_test.copy()
    t["y"] = y_test.values
    t["y_hat"] = (p >= thr).astype(int)
    return t.groupby("gender").apply(lambda g: pd.Series({
        "n": len(g),
        "actual_rate": g["y"].mean(),
        "selection_rate": g["y_hat"].mean(),
        "TPR": g.loc[g["y"] == 1, "y_hat"].mean(),
        "FPR": g.loc[g["y"] == 0, "y_hat"].mean(),
    }), include_groups=False).round(3)

print(group_report(p))

# 성별을 빼고 학습하면?
pre2 = ColumnTransformer([("cat", OneHotEncoder(drop="first"), ["department", "degree"]),
                          ("num", StandardScaler(), num)])
logit2 = Pipeline([("pre", pre2), ("clf", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
p2 = logit2.predict_proba(X_test)[:, 1]
print(group_report(p2))

# 반사실: 성별만 바꾸면 예측이 얼마나 바뀌나
X_flip = X_test.copy()
X_flip["gender"] = X_flip["gender"].map({"M": "F", "F": "M"})
p_flip = logit.predict_proba(X_flip)[:, 1]
print("결정이 바뀐 사람:", ((p_flip >= 0.5) != (p >= 0.5)).sum(), "/", len(X_test))   # 14 / 146""",
     """def group_report(p, thr=0.5):
    t = X_test.copy()
    t["y"] = y_test.values
    t["y_hat"] = (p >= thr).astype(int)
    return t.groupby(___).apply(lambda g: pd.Series({
        "n": len(g),
        "actual_rate": g["y"].mean(),
        "selection_rate": g["y_hat"].mean(),
        "TPR": g.loc[g["y"] == ___, "y_hat"].mean(),   # 실제 합격자 중 합격 예측
        "FPR": g.loc[g["y"] == ___, "y_hat"].mean(),   # 실제 불합격자 중 합격 예측
    }), include_groups=False).round(3)

print(group_report(p))

# 성별을 빼고 학습하면?
pre2 = ColumnTransformer([("cat", OneHotEncoder(drop="first"), [___]),
                          ("num", StandardScaler(), num)])
logit2 = Pipeline([("pre", pre2), ("clf", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
p2 = logit2.predict_proba(X_test)[:, 1]
print(group_report(p2))

# 반사실: 성별만 바꾸면 예측이 얼마나 바뀌나
X_flip = X_test.copy()
X_flip["gender"] = X_flip["gender"].map({"M": ___, "F": ___})
p_flip = logit.predict_proba(X_flip)[:, 1]
print("결정이 바뀐 사람:", ((p_flip >= 0.5) != (p >= 0.5)).sum(), "/", len(X_test))   # 14 / 146"""),
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
    nbf.write(build(False), "../classification_practice.ipynb")
    nbf.write(build(True), "../classification_solution.ipynb")
    print("written")
