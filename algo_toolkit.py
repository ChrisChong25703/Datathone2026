"""알고리즘 책 1-8장을 datathon에서 바로 쓰는 도구 모음.

사용법:  python algo_toolkit.py hiring_practice.csv
다른 데이터면 아래 CONFIG만 바꾸면 된다.
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeClassifier, export_text

CONFIG = {
    "outcome": "hired",                       # Y (0/1)
    "protected": "gender",                    # A
    "categorical": ["department", "degree"],  # 범주형 자격/구조 변수
    "numeric": ["test_score", "years_experience"],
}


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# 1장 벡터화 + 4장 groupby(counting sort): 그룹 비율표를 한 번에
def rate_table(df, by):
    y = CONFIG["outcome"]
    t = df.groupby(by, observed=True)[y].agg(n="size", rate="mean")
    t["se"] = np.sqrt(t.rate * (1 - t.rate) / t.n)
    return t.round(3)


# 8장 minimax: 가장 불리한/유리한 셀
def worst_and_best(df, by):
    t = rate_table(df, by)
    return t.loc[[t.rate.idxmin(), t.rate.idxmax()]]


# 3장 selection: 상위 k (전체 정렬 없이)
def top_k(df, col, k=5):
    return df.nlargest(k, col)


# 5장 이진 탐색: pd.cut (경계 배열에서 searchsorted) -> 평균의 그래프
def graph_of_averages(df, col, q=8):
    bins = pd.qcut(df[col], q, duplicates="drop")   # 같은 인원수로 나누기
    return df.groupby(bins, observed=True)[CONFIG["outcome"]].agg(n="size", rate="mean").round(3)


# 6장 DFS 트리: 회귀가 놓친 interaction 후보 찾기
def tree_patterns(df, depth=3):
    cols = CONFIG["categorical"] + CONFIG["numeric"] + [CONFIG["protected"]]
    X = pd.get_dummies(df[cols], drop_first=True).astype(float)
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, random_state=0)
    clf.fit(X, df[CONFIG["outcome"]])
    print(export_text(clf, feature_names=list(X.columns)))
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp[imp > 0].round(3)


# 7장 greedy heuristic: AIC 기준 forward stepwise
def forward_select(df):
    terms = [f"C({c})" for c in CONFIG["categorical"] + [CONFIG["protected"]]] + CONFIG["numeric"]
    chosen, y = [], CONFIG["outcome"]
    best_aic = smf.logit(f"{y} ~ 1", df).fit(disp=0).aic
    while terms:
        aics = {t: smf.logit(f"{y} ~ " + " + ".join(chosen + [t]), df).fit(disp=0).aic for t in terms}
        t = min(aics, key=aics.get)
        if aics[t] >= best_aic:
            break
        print(f"  + {t:<20} AIC {best_aic:.1f} -> {aics[t]:.1f}")
        best_aic = aics[t]
        chosen.append(t)
        terms.remove(t)
    return chosen


# 9장(통계책) + 1장 벡터화: bootstrap을 반복문 대신 행렬로
def bootstrap_gap(df, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = df[CONFIG["outcome"]].to_numpy()
    g = (df[CONFIG["protected"]] == df[CONFIG["protected"]].value_counts().index[-1]).to_numpy()
    idx = rng.integers(0, len(df), size=(B, len(df)))   # B x n 재표본 인덱스를 한 번에
    yy, gg = y[idx], g[idx]
    gap = (yy * gg).sum(1) / gg.sum(1) - (yy * ~gg).sum(1) / (~gg).sum(1)
    return gap.mean().round(3), np.percentile(gap, [2.5, 97.5]).round(3)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "hiring_practice.csv"
    df = pd.read_csv(path).dropna()
    A = CONFIG["protected"]

    section("그룹별 합격률 (groupby = counting sort)")
    print(rate_table(df, A))
    section("가장 불리한 / 유리한 셀 (minimax)")
    print(worst_and_best(df, ["department", A]))
    section("점수 상위 5명 (nlargest = selection)")
    print(top_k(df, "test_score")[["gender", "department", "test_score", "hired"]])
    section("점수 구간별 합격률 (qcut = binary search)")
    print(graph_of_averages(df, "test_score"))
    section("결정트리 패턴 (DFS)")
    print(tree_patterns(df))
    section("Forward stepwise (greedy)")
    print("선택 순서:", forward_select(df))
    section("성별 격차 bootstrap (벡터화)")
    mean_gap, ci = bootstrap_gap(df)
    print(f"{A} 격차 평균 {mean_gap}, 95% 구간 {ci}")
