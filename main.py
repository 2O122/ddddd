# app.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

START_YEAR = 1908
END_YEAR = 2025
MIN_DAYS = 300
TRAIN_END = 2004
TEST_START = 2005


st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year

    df = df[df["연도"] <= END_YEAR]

    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    yearly = yearly[yearly["관측일수"] >= MIN_DAYS]
    yearly = yearly[yearly["연도"] >= START_YEAR]

    return yearly.sort_values("연도").reset_index(drop=True)


df = load_data()

if len(df) < 2:
    st.error("데이터가 부족합니다.")
    st.stop()


# ============================================================
# 전체 기간 선형회귀
# ============================================================

X_all = (df["연도"] - START_YEAR).to_numpy().reshape(-1, 1)
y_all = df["평균기온"].to_numpy()

linear_model = LinearRegression()
linear_model.fit(X_all, y_all)

slope_all = linear_model.coef_[0]
slope_100 = slope_all * 100

correlation = df["연도"].corr(df["평균기온"])


# ============================================================
# 최근 20년 선형회귀
# ============================================================

recent = df[
    (df["연도"] >= 2006) &
    (df["연도"] <= 2025)
].copy()

X_recent = (
    recent["연도"] - 2006
).to_numpy().reshape(-1, 1)

y_recent = recent["평균기온"].to_numpy()

recent_model = LinearRegression()
recent_model.fit(X_recent, y_recent)

recent_slope = recent_model.coef_[0]
recent_slope_100 = recent_slope * 100


# ============================================================
# 화면
# ============================================================

st.title("🌡️ 서울 기온 예측기")

st.header("장기적인 기온 상승 추세")

st.metric(
    "100년에 기온이 얼마나 오르는가?",
    f"{slope_100:+.2f} °C"
)


col1, col2 = st.columns(2)

with col1:
    st.metric(
        f"전체 기간 ({df['연도'].min()}~{df['연도'].max()})",
        f"{slope_100:+.2f} °C / 100년"
    )

with col2:
    st.metric(
        "최근 20년 (2006~2025)",
        f"{recent_slope_100:+.2f} °C / 100년"
    )


st.write(
    f"전체 기간 상관계수: **{correlation:.4f}**"
)


# ============================================================
# 연도 슬라이더
# ============================================================

st.header("연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도",
    1900,
    2100,
    2025
)

prediction = linear_model.predict(
    np.array([[selected_year - START_YEAR]])
)[0]

st.metric(
    f"{selected_year}년 예상 연평균기온",
    f"{prediction:.2f} °C"
)


# ============================================================
# 선형회귀 그래프
# ============================================================

years = np.arange(1900, 2101)

linear_pred = linear_model.predict(
    (years - START_YEAR).reshape(-1, 1)
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["연도"],
        y=df["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=6,
            color="#1f77b4"
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=years,
        y=linear_pred,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            color="red",
            width=3
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[prediction],
        mode="markers",
        name="선택 연도 예측",
        marker=dict(
            size=14,
            color="green"
        )
    )
)

fig.update_layout(
    title="서울 연평균기온과 선형회귀선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(range=[1900, 2100]),
    height=550
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 다항회귀
# ============================================================

st.header("1차·3차·9차 곡선 비교")

train = df[df["연도"] <= TRAIN_END].copy()
test = df[df["연도"] >= TEST_START].copy()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "훈련용 연도 수",
        f"{len(train)}개"
    )
    st.caption(
        f"{train['연도'].min()}~{train['연도'].max()}"
    )

with col2:
    st.metric(
        "테스트용 연도 수",
        f"{len(test)}개"
    )
    st.caption(
        f"{test['연도'].min()}~{test['연도'].max()}"
    )


# 연도를 작은 숫자로 변환
X_train = (
    train["연도"] - START_YEAR
).to_numpy().reshape(-1, 1)

y_train = train["평균기온"].to_numpy()

X_test = (
    test["연도"] - START_YEAR
).to_numpy().reshape(-1, 1)

y_test = test["평균기온"].to_numpy()


degrees = [1, 3, 9]

models = {}
result_rows = []


for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    # 훈련 데이터만 사용
    model.fit(X_train, y_train)

    # 테스트 데이터로만 평가
    test_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_pred
    )

    mse = mean_squared_error(
        y_test,
        test_pred
    )

    r2 = r2_score(
        y_test,
        test_pred
    )

    # 2050년 예측
    pred_2050 = model.predict(
        np.array([[2050 - START_YEAR]])
    )[0]

    models[degree] = model

    result_rows.append(
        {
            "모델": f"{degree}차",
            "테스트 MAE (°C)": mae,
            "테스트 MSE": mse,
            "테스트 R²": r2,
            "2050년 예측 (°C)": pred_2050
        }
    )


results = pd.DataFrame(result_rows)


st.subheader("테스트 데이터 평가 및 2050년 예측")

show_results = results.copy()

show_results["테스트 MAE (°C)"] = show_results[
    "테스트 MAE (°C)"
].map(lambda x: f"{x:.2f}")

show_results["테스트 MSE"] = show_results[
    "테스트 MSE"
].map(lambda x: f"{x:.2f}")

show_results["테스트 R²"] = show_results[
    "테스트 R²"
].map(lambda x: f"{x:.4f}")

show_results["2050년 예측 (°C)"] = show_results[
    "2050년 예측 (°C)"
].map(lambda x: f"{x:.2f}")

st.table(show_results)


# ============================================================
# 다항회귀 그래프
# ============================================================

st.subheader("훈련 데이터 / 테스트 데이터 / 회귀곡선")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["평균기온"],
        mode="markers",
        name="훈련용 데이터",
        marker=dict(
            size=6,
            color="blue",
            opacity=0.65
        )
    )
)

fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="markers",
        name="테스트용 데이터",
        marker=dict(
            size=8,
            color="orange",
            symbol="diamond"
        )
    )
)


plot_years = np.arange(
    int(df["연도"].min()),
    2051
)

plot_x = (
    plot_years - START_YEAR
).reshape(-1, 1)


colors = {
    1: "red",
    3: "green",
    9: "purple"
}


for degree in degrees:

    pred = models[degree].predict(plot_x)

    fig2.add_trace(
        go.Scatter(
            x=plot_years,
            y=pred,
            mode="lines",
            name=f"{degree}차",
            line=dict(
                color=colors[degree],
                width=3 if degree != 9 else 2,
                dash="dash" if degree == 9 else "solid"
            )
        )
    )


fig2.add_vline(
    x=2005,
    line_dash="dot",
    line_color="black"
)

fig2.add_annotation(
    x=2005,
    y=1,
    yref="paper",
    text="2005년 테스트 시작",
    showarrow=False,
    xanchor="left"
)

fig2.update_layout(
    title="다항회귀 모델 비교",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    height=650
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# ============================================================
# 테스트 실제값 / 예측값
# ============================================================

st.subheader("테스트 데이터 실제값과 예측값")

test_result = test[
    ["연도", "평균기온"]
].copy()

for degree in degrees:

    test_result[
        f"{degree}차 예측"
    ] = models[degree].predict(
        X_test
    )

st.dataframe(
    test_result.style.format(
        {
            "평균기온": "{:.2f}",
            "1차 예측": "{:.2f}",
            "3차 예측": "{:.2f}",
            "9차 예측": "{:.2f}"
        }
    ),
    use_container_width=True
)


# ============================================================
# 데이터
# ============================================================

with st.expander("연도별 데이터 보기"):

    st.dataframe(
        df[
            [
                "연도",
                "관측일수",
                "평균기온"
            ]
        ],
        use_container_width=True
    )
