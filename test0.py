import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 폰트 설정: 기본 폰트로 설정하여 글씨체 깨짐 방지
matplotlib.rcParams['font.family'] = 'Arial'

def main():
    st.sidebar.title("This is Text Elements")
    st.sidebar.header("This is Header")
    st.sidebar.subheader("This is Subheader")

    st.markdown("This text is :red[colored red], and this is :blue[colored blue] and **bold**.")
    st.write("-" * 50)

    st.markdown(""" 
    ### SubChapter 1  
    - :red[$\\sqrt{x^2+y^2}=1$] is a Pythagorean identity. :pencil:
    """)
    st.write("-" * 50)

    st.markdown(
        "## Chapter 1.  \n"
        "- Streamlit is **_really_ cool**.  \n"
        "- This text is :blue[colored blue], and this is :red[colored red] and **bold**."
    )

    st.write("## 📊 변수 선택 및 모델 학습")

    # ✅ 데이터 로딩 (로컬 파일)
    try:
        df = pd.read_csv("SN_total.csv")
    except FileNotFoundError:
        st.error("❌ 'SN_total.csv' 파일을 찾을 수 없습니다. 같은 폴더에 파일이 있는지 확인해주세요.")
        return

    columns = df.columns.tolist()

    # 🎯 타겟 선택
    target = st.sidebar.radio("🎯 타겟 변수 선택", options=columns)

    # 🔢 입력 변수 선택
    input_vars = st.sidebar.multiselect("📥 입력 변수 선택", options=[col for col in columns if col != target])

    if input_vars:
        selected_df = df[[target] + input_vars]
        st.write("✅ 선택된 데이터")
        st.dataframe(selected_df)

        # 하이퍼파라미터 슬라이더
        st.write("## ⚙️ 하이퍼파라미터 설정")
        max_depth = st.sidebar.slider("max_depth", 1, 20, 5)
        n_estimators = st.sidebar.slider("n_estimators", 10, 500, 100, step=10)
        learning_rate = st.sidebar.slider("learning_rate", 0.001, 1.0, 0.1, step=0.01)
        subsample = st.sidebar.slider("subsample", 0.1, 1.0, 1.0, step=0.1)

        # 학습 버튼
        if st.button("🚀 모델 학습 및 예측"):
            X = df[input_vars]
            y = df[target]

            # 훈련 데이터와 테스트 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # XGBoost 모델 학습
            model = XGBRegressor(
                max_depth=max_depth,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)  # 훈련 데이터 예측
            y_pred_test = model.predict(X_test)   # 테스트 데이터 예측

            # MSE, RMSE, R² 계산
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)

            rmse_train = np.sqrt(mse_train)
            rmse_test = np.sqrt(mse_test)

            # 성능 지표 출력
            st.success("✅ 모델 학습 완료!")
            st.write(f"📉 **훈련 데이터 MSE:** {mse_train:.4f}, **테스트 데이터 MSE:** {mse_test:.4f}")
            st.write(f"📊 **훈련 데이터 RMSE:** {rmse_train:.4f}, **테스트 데이터 RMSE:** {rmse_test:.4f}")
            st.write(f"📈 **훈련 데이터 R²:** {r2_train:.4f}, **테스트 데이터 R²:** {r2_test:.4f}")

            # 훈련 데이터 그래프 (실제값 vs 예측값)
            fig_train, ax_train = plt.subplots(figsize=(10, 6))
            ax_train.scatter(y_train, y_pred_train, color='green', label=f"훈련 데이터 (R²={r2_train:.4f}, MSE={mse_train:.4f})")
            ax_train.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label="완벽한 예측선")
            ax_train.set_title("학습 데이터: 실제값 vs 예측값")
            ax_train.set_xlabel("실제값")
            ax_train.set_ylabel("예측값")
            ax_train.legend()

            # 훈련 데이터 그래프 출력
            st.pyplot(fig_train)

            # 테스트 데이터 그래프 (실제값 vs 예측값)
            fig_test, ax_test = plt.subplots(figsize=(10, 6))
            ax_test.scatter(y_test, y_pred_test, color='blue', label=f"테스트 데이터 (R²={r2_test:.4f}, MSE={mse_test:.4f})")
            ax_test.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="완벽한 예측선")
            ax_test.set_title("테스트 데이터: 실제값 vs 예측값")
            ax_test.set_xlabel("실제값")
            ax_test.set_ylabel("예측값")
            ax_test.legend()

            # 테스트 데이터 그래프 출력
            st.pyplot(fig_test)

if __name__ == '__main__':
    main()