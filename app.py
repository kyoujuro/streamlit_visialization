import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.tools.tools import add_constant
from bayes_opt import BayesianOptimization

st.set_page_config(layout="wide")
st.title("高機能データ分析 & 可視化ダッシュボード")

uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("データプレビュー")
    st.dataframe(df.head(), use_container_width=True)

if df is not None:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    st.sidebar.markdown("### 数値フィルター")
    filtered_df = df.copy()
    for col in num_cols:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        selected = st.sidebar.slider(f"{col}の範囲", min_val, max_val, (min_val, max_val))
        filtered_df = filtered_df[(filtered_df[col] >= selected[0]) & (filtered_df[col] <= selected[1])]

    st.sidebar.markdown("### 機能モードを選択")
    mode = st.sidebar.radio("選択してください", [
        "可視化", "統計分析", "回帰分析", "ベイズ最適化", "主成分分析", "クラスタリング" , "異常検知", "時系列解析"
    ])
    if mode == "可視化":
        st.header("可視化")
        x_col = st.selectbox("X軸を選択", num_cols)
        y_col = st.selectbox("Y軸を選択（省略可）", [None] + num_cols)
        z_col = st.selectbox("Z軸を選択（3D用・省略可）", [None] + num_cols)
        color_col = st.selectbox("色分け（オプション）", [None] + cat_cols)
        chart_type = st.radio("チャートタイプを選択", ["散布図", "ヒストグラム", "箱ひげ図", "密度プロット", "3D散布図"])

        if chart_type == "散布図" and y_col:
            fig = px.scatter(filtered_df, x=x_col, y=y_col, color=color_col)
            st.plotly_chart(fig)
        elif chart_type == "ヒストグラム":
            fig = px.histogram(filtered_df, x=x_col, color=color_col)
            st.plotly_chart(fig)
        elif chart_type == "箱ひげ図":
            fig = px.box(filtered_df, x=color_col, y=x_col)
            st.plotly_chart(fig)
        elif chart_type == "密度プロット" and y_col:
            fig = px.density_contour(filtered_df, x=x_col, y=y_col)
            st.plotly_chart(fig)
        elif chart_type == "3D散布図" and y_col and z_col:
            fig = px.scatter_3d(filtered_df, x=x_col, y=y_col, z=z_col, color=color_col)
            st.plotly_chart(fig)

    elif mode == "回帰分析":
        st.header("回帰分析")
        target_col = st.selectbox("目的変数を選択", num_cols)
        feature_cols = st.multiselect("説明変数を選択", [col for col in num_cols if col != target_col])
        reg_type = st.radio("回帰モデルを選択", ["線形回帰", "ロジスティック回帰", "SVM回帰", "ランダムフォレスト回帰"])

        if st.button("回帰分析を実行") and feature_cols:
            X = df[feature_cols]
            y = df[target_col]

            if reg_type == "線形回帰":
                X_const = add_constant(X)
                model = sm.OLS(y, X_const).fit()
                st.write(f"R²: {model.rsquared:.3f}")
                st.dataframe(pd.DataFrame({"係数": model.params, "P値": model.pvalues}))

            elif reg_type == "ロジスティック回帰":
                if y.nunique() > 2:
                    st.warning("ロジスティック回帰には2値分類が必要です")
                else:
                    model = LogisticRegression().fit(X, y)
                    pred = model.predict(X)
                    st.text("分類結果")
                    st.text(confusion_matrix(y, pred))
                    st.text(classification_report(y, pred))

            elif reg_type == "SVM回帰":
                kernel = st.selectbox("カーネルを選択", ["linear", "rbf"])
                model = SVR(kernel=kernel)
                model.fit(X, y)
                y_pred = model.predict(X)
                st.write(f"R²: {r2_score(y, y_pred):.3f}")
                st.write(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.3f}")
                fig = px.scatter(x=y, y=y_pred, labels={"x": "実測値", "y": "予測値"}, title="予測値 vs 実測値")
                st.plotly_chart(fig)

            elif reg_type == "ランダムフォレスト回帰":
                model = RandomForestRegressor(random_state=0)
                model.fit(X, y)
                y_pred = model.predict(X)
                st.write(f"R²: {r2_score(y, y_pred):.3f}")
                st.write(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.3f}")
                importance_df = pd.DataFrame({"特徴量": feature_cols, "重要度": model.feature_importances_})
                st.dataframe(importance_df.sort_values(by="重要度", ascending=False))
                fig = px.scatter(x=y, y=y_pred, labels={"x": "実測値", "y": "予測値"}, title="予測値 vs 実測値")
                st.plotly_chart(fig)
    elif mode == "統計分析":
        st.header("統計分析と多重比較法")
        group_col = st.selectbox("グループ（カテゴリ）列を選択", cat_cols)
        value_col = st.selectbox("対象の数値列を選択", num_cols)
        test_type = st.selectbox("多重比較法の選択", ["Tukey", "Dunn", "Bonferroni"])

        if st.button("統計解析を実行"):
            try:
                if test_type == "Tukey":
                    tukey = pairwise_tukeyhsd(endog=df[value_col], groups=df[group_col], alpha=0.05)
                    st.write(tukey.summary())

                elif test_type == "Dunn":
                    pvals = sp.posthoc_dunn(df, val_col=value_col, group_col=group_col, p_adjust='bonferroni')
                    st.write(pvals)

                elif test_type == "Bonferroni":
                    groups = [g[value_col].dropna().values for name, g in df.groupby(group_col)]
                    stat, p = stats.f_oneway(*groups)
                    st.write(f"ANOVA p値: {p:.4f}")

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
    elif mode == "異常検知":
        st.header("異常検知（Anomaly Detection）")
        selected_cols = st.multiselect("異常検知に使用する数値列を選択", num_cols)
        contamination = st.slider("異常データの割合 (contamination)", 0.01, 0.2, 0.05, step=0.01)

        if st.button("異常検知を実行") and selected_cols:
            from sklearn.ensemble import IsolationForest
            X = df[selected_cols].dropna()
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(X)
            scores = model.decision_function(X)

            result_df = X.copy()
            result_df['anomaly'] = preds
            result_df['score'] = scores

            st.subheader("異常検知結果")
            st.dataframe(result_df)

            if len(selected_cols) >= 2:
                fig = px.scatter(result_df, x=selected_cols[0], y=selected_cols[1], color=result_df['anomaly'].astype(str), title="異常 vs 正常 (2D)", labels={"color": "anomaly"})
                st.plotly_chart(fig)

            if len(selected_cols) >= 3:
                fig = px.scatter_3d(result_df, x=selected_cols[0], y=selected_cols[1], z=selected_cols[2], color=result_df['anomaly'].astype(str), title="異常 vs 正常 (3D)", labels={"color": "anomaly"})
                st.plotly_chart(fig)

    elif mode == "時系列解析":
        st.header("時系列解析")
        datetime_col = st.selectbox("日付列を選択", df.columns[df.dtypes == 'object'])
        value_col = st.selectbox("対象とする数値列を選択", num_cols)

        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df_sorted = df.sort_values(datetime_col)
            df_sorted = df_sorted[[datetime_col, value_col]].dropna()
            df_sorted.set_index(datetime_col, inplace=True)

            st.line_chart(df_sorted)

            from statsmodels.tsa.seasonal import STL
            stl = STL(df_sorted[value_col], period=7)
            result = stl.fit()

            st.subheader("STL 分解結果")
            fig, axs = plt.subplots(4, 1, figsize=(10, 8))
            axs[0].plot(result.observed)
            axs[0].set_title("Observed")
            axs[1].plot(result.trend)
            axs[1].set_title("Trend")
            axs[2].plot(result.seasonal)
            axs[2].set_title("Seasonal")
            axs[3].plot(result.resid)
            axs[3].set_title("Residual")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

    elif mode == "主成分分析":
        st.header("主成分分析（PCA）")
        selected_cols = st.multiselect("主成分分析に使用する数値列を選択", num_cols)
        if st.button("主成分分析を実行") and selected_cols:
            X = df[selected_cols].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA()
            components = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(data=components, columns=[f"PC{i+1}" for i in range(components.shape[1])])
            st.subheader("主成分スコア")
            st.dataframe(pca_df)

            pcx = st.selectbox("X軸の主成分", pca_df.columns, index=0)
            pcy = st.selectbox("Y軸の主成分", pca_df.columns, index=1)
            fig = px.scatter(pca_df, x=pcx, y=pcy, title=f"PCA散布図: {pcx} vs {pcy}")
            st.plotly_chart(fig)

    elif mode == "クラスタリング":
        st.header("KMeansクラスタリング")
        selected_cols = st.multiselect("クラスタリングに使用する数値列を選択", num_cols)
        n_clusters = st.slider("クラスタ数を選択", 2, 10, 3)
        is_3d = st.checkbox("3D表示")

        if st.button("クラスタリングを実行") and selected_cols:
            X = df[selected_cols].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_scaled)
            cluster_labels = kmeans.labels_
            cluster_df = pd.DataFrame(X, columns=selected_cols)
            cluster_df["cluster"] = cluster_labels

            if is_3d and len(selected_cols) >= 3:
                fig = px.scatter_3d(cluster_df, x=selected_cols[0], y=selected_cols[1], z=selected_cols[2],
                                    color=cluster_df["cluster"].astype(str), title="KMeansクラスタリング（3D）")
            else:
                fig = px.scatter(cluster_df, x=selected_cols[0], y=selected_cols[1],
                                 color=cluster_df["cluster"].astype(str), title="KMeansクラスタリング（2D）")
            st.plotly_chart(fig)

    elif mode == "ベイズ最適化":
        st.header("ベイズ最適化（Bayesian Optimization）")
        st.markdown("CSVデータから目的変数と説明変数を指定し、最適なパラメータを探索します。")

        target = st.selectbox("目的変数を選択（数値）", num_cols)
        features = st.multiselect("説明変数を選択", [c for c in num_cols if c != target])

        if features and st.button("最適化を実行"):
            X = df[features]
            y = df[target]
            model = RandomForestRegressor(random_state=0)
            model.fit(X, y)

            pbounds = {col: (float(X[col].min()), float(X[col].max())) for col in features}

            def black_box(**kwargs):
                X_new = pd.DataFrame([kwargs])
                X_new = X_new[features]
                return model.predict(X_new)[0]

            optimizer = BayesianOptimization(f=black_box, pbounds=pbounds, random_state=42)
            optimizer.maximize(init_points=5, n_iter=15)

            st.subheader("最適化結果")
            best = optimizer.max
            st.write("最適値:", best["target"])
            st.write("最適パラメータ:", best["params"])
            hist = pd.DataFrame([res for res in optimizer.res])
            st.dataframe(hist)
            fig = px.line(hist, y="target", title="最適化履歴（目的関数値）")
            st.plotly_chart(fig)