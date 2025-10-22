# Optunaのインストール
!pip install optuna -q

# XGBoostのインストール
!pip3 install xgboost -q
!pip3 install -q pydot
!pip3 install graphviz -q

# CatBoostのインストール
!pip install catboost --quiet

# 拡張子 xls を読み込むためのライブラリーのインストール
!pip install xlrd

# rickのインストール
# !pip install rich

# SHAPのインストール
!pip install shap -q

import os
# 分析対象のExcelファイルのパスを path_in とする。
path_in = '/content/drive/MyDrive/RTXML/kCVwith21Methods/rtx_ssc_hc_for_python.csv'
# 分析結果をExcelファイルで出力するフォルダーを path_out で指定する。計算するたびに結果を保存するfolderを変えると良い。
path_out = '/content/drive/MyDrive/RTXML/kCVwith21Methods/run2'
os.makedirs(path_out, exist_ok=True)
# 予測モデルをPickleで保存するフォルダーを path_model で指定する。計算するたびに結果を保存するfolderを変えると良い。
path_model = '/content/drive/MyDrive/RTXML/kCVwith21Methods/run2'
os.makedirs(path_model, exist_ok=True)

# データが線形(1)か非線形(0)か。defaultは0 (非線形) 。通常は非線形で計算する。
linear = 1
# 線形のデータでPrecisionやRecallが偏る場合、linear = 1 とすることで、SVMのkernelとXGBoostのboosterが線形になる。

# データがサンプル数が多いBig Dataか(1)否か(0)。defaultは0 (非Big Data)。通常は非BigDAtaで計算する。
bigdata = 1
# Big Dataで計算時間が数時間に及ぶ場合、bigdata = 1 とすると、RandomForestとLightGBMのOptunaとCatBoostが省かれる。

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import pickle
import os, sys
from collections import defaultdict
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score, precision_score, recall_score, f1_score
warnings.simplefilter(action='ignore', category=UserWarning)



# Pandasによるデータの読み込み。　path_inは分析対象のCSVファイルのGoogle Drive上のディレクトリ
root, ext0 = os.path.splitext(path_in)
ext = ext0[1:]
if ext == 'xlsx':
    df = pd.read_excel(str(path_in))
elif ext == 'csv':
    df = pd.read_csv(str(path_in))
elif ext == 'xls':
    df = pd.read_excel(str(path_in))
else:
    print('The input data was neither Excle file nor CSV file.')
    sys.exit()

# Numpyのデータの作成
data = df.values
x = df.iloc[:, :-1].values
t = df.iloc[:, -1].values



# kの設定
n_samples = len(df)
# Minimum number of samples you want in the test set
min_test_samples = 50
# Determine k based on the dataset size and the minimum number of test samples
k = max(2, np.floor(n_samples / min_test_samples).astype(int))
# Adjust k if necessary based on additional criteria
# For example, ensure k does not exceed a certain maximum value
max_k = 10
k = min(k, max_k)
# k-foldのkをマニュアルで決める場合は次の行を加える｡
k = 5



if bigdata == 0 and linear == 0:
    print('')
    print(f'■ The {k}-fold Cross Validation with 21 different Machien Learning models for Non-Linear Standard data')
    print('')
elif bigdata == 0 and linear == 1:
    print('')
    print(f'■ The {k}-fold Cross Validation with 21 different Machien Learning models for Linear Standard data')
    print('')
elif bigdata == 1 and linear == 0:
    print('')
    print(f'■ The {k}-fold Cross Validation with 21 different Machien Learning models for Non-Linear Big data')
    print('')
elif bigdata == 1 and linear == 1:
    print('')
    print(f'■ The {k}-fold Cross Validation with 21 different Machien Learning models for Linear Big data')
    print('')
else:
    print('Unexpected error concerning Linearity and Size of data')
    sys.exit()


print('')
print(f'◆ The sample size of dataset: {n_samples}')
print('')
# データの大きさ
print('◇　行と列の数')
print(df.shape)
print('')
# データの分割数 k
print(f"Chosen value of k for k-fold cross-validation: {k}")
print('')
print('')



# Optunaにおける計算回数の設定。Trialの数を増やすと予測精度は上がるが、計算時間が長くなる。
random_state = 42
n_trialsA = 100 # Lasso, Ridge
n_trialsB = 30  # RandomForest
n_trialsC = 40. # XGBoost
n_trialsD = 10. # LightGBM



# Feature Importanceの図のサイズ
size_x = 8
size_y = 4.8



# 目的変数の分布の表示
print('◇　目的変数の分布')
sns.displot(df.iloc[:,-1].dropna())
print('　　　　　陽性症例の割合(事前確率): ' + str(round(100*(np.count_nonzero(t>0)/len(t)), 5)) + ' %')
print('')
print('')
plt.show()


# 現在時刻（計算開始時）
from datetime import datetime
start_time = datetime.now()
formatted_start_time = start_time.strftime('%Y年%m月%d日 %H時%M分%S秒')
print(f'◆ Present Time (Start): {formatted_start_time}')
print('')


# 出力ファイル
filename0 = os.path.basename(path_in)
filename = os.path.splitext(filename0)[0]
filepath = path_out + '/AllEvaluationResultsFor' + filename + str(formatted_start_time) + '.xlsx'
# filepath1 = path_out + '/ResultsOf' + filename + 'wihtoutNB&kNN' + str(formatted_time) + '.xlsx'
# filepath2 = path_out + '/ResultsOf' + filename + 'withNB' + str(formatted_time) + '.xlsx'
# filepath3 = path_out + '/ResultsOf' + filename + 'withNB&kNN' + str(formatted_time) + '.xlsx'


print('◇　計算結果の出力先')
print(filepath)
# print('1. Naive Bayes も k-Nearest Neighbours も含めない結果')
# print(filepath1)
# print('2. Naive Bayes を含めた結果')
# print(filepath2)
# print('3. Naive Bayes も k-NN も含めた結果')
# print(filepath3)
print('')
print('')

# simple Linear Regression
def LinearKFold(k=k):
    from sklearn.linear_model import LinearRegression

    print('')
    print('■ Linear model')
    print('')

    # df = pd.read_csv(str(path_in))
    x = df.iloc[:, :-1].values
    t = df.iloc[:, -1].values

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    lr = LinearRegression()

    aucs = []
    r2s = []
    accs = []
    precs = []
    recs = []
    f1s = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        t_train, t_test = t[train_index], t[test_index]

        lr.fit(x_train, t_train)
        predictions = lr.predict(x_test)

        # Assuming the task is a regression with binary targets; adjust accordingly
        predictions_round = [1 if p >= 0.5 else 0 for p in predictions]

        auc = roc_auc_score(t_test, predictions)
        # r2 = r2_score(t_test, predictions)
        acc = accuracy_score(t_test, predictions_round)
        prec = precision_score(t_test, predictions_round)
        rec = recall_score(t_test, predictions_round)
        f1 = f1_score(t_test, predictions_round)

        aucs.append(auc)
        # r2s.append(r2)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    # Calculate mean scores
    mean_auc = np.mean(aucs)
    # mean_r2 = np.mean(r2s)
    mean_acc = np.mean(accs)
    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)
    mean_f1 = np.mean(f1s)

    # Retrain the model on the full dataset
    model_full = LinearRegression()
    model_full.fit(x, t)

    # Extracting feature importances (coefficients in this case)
    feature_importances = model_full.coef_
    column_names = df.columns[:-1]
    feature_importances_df = pd.DataFrame({'Feature': column_names, 'Importance': feature_importances})

    # Sort by absolute values of feature importances for both export and identification of top features
    # feature_importances_df_sorted = feature_importances_df.assign(Abs_Importance=feature_importances_df['Importance'].abs())\
    #                                                     .sort_values(by='Abs_Importance', ascending=False)\
    #                                                     .drop('Abs_Importance', axis=1)

    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)

    # os.makedirs(path_out, exist_ok=True)
    excel_filename = os.path.join(path_out, 'feature_importances_linear_kfold.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)

    # print('')
    # print('◆　Coefficients of Linear model are exported as feature_importances_linear_kfold.xlsx')
    # print('')

    # Sort the features by their importances
    importance_sorted_idx = np.argsort(np.abs(feature_importances))[::-1]
    top_idxs = importance_sorted_idx[:10]

    print(f'□ {k}-fold Cross Validation を用いた Simple Linear Regression による2値分類の検定結果')
    print('')
    print(f'Mean AUC: {mean_auc:.4f}')
    # print(f'Mean R2: {mean_r2}')
    print(f'Mean Accuracy: {mean_acc:.4f}')
    print(f'Mean Precision: {mean_prec:.4f}')
    print(f'Mean Recall: {mean_rec:.4f}')
    print(f'Mean F1 Score: {mean_f1:.4f}')
    print('')

    print('')
    print('◇ Linear Regression model の 係数による Feature Importance')
    print('')

    # Plotting
    plt.figure(figsize=(size_x, size_y))
    plt.barh(range(len(top_idxs)), feature_importances[top_idxs], align='center')
    plt.yticks(range(len(top_idxs)), [column_names[i] for i in top_idxs])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 10 Feature Importances in {k}-fold CV of Linear Regression')
    plt.gca().invert_yaxis()  # Invert y-axis to have the feature with highest importance at the top
    plt.show()

    # Saving the model
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_linear_model.pkl')

    with open(model_filename, 'wb') as file:
        pickle.dump(model_full, file)

    res = np.array([mean_auc, mean_acc, mean_prec, mean_rec, mean_f1], dtype=object)
    res2=pd.DataFrame([res], columns=['AUC','Accuracy','Precision','Recall','f1-score'], index=['Linear Regression'])

    return res2
    
# Lasso Regression
def LassoKFold(alpha_value=1.0, k=k):
    from sklearn.linear_model import Lasso

    print('')
    print('■ Lasso Regression model')
    print('')

    # Assuming 'path_in' is a variable holding the path to your dataset
    # df = pd.read_csv(str(path_in))
    x = df.iloc[:, :-1].values
    t = df.iloc[:, -1].values

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    lasso = Lasso(alpha=alpha_value)

    feature_importances = np.zeros((x.shape[1],))

    aucs = []
    r2s = []
    accs = []
    precs = []
    recs = []
    f1s = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        t_train, t_test = t[train_index], t[test_index]

        lasso.fit(x_train, t_train)
        predictions = lasso.predict(x_test)

        # Collect feature importance
        feature_importances += np.abs(lasso.coef_)

        # Rounding predictions for classification metrics, assuming binary classification
        predictions_round = [1 if p >= 0.5 else 0 for p in predictions]

        auc = roc_auc_score(t_test, predictions)
        # r2 = r2_score(t_test, predictions)
        acc = accuracy_score(t_test, predictions_round)
        prec = precision_score(t_test, predictions_round, zero_division=0)
        rec = recall_score(t_test, predictions_round)
        f1 = f1_score(t_test, predictions_round)

        aucs.append(auc)
        # r2s.append(r2)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    # Calculate mean scores
    mean_auc = np.mean(aucs)
    # mean_r2 = np.mean(r2s)
    mean_acc = np.mean(accs)
    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)
    mean_f1 = np.mean(f1s)


    # Calculate the average feature importance over all k-folds
    feature_importances /= k
    # Get the feature names from the DataFrame columns
    feature_names = df.columns[:-1]
    # Sort the features by importance
    sorted_idx = np.argsort(feature_importances)[-10:]
    sorted_importance = feature_importances[sorted_idx]
    sorted_features = feature_names[sorted_idx]


    # Export feature importances
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
    excel_filename = os.path.join(path_out, 'feature_importances_lasso_kfold.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)

    # Create a series to hold feature importances with their corresponding feature names
    # feature_importance_series = pd.Series(feature_importances, index=feature_names)
    # Sort the feature importances and take the top 10
    # sorted_importance = feature_importance_series.sort_values(ascending=False)[:10]
    # sorted_importance_all = feature_importance_series.sort_values(ascending=False)
    # os.makedirs(path_out, exist_ok=True)
    # excel_filename = os.path.join(path_out, 'feature_importances_lasso_kfold.xlsx')
    # sorted_importance_all.to_excel(excel_filename, index=False)
    # print('')
    # print('◆　Coefficients of Lasso model are exported as feature_importances_lasso_kfold.xlsx')
    # print('')

    print(f'□ {k}-fold Cross Validation を用いた Lasso Regression による2値分類の検定結果')
    print(f'    Alpha value = {alpha_value}')
    print('')
    print(f'Mean AUC: {mean_auc:.4f}')
    # print(f'Mean R2: {mean_r2}')
    print(f'Mean Accuracy: {mean_acc:.4f}')
    print(f'Mean Precision: {mean_prec:.4f}')
    print(f'Mean Recall: {mean_rec:.4f}')
    print(f'Mean F1 Score: {mean_f1:.4f}')
    print('')


    print('')
    print(f'◇ {k}-fold Cross Validation で得られた Lasso regression における Feature Importance')
    print('')
    # Plot the feature importances
    plt.figure(figsize=(size_x, size_y))
    plt.title(f'Top 10 Feature Importances in {k}-fold CV of Lasso Regression')
    plt.barh(sorted_features, sorted_importance, color='skyblue')
    plt.xlabel('Average Feature Importance')
    plt.show()
    print('')

    # feature_importances_df.head(10).plot(kind='barh', x='Feature', y='Importance', legend=False)
    # plt.gca().invert_yaxis()  # To plot the highest importance at the top
    # plt.xlabel('Average Absolute Coefficient')
    # plt.show()
    # print('')

    # Retrain the model on the full dataset
    model_full = Lasso(alpha=alpha_value)
    model_full.fit(x, t)

    # Saving the model
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_lasso_model.pkl')
    with open(model_filename, 'wb') as file:
        pickle.dump(model_full, file)

    res = np.array([mean_auc, mean_acc, mean_prec, mean_rec, mean_f1], dtype=object)
    res2=pd.DataFrame([res], columns=['AUC','Accuracy','Precision','Recall','f1-score'], index=[f'Lasso Regression (α={alpha_value:.4f})'])

    return res2




# simple Lasso Regression
def simpleLassoKFold(alpha_value, k=k):
    from sklearn.linear_model import Lasso
    # Assuming 'path_in' is a variable holding the path to your dataset
    # df = pd.read_csv(str(path_in))
    x = df.iloc[:, :-1].values
    t = df.iloc[:, -1].values

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    lasso = Lasso(alpha=alpha_value)

    feature_importance = np.zeros((x.shape[1],))

    aucs = []
    r2s = []
    accs = []
    precs = []
    recs = []
    f1s = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        t_train, t_test = t[train_index], t[test_index]

        lasso.fit(x_train, t_train)
        predictions = lasso.predict(x_test)

        # Rounding predictions for classification metrics, assuming binary classification
        predictions_round = [1 if p >= 0.5 else 0 for p in predictions]

        auc = roc_auc_score(t_test, predictions)
        # r2 = r2_score(t_test, predictions)
        acc = accuracy_score(t_test, predictions_round)
        prec = precision_score(t_test, predictions_round, zero_division=0)
        rec = recall_score(t_test, predictions_round)
        f1 = f1_score(t_test, predictions_round)

        aucs.append(auc)
        # r2s.append(r2)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    # Calculate mean scores
    mean_auc = np.mean(aucs)
    # mean_r2 = np.mean(r2s)
    mean_acc = np.mean(accs)
    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)
    mean_f1 = np.mean(f1s)

    print(f'□ {k}-fold CV と Optuna を用いた Lasso による二値分類の評価結果')
    print('')
    print(f'Mean AUC: {mean_auc:.4f}')
    # print(f'Mean R2: {mean_r2}')
    print(f'Mean Accuracy: {mean_acc:.4f}')
    print(f'Mean Precision: {mean_prec:.4f}')
    print(f'Mean Recall: {mean_rec:.4f}')
    print(f'Mean F1 Score: {mean_f1:.4f}')
    print('')

    res = np.array([mean_auc, mean_acc, mean_prec, mean_rec, mean_f1], dtype=object)
    res2=pd.DataFrame([res], columns=['AUC','Accuracy','Precision','Recall','f1-score'], index=[f'Lasso Regression with Optuna'])

    return res2

# Lasso Optimized with Optuna
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.model_selection import KFold


# def custom_alpha_sampler(trial):
#     # 確率的に低い範囲と高い範囲を選択する
#     if trial.suggest_categorical('alpha_range', ['low', 'high']) == 'low':
#         # 低い範囲でのalphaをより高い確率でサンプリング
#         return trial.suggest_float('alpha_low', 0.0001, 0.01, log=True)
#     else:
#         # 高い範囲でのalphaを低い確率でサンプリング
#         return trial.suggest_float('alpha_high', 0.01, 1, log=True)


def objective(trial, X, y, k=k):
    # Suggest a value for the alpha hyperparameter
    alpha = trial.suggest_float('alpha', 0.0001, 0.9, log=False)  # Using log=True for a wider range exploration

    # Suggest a value for the alpha hyperparameter using the custom sampler
    # alpha = custom_alpha_sampler(trial)

    # KFold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # mse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize the Lasso model with the trial's suggested alpha
        model = Lasso(alpha=alpha, random_state=random_state)
        model.fit(X_train, y_train)

        # Predict and calculate MSE for the current fold
        y_pred = model.predict(X_test)
        #mse = mean_squared_error(y_test, y_pred)
        #mse_scores.append(mse)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

    # Return the average MSE over all folds
    # avg_mse = np.mean(mse_scores)
    # return avg_mse
    avg_r2 = np.mean(r2_scores)
    return avg_r2


def LassoKFoldOptuna(k=k):
    # Load your dataset

    print('')
    print('■ Lasso Regression model (with Optuna)')
    print('')

    # df = pd.read_csv(str(path_in))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Optunaのstudyを作成、TPEサンプラーを指定
    # tpe_sampler = optuna.samplers.TPESampler(seed=random_state)  # TPEサンプラーのインスタンス化
    # Create the Optuna study which aims to minimize the MSE
    study = optuna.create_study(direction='minimize')
    # study = optuna.create_study(direction='minimize', sampler=tpe_sampler)
    optuna.logging.disable_default_handler()
    study.optimize(lambda trial: objective(trial, X, y, k), n_trials=n_trialsA)

    # Best hyperparameters
    best_alpha = study.best_trial.params['alpha']

    print('◇ Hyperparameter Optimization for Lasso with Optuna')
    print(f'Best trial for minimizing MSE:')
    print(f'  Value: {study.best_trial.value}')
    print(f'  Params: ')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')
    print('')

    # Retrain model on the full dataset with the best hyperparameters
    best_model = Lasso(alpha=best_alpha, random_state=random_state)
    best_model.fit(X, y)

    res = simpleLassoKFold(study.best_trial.value, k=k)


    print('')
    print(f'◇ {k}-fold Cross Validation と Optunna で得られた Lasso regresssion の Best Model における Feature Importance')
    print('')
    # Get the feature names from the DataFrame columns
    feature_names = df.columns[:-1]
    # Since Lasso can zero out some coefficients, there might be fewer non-zero features to report
    feature_importances = np.abs(best_model.coef_)
    # Sort the features by importance
    sorted_idx = np.argsort(feature_importances)[-10:]
    sorted_importance = feature_importances[sorted_idx]
    sorted_features = feature_names[sorted_idx]


    # Export feature importances
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
    excel_filename = os.path.join(path_out, 'feature_importances_lasso_kfold_optuna.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)

    # Plot the feature importances
    plt.figure(figsize=(size_x, size_y))
    plt.title(f'Top 10 Feature Importances in {k}-fold CV of Lasso Regression optimized with Optuna')
    plt.barh(sorted_features, sorted_importance, color='skyblue')
    plt.xlabel('Average Feature Importance')
    plt.show()

    # Plot the top 10 feature importances
    # top_features = feature_importances_df.head(10)
    # sns.barplot(x='Importance', y='Feature', data=top_features, orient='h')  # Using seaborn for better default visuals
    # plt.xlabel('Absolute Coefficient')
    # plt.tight_layout()  # This will adjust subplot params so the plot fits into the figure area.
    # plt.show()


    # feature_importances_df.head(10).plot(kind='barh', x='Feature', y='Importance', legend=False)
    # plt.gca().invert_yaxis()  # To plot the highest importance at the top
    # plt.xlabel('Absolute Coefficient')
    # plt.show()


    # Create a series to hold feature importances with their corresponding feature names
    # feature_importance_series = pd.Series(feature_importance, index=feature_names)
    # Sort the feature importances and take the top 10 (or fewer if not available)
    # sorted_importance = feature_importance_series.sort_values(ascending=False)[:10]
    # sorted_importance_all = feature_importance_series.sort_values(ascending=False)
    # excel_filename = os.path.join(path_out, 'feature_importances_lasso_kfold_optuna.xlsx')
    # sorted_importance_all.to_excel(excel_filename, index=False)




    # Path where to save the model
    os.makedirs(path_model, exist_ok=True)  # Ensure the directory exists
    model_filename = os.path.join(path_model, 'kFCV_lasso_optuna.pkl')
    # Save the model
    with open(model_filename, 'wb') as file:
        pickle.dump(best_model, file)

    return res
    
# Ridge Regression
def RidgeKFold(alpha_value=1.0, k=k):
    from sklearn.linear_model import Ridge

    print('')
    print('■ Ridge Regression model')
    print('')

    # Load your dataset
    # df = pd.read_csv(str(path_in))
    x = df.iloc[:, :-1].values
    t = df.iloc[:, -1].values

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    ridge = Ridge(alpha=alpha_value)

    # Initialize the array to store the feature importances
    feature_importances = np.zeros(x.shape[1])

    aucs = []
    r2s = []
    accs = []
    precs = []
    recs = []
    f1s = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        t_train, t_test = t[train_index], t[test_index]

        ridge.fit(x_train, t_train)
        predictions = ridge.predict(x_test)

        # Add the feature importances (absolute values)
        feature_importances += np.abs(ridge.coef_)

        # Rounding predictions for classification metrics, assuming binary classification
        predictions_round = [1 if p >= 0.5 else 0 for p in predictions]

        auc = roc_auc_score(t_test, predictions)
        r2 = r2_score(t_test, predictions)
        acc = accuracy_score(t_test, predictions_round)
        prec = precision_score(t_test, predictions_round, zero_division=0)
        rec = recall_score(t_test, predictions_round)
        f1 = f1_score(t_test, predictions_round)

        aucs.append(auc)
        r2s.append(r2)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    # Calculate mean scores
    mean_auc = np.mean(aucs)
    mean_r2 = np.mean(r2s)
    mean_acc = np.mean(accs)
    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)
    mean_f1 = np.mean(f1s)

    print(f'□ {k}-fold Cross Validation を用いた Ridge Regression による2値分類の検定結果')
    print(f'    Alpha value = {alpha_value}')
    print('')
    print(f'Mean AUC: {mean_auc:.4f}')
    # print(f'Mean R2: {mean_r2}')
    print(f'Mean Accuracy: {mean_acc:.4f}')
    print(f'Mean Precision: {mean_prec:.4f}')
    print(f'Mean Recall: {mean_rec:.4f}')
    print(f'Mean F1 Score: {mean_f1:.4f}')
    print('')

    print('')
    print(f'◇ {k}-fold Cross Validation で得られた Ridge regression における Feature Importance')
    print('')

    # Calculate the mean of the feature importances over all folds
    feature_importances /= k
    # Get the feature names from the DataFrame columns
    feature_names = df.columns[:-1]
    # Sort the features by importance
    sorted_idx = np.argsort(feature_importances)[-10:]
    sorted_importance = feature_importances[sorted_idx]
    sorted_features = feature_names[sorted_idx]


    # Export feature importances
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
    excel_filename = os.path.join(path_out, 'feature_importances_ridge_kfold.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)


    # Create a Series with feature importances
    # feature_importances = pd.Series(feature_importances, index=df.columns[:-1])
    # Get the top 10 most important features
    # top10_features = feature_importances.nlargest(10)
    # feature_importances_sorted = feature_importances.sort_values(ascending=False)
    # excel_filename = os.path.join(path_out, 'feature_importances_ridge_kfold.xlsx')
    # feature_importances_sorted.to_excel(excel_filename, index=False)

    # Plot the top 10 feature importances
    plt.figure(figsize=(size_x, size_y))
    plt.title(f'Top 10 Feature Importances in {k}-fold CV of Ridge Regression')
    plt.barh(sorted_features, sorted_importance, color='skyblue')
    plt.xlabel('Average Feature Importance')
    plt.show()

    # feature_importances_df.head(10).plot(kind='barh', x='Feature', y='Importance', legend=False)
    # plt.gca.invert_yaxis()
    # plt.xlabel('Mean Absolute Coefficient')
    # plt.show()

    # Retrain the model on the full dataset
    model_full = Ridge(alpha=alpha_value)
    model_full.fit(x, t)

    # Saving the model
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_ridge_model.pkl')
    with open(model_filename, 'wb') as file:
        pickle.dump(model_full, file)

    res = np.array([mean_auc, mean_acc, mean_prec, mean_rec, mean_f1], dtype=object)
    res2=pd.DataFrame([res], columns=['AUC','Accuracy','Precision','Recall','f1-score'], index=[f'Ridge Regression (α={alpha_value:.4f})'])

    return res2

# simple Ridge Regression
def simpleRidgeKFold(alpha_value, k=k):
    from sklearn.linear_model import Ridge

    # Load your dataset
    # df = pd.read_csv(str(path_in))
    x = df.iloc[:, :-1].values
    t = df.iloc[:, -1].values

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    ridge = Ridge(alpha=alpha_value)

    # Initialize the array to store the feature importances
    feature_importances = np.zeros(x.shape[1])

    aucs = []
    r2s = []
    accs = []
    precs = []
    recs = []
    f1s = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        t_train, t_test = t[train_index], t[test_index]

        ridge.fit(x_train, t_train)
        predictions = ridge.predict(x_test)

        # Rounding predictions for classification metrics, assuming binary classification
        predictions_round = [1 if p >= 0.5 else 0 for p in predictions]

        auc = roc_auc_score(t_test, predictions)
        r2 = r2_score(t_test, predictions)
        acc = accuracy_score(t_test, predictions_round)
        prec = precision_score(t_test, predictions_round, zero_division=0)
        rec = recall_score(t_test, predictions_round)
        f1 = f1_score(t_test, predictions_round)

        aucs.append(auc)
        r2s.append(r2)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    # Calculate mean scores
    mean_auc = np.mean(aucs)
    mean_r2 = np.mean(r2s)
    mean_acc = np.mean(accs)
    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)
    mean_f1 = np.mean(f1s)

    print(f'□ {k}-fold Cross Validation を用いた Ridge Regression による2値分類の検定結果')
    print('')
    print(f'Mean AUC: {mean_auc:.4f}')
    # print(f'Mean R2: {mean_r2}')
    print(f'Mean Accuracy: {mean_acc:.4f}')
    print(f'Mean Precision: {mean_prec:.4f}')
    print(f'Mean Recall: {mean_rec:.4f}')
    print(f'Mean F1 Score: {mean_f1:.4f}')
    print('')

    res = np.array([mean_auc, mean_acc, mean_prec, mean_rec, mean_f1], dtype=object)
    res2=pd.DataFrame([res], columns=['AUC','Accuracy','Precision','Recall','f1-score'], index=[f'Ridge Regression with Optuna'])

    return res2

# Ridge Optimized with Optuna
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import optuna

def objective(trial, X, y, k=k):
    # Suggest a value for the alpha hyperparameter
    alpha = trial.suggest_float('alpha', 0.0001, 10.0, log=True)  # Using log=True for a wider range exploration

    # KFold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    mse_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize the Ridge model with the trial's suggested alpha
        model = Ridge(alpha=alpha, random_state=random_state)
        model.fit(X_train, y_train)

        # Predict and calculate MSE for the current fold
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    # Return the average MSE over all folds
    avg_mse = np.mean(mse_scores)
    return avg_mse

def RidgeKFoldOptuna(k=k):

    print('')
    print('■ Ridge Regression model (with Optuna)')
    print('')
    # Load your dataset
    # df = pd.read_csv(str(path_in))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Create the Optuna study which aims to minimize the MSE
    study = optuna.create_study(direction='minimize')
    optuna.logging.disable_default_handler()
    study.optimize(lambda trial: objective(trial, X, y, k), n_trials=n_trialsA)

    # Best hyperparameters
    best_alpha = study.best_trial.params['alpha']

    print('◇ Hypterparameter Optimization for Ridge with Optuna')
    print(f'Best trial for minimizing MSE:')
    print(f'  Value: {study.best_trial.value}')
    print(f'  Params: ')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')
    print('')

    # Retrain model on the full dataset with the best hyperparameters
    best_model = Ridge(alpha=best_alpha, random_state=random_state)
    best_model.fit(X, y)

    res = simpleRidgeKFold(best_alpha, k=k)

    print('')
    print(f'◇ {k}-fold Cross Validation と Optunna で得られた Ridge regression の Best Model における Feature Importance')
    print('')
    # Get the feature names from the DataFrame columns
    feature_names = df.columns[:-1]
    # Collect feature importances from the best model
    feature_importances = np.abs(best_model.coef_)
    # Sort the features by importance
    sorted_idx = np.argsort(feature_importances)[-10:]
    sorted_importance = feature_importances[sorted_idx]
    sorted_features = feature_names[sorted_idx]

    # Export feature importances
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
    excel_filename = os.path.join(path_out, 'feature_importances_lasso_kfold.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)

    # Create a series to hold feature importances with their corresponding feature names
    # feature_importance_series = pd.Series(feature_importance, index=feature_names)
    # Sort the feature importances and take the top 10
    # sorted_importance = feature_importance_series.sort_values(ascending=False)[:10]
    # sorted_importance_all = feature_importance_series.sort_values(ascending=False)
    # excel_filename = os.path.join(path_out, 'feature_importances_ridge_kfold_optuna.xlsx')
    # sorted_importance_all.to_excel(excel_filename, index=False)

    # Plot the feature importances
    plt.figure(figsize=(size_x, size_y))
    plt.title(f'Top 10 Feature Importances in {k}-fold CV of Ridge Regression optimized with Optuna')
    plt.barh(sorted_features, sorted_importance, color='skyblue')
    plt.xlabel('Average Feature Importance')
    plt.show()


    # feature_importances_df.head(10).plot(kind='barh', x='Feature', y='Importance', legend=False)
    # plt.gca().invert_yaxis()  # To plot the highest importance at the top
    # plt.xlabel('Absolute Coefficient')
    # plt.show()


    # Path where to save the model
    os.makedirs(path_model, exist_ok=True)  # Ensure the directory exists
    model_filename = os.path.join(path_model, 'kFCV_ridge_optuna.pkl')
    # Save the model
    with open(model_filename, 'wb') as file:
        pickle.dump(best_model, file)

    return res


# Logistic Normalized
def LogisticKFoldNormalized(k=k):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import MinMaxScaler

    print('')
    print('■ Logistic Regression model (normalized)')
    print('')

    # Load your dataset
    # df = pd.read_csv(str(path_in))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # KFold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    auc_scores, accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], [], []

    # Initialize array to store feature importances
    feature_importances = np.zeros((df.shape[1] - 1,))

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize the Logistic Regression model
        model = LogisticRegression(random_state=random_state)
        model.fit(X_train, y_train)

        # Add up the absolute values of the coefficients for each feature
        feature_importances += np.abs(model.coef_[0])

        # Predict classes and probabilities for evaluation
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Compute metrics for the current fold
        auc = roc_auc_score(y_test, y_pred_prob)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Append scores
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # Calculate mean scores
        mean_auc = np.mean(auc_scores)
        mean_acc = np.mean(accuracy_scores)
        mean_prec = np.mean(precision_scores)
        mean_rec = np.mean(recall_scores)
        mean_f1 = np.mean(f1_scores)

    # Calculate and print the mean of each metric
    print(f'□ {k}-fold Cross Validation を用いた Logistic Regression with Normalization による2値分類の検定結果')
    print('')
    print(f'Mean AUC: {np.mean(auc_scores):.4f}')
    print(f'Mean Accuracy: {np.mean(accuracy_scores):.4f}')
    print(f'Mean Precision: {np.mean(precision_scores):.4f}')
    print(f'Mean Recall: {np.mean(recall_scores):.4f}')
    print(f'Mean F1 Score: {np.mean(f1_scores):.4f}')
    print('')


    # Average the feature importances over all folds
    feature_importances /= k
    # Get the feature names
    feature_names = df.columns[:-1]
    # Sort the features by importance
    sorted_idx = np.argsort(feature_importances)[-10:]
    sorted_importance = feature_importances[sorted_idx]
    sorted_features = feature_names[sorted_idx]

    # Export feature importances
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
    excel_filename = os.path.join(path_out, 'feature_importances_logistic_normalized_kfold.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)


    print('')
    print(f'◇ {k}-fold Cross Validation で得られた Logistic regression normalized における Feature Importance')
    print('')
    # Plot
    plt.figure(figsize=(size_x, size_y))
    plt.barh(sorted_features, sorted_importance, color='skyblue')
    plt.xlabel('Average Feature Importance')
    plt.title(f'Top 10 Feature Importances in {k}-fold CV of Logistic Regression normalized')
    plt.show()


    # Optionally retrain the model on the full dataset
    model_full = LogisticRegression(random_state=random_state)
    model_full.fit(X_scaled, y)

    # Ensure the path exists
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_logistic_normalized.pkl')
    # Save the full model
    with open(model_filename, 'wb') as file:
        pickle.dump(model_full, file)

    res = np.array([mean_auc, mean_acc, mean_prec, mean_rec, mean_f1], dtype=object)
    res2=pd.DataFrame([res], columns=['AUC','Accuracy','Precision','Recall','f1-score'], index=['Logistic Regression normalized'])

    return res2





# Logistic Regression with standadization
def LogisticKFoldStandardized(k=k):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print('')
    print('■ Logistic Regression model (standardized)')
    print('')

    # Load your dataset
    # df = pd.read_csv(str(path_in))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Initialize StandardScaler
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # KFold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Initialize array to store feature importances
    feature_importances = np.zeros((df.shape[1] - 1,))

    auc_scores, accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], [], []

    for train_index, test_index in kf.split(X_standardized):
        X_train, X_test = X_standardized[train_index], X_standardized[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize the Logistic Regression model
        model = LogisticRegression(random_state=random_state)
        model.fit(X_train, y_train)

        # Add up the absolute values of the coefficients for each feature
        feature_importances += np.abs(model.coef_[0])

        # Predict classes and probabilities for evaluation
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Compute metrics for the current fold
        auc = roc_auc_score(y_test, y_pred_prob)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Append scores
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # Calculate mean scores
        mean_auc = np.mean(auc_scores)
        mean_acc = np.mean(accuracy_scores)
        mean_prec = np.mean(precision_scores)
        mean_rec = np.mean(recall_scores)
        mean_f1 = np.mean(f1_scores)

    # Calculate and print the mean of each metric
    print(f'□ {k}-fold Cross Validation を用いた Logistic Regression with Standardization による2値分類の検定結果')
    print('')
    print(f'Mean AUC: {np.mean(auc_scores):.4f}')
    print(f'Mean Accuracy: {np.mean(accuracy_scores):.4f}')
    print(f'Mean Precision: {np.mean(precision_scores):.4f}')
    print(f'Mean Recall: {np.mean(recall_scores):.4f}')
    print(f'Mean F1 Score: {np.mean(f1_scores):.4f}')
    print('')

    # Average the feature importances over all folds
    feature_importances /= k
    # Get the feature names
    feature_names = df.columns[:-1]
    # Sort the features by importance
    sorted_idx = np.argsort(feature_importances)[-10:]
    sorted_importance = feature_importances[sorted_idx]
    sorted_features = feature_names[sorted_idx]

    # Export feature importances
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
    excel_filename = os.path.join(path_out, 'feature_importances_logistic_standardized_kfold.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)

    print('')
    print(f'◇ {k}-fold Cross Validation で得られた Logistic regression standardized における Feature Importance')
    print('')
    # Plot
    plt.figure(figsize=(size_x, size_y))
    plt.barh(sorted_features, sorted_importance, color='skyblue')
    plt.xlabel('Average Feature Importance')
    plt.title(f'Top 10 Feature Importances in {k}-fold CV of Logistic Regression standardized')
    plt.show()

    # Optionally retrain the model on the full dataset
    model_full = LogisticRegression(random_state=random_state)
    model_full.fit(X_standardized, y)

    # Ensure the path exists
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_logistic_standardized.pkl')
    # Save the full model
    with open(model_filename, 'wb') as file:
        pickle.dump(model_full, file)

    res = np.array([mean_auc, mean_acc, mean_prec, mean_rec, mean_f1], dtype=object)
    res2=pd.DataFrame([res], columns=['AUC','Accuracy','Precision','Recall','f1-score'], index=['Logistic Regression standardized'])

    return res2
    
# SVM Normalized
def SVMKFoldNormalized(k=k):
    from sklearn.svm import SVC
    from sklearn.preprocessing import MinMaxScaler

    print('')
    print('■ Support Vector Machine (normalized)')
    print('')

    # Assuming df is already loaded and available
    # df = pd.read_csv(str(path_in))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    kernel_option = 'linear' if linear ==1 else 'rbf'

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # KFold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Initialize array to store feature importances
    feature_importances = np.zeros((df.shape[1] - 1,))

    auc_scores, accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], [], []


    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize the SVM model with probability estimation enabled
        model = SVC(kernel=kernel_option, probability=True, random_state=random_state)
        model.fit(X_train, y_train)

        # Assuming a linear SVM, extract the coefficients
        # For non-linear SVM, this part is not applicable
        if isinstance(model, SVC) and model.kernel == 'linear':
            feature_importances += np.abs(model.coef_[0])

        # Predict classes and probabilities for evaluation
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Compute metrics for the current fold
        auc = roc_auc_score(y_test, y_pred_prob)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Append scores
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Calculate and print the mean of each metric
    print(f'□ {k}-fold Cross Validation を用いた SVM with Normalization による2値分類の検定結果')
    print(f'     The kernel used: {kernel_option}')
    print('')
    print(f'Mean AUC: {np.mean(auc_scores):.4f}')
    print(f'Mean Accuracy: {np.mean(accuracy_scores):.4f}')
    print(f'Mean Precision: {np.mean(precision_scores):.4f}')
    print(f'Mean Recall: {np.mean(recall_scores):.4f}')
    print(f'Mean F1 Score: {np.mean(f1_scores):.4f}')
    print('')

    # Feature importance for SVM can be calculated only when the kernel is linear.
    if kernel_option == 'linear':
        # Average the feature importances over all folds
        feature_importances /= k
        # Get the feature names
        feature_names = df.columns[:-1]
        # Sort the features by importance
        sorted_idx = np.argsort(feature_importances)[-10:]
        sorted_importance = feature_importances[sorted_idx]
        sorted_features = feature_names[sorted_idx]

        # Export feature importances
        feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        # Sorting by absolute values for export
        feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
        feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
        excel_filename = os.path.join(path_out, 'feature_importances_svm_normalized_kfold.xlsx')
        feature_importances_df.to_excel(excel_filename, index=False)

        print('')
        print(f'◇ {k}-fold Cross Validation で得られた SVM normazlied における Feature Importance')
        print('')
        # Plot
        plt.figure(figsize=(size_x, size_y))
        plt.barh(sorted_features, sorted_importance, color='skyblue')
        plt.xlabel('Average Feature Importance')
        plt.title(f'Top 10 Feature Importances in {k}-foldCV of SVM normalized using linear kernel')
        plt.show()
        print('')
    else:
        print('')
        print('・ Feature importance cannot be figured out from SVM with NON-LINEAR kernel due to the kernel trick.')
        print('')


    # Initialize the SVM model with probability estimation enabled
    model_full = SVC(kernel=kernel_option, probability=True, random_state=random_state)
    # Fit the model to the full normalized dataset
    model_full.fit(X_scaled, y)

    # Ensure the path exists
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_svm_normalized.pkl')
    # Save the full model
    with open(model_filename, 'wb') as file:
        pickle.dump(model_full, file)

    res = np.array([np.mean(auc_scores), np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)], dtype=object)
    res2 = pd.DataFrame([res], columns=['AUC','Accuracy','Precision','Recall','f1-score'], index=['SVM normalized'])

    return res2

# SVM Standardized
def SVMKFoldStandardized(k=k):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    import numpy as np
    import pandas as pd

    print('')
    print('■ Support Vector Machine (standardized)')
    print('')

    # Assume df is your DataFrame loaded previously
    # df = pd.read_csv(str(path_in))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    kernel_option = 'linear' if linear ==1 else 'rbf'

    # Initialize StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KFold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Initialize array to store feature importances for a linear kernel
    feature_importances = np.zeros((X.shape[1],))

    auc_scores, accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], [], []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize the SVM model
        model = SVC(kernel = kernel_option, probability=True, random_state=random_state)
        model.fit(X_train, y_train)

        # Assuming a linear SVM, extract the coefficients
        # For non-linear SVM, this part is not applicable
        if isinstance(model, SVC) and model.kernel == 'linear':
            feature_importances += np.abs(model.coef_[0])

        # Predict classes and probabilities for evaluation
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Compute metrics for the current fold
        auc = roc_auc_score(y_test, y_pred_prob)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Append scores
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)


    # Calculate and print the mean of each metric
    print(f'□ {k}-fold Cross Validation を用いた SVM with Standardization による2値分類の検定結果')
    print(f'     The kernel used: {kernel_option}')
    print('')
    print(f'Mean AUC: {np.mean(auc_scores):.4f}')
    print(f'Mean Accuracy: {np.mean(accuracy_scores):.4f}')
    print(f'Mean Precision: {np.mean(precision_scores):.4f}')
    print(f'Mean Recall: {np.mean(recall_scores):.4f}')
    print(f'Mean F1 Score: {np.mean(f1_scores):.4f}')
    print('')


    # Feature importance for SVM can be calculated only when the kernel is linear.
    if kernel_option == 'linear':
        # Average the feature importances over all folds
        feature_importances /= k
        # Get the feature names
        feature_names = df.columns[:-1]
        # Sort the features by importance
        sorted_idx = np.argsort(feature_importances)[-10:]
        sorted_importance = feature_importances[sorted_idx]
        sorted_features = feature_names[sorted_idx]

        # Export feature importances
        feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        # Sorting by absolute values for export
        feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
        feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
        excel_filename = os.path.join(path_out, 'feature_importances_svm_standardized_kfold.xlsx')
        feature_importances_df.to_excel(excel_filename, index=False)

        print('')
        print(f'◇ {k}-fold Cross Validation で得られた SVM standardized における Feature Importance')
        print('')
        # Plot
        plt.figure(figsize=(size_x, size_y))
        plt.barh(sorted_features, sorted_importance, color='skyblue')
        plt.xlabel('Average Feature Importance')
        plt.title(f'Top 10 Feature Importances in {k}-fold CV of SVM standardized using linear kernel')
        plt.show()
        print('')
    else:
        print('')
        print('・ Feature importance cannot be figured out from SVM with NON-LINEAR kernel due to the kernel trick.')
        print('')


    # Initialize the SVM model with probability estimation enabled
    model_full = SVC(kernel=kernel_option, probability=True, random_state=random_state)
    # Fit the model to the full normalized dataset
    model_full.fit(X_scaled, y)

    # Ensure the path exists
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_svm_standardized.pkl')

    # Save the full model
    with open(model_filename, 'wb') as file:
        pickle.dump(model_full, file)

    res = np.array([np.mean(auc_scores), np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)], dtype=object)
    res2 = pd.DataFrame([res], columns=['AUC', 'Accuracy', 'Precision', 'Recall', 'f1-score'], index=['SVM standardized'])

    return res2
    
# RandomForest
def RandomForestKFold(k=k):
    from sklearn.ensemble import RandomForestClassifier

    print('')
    print('■ Random Forest')
    print('')

    # Assume df is your DataFrame loaded previously
    # df = pd.read_csv(str(path_in))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values

    # KFold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    auc_scores, accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], [], []
    feature_importance_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize the RandomForest model with default parameters
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)

        # Collect feature importances
        feature_importances = model.feature_importances_
        feature_importance_list.append(feature_importances)

        # Predict classes and probabilities for evaluation
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Compute metrics for the current fold
        auc = roc_auc_score(y_test, y_pred_prob)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Append scores
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Calculate the average feature importance across all folds
    mean_feature_importances = np.mean(feature_importance_list, axis=0)

    column_names = df.columns[:-1]
    feature_importances_df = pd.DataFrame({'Feature': column_names, 'Importance': mean_feature_importances})
    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
    excel_filename = os.path.join(path_out, 'feature_importances_random_forest_kfold.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)


    # Get the top 10 features and their importances
    indices = np.argsort(mean_feature_importances)[-10:]
    top_features = X.columns[indices]
    top_importances = mean_feature_importances[indices]

    # Calculate and print the mean of each metric
    print(f'□ {k}-fold Cross Validation Results with Default Hyperparameters for Random Forest Binary Classification')
    print('')
    print(f'Mean AUC: {np.mean(auc_scores):.4f}')
    print(f'Mean Accuracy: {np.mean(accuracy_scores):.4f}')
    print(f'Mean Precision: {np.mean(precision_scores):.4f}')
    print(f'Mean Recall: {np.mean(recall_scores):.4f}')
    print(f'Mean F1 Score: {np.mean(f1_scores):.4f}')
    print('')

    print('')
    print(f'◇ {k}-fold Cross Validation を用いた Random Forest による Feature Importance')
    print('')

    # Plot the top 10 feature importances
    plt.figure(figsize=(size_x, size_y))
    plt.barh(range(len(indices)), top_importances, color='b', align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title(f'Top 10 Feature Importances in {k}-fold CV of Random Forest')
    plt.show()

    # Optionally retrain the model on the full dataset
    model_full = RandomForestClassifier(random_state=random_state)
    model_full.fit(X, y)

    # Ensure the path exists
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_random_forest.pkl')

    # Save the full model
    with open(model_filename, 'wb') as file:
        pickle.dump(model_full, file)

    res = np.array([np.mean(auc_scores), np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)], dtype=object)
    res2 = pd.DataFrame([res], columns=['AUC', 'Accuracy', 'Precision', 'Recall', 'f1-score'], index=['Random Forest'])

    return res2



def RandomForestOptunaKFold(k=k):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    print('')
    print('■ Random Forest (with Optuna)')
    print('')

    # Assume df is loaded as before
    # df = pd.read_csv(str(path_in))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    def objective(trial):
        # Hyperparameter space definition
        n_estimators = trial.suggest_int('n_estimators', 10, 500)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 14)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 14)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

        # Model setup with suggested hyperparameters
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )

        # KFold cross-validation setup
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

        # Cross-validation with ROC AUC as the metric to maximize
        auc_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')

        # Objective: Maximize mean AUC
        return np.mean(auc_scores)

    # Create an Optuna study object
    study = optuna.create_study(direction='maximize')
    optuna.logging.disable_default_handler()
    study.optimize(objective, n_trials=n_trialsB)  # Adjust n_trials for your computational budget

    # Best hyperparameters found
    print('◇ Hyperparameter Optimization for Ranom Forest with Optuna')
    print("Best hyperparameters: ", study.best_params)

    # Train and evaluate a model with the best hyperparameters
    best_model = RandomForestClassifier(
        n_estimators=study.best_params['n_estimators'],
        max_depth=study.best_params['max_depth'],
        min_samples_split=study.best_params['min_samples_split'],
        min_samples_leaf=study.best_params['min_samples_leaf'],
        max_features=study.best_params['max_features'],
        random_state=random_state,
    )

    # Train and evaluate a model with the best hyperparameters
    best_model_params = {
        'n_estimators': study.best_params['n_estimators'],
        'max_depth': study.best_params['max_depth'],
        'min_samples_split': study.best_params['min_samples_split'],
        'min_samples_leaf': study.best_params['min_samples_leaf'],
        'max_features': study.best_params['max_features'],
        'random_state': random_state,
    }

    # Initialize an array to store feature importances across folds
    feature_importances = np.zeros(len(df.columns) - 1)

    # Fit the best model to the full dataset
    best_model.fit(X, y)

    # You could use cross-validation again here with the best model for evaluation
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        fold_model = RandomForestClassifier(**best_model_params)
        fold_model.fit(X_train, y_train)

        # Accumulate feature importances
        feature_importances += fold_model.feature_importances_

    # Average feature importances across all folds
    feature_importances /= k

    column_names = df.columns[:-1]
    feature_importances_df = pd.DataFrame({'Feature': column_names, 'Importance': feature_importances})

    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)

    excel_filename = os.path.join(path_out, 'feature_importances_random_forest_kfold_optuna.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)

    # Additional metrics
    scores = cross_val_score(best_model, X, y, cv=kf, scoring='roc_auc')
    # Display final evaluation
    print(f"Final Evaluation with Best Hyperparameters (ROC_AUC): Mean={scores.mean():.4f}, Std={scores.std():.4f}")

    # If multiple metrics evaluation is needed
    from sklearn.model_selection import cross_validate

    scoring_metrics = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
    final_scores = cross_validate(best_model, X, y, cv=kf, scoring=scoring_metrics)
    print('')
    print(f'□ {k}-fold Cross Validation と Optuna による最適化を行った RandomForest　による2値分類の検定結果')
    print('')
    res = []
    for metric in scoring_metrics:
        metric_scores = final_scores[f'test_{metric}']
        res.append(metric_scores.mean())
        print(f"{metric}: Mean={metric_scores.mean():.4f}, Std={metric_scores.std():.4f}")

    print('')
    print('')
    print(f'◇ {k}-fold Cross Validation と Optuna で得られた Random Forest の Best Model による Feature Importance')
    print('')

    plt.figure(figsize=(size_x, size_y))  # Adjust the size as needed
    plt.title(f'Top 10 Feature Importances in {k}-fold CV of Random Forest optimized with Optuna')
    plt.barh(feature_importances_df['Feature'][:10], feature_importances_df['Importance'][:10], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.gca().invert_yaxis()  # Invert the y-axis to have the most important feature on top
    plt.show()


    # Ensure the directory exists
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_random_forest_optuna.pkl')
    # Save the best model
    with open(model_filename, 'wb') as file:
        pickle.dump(best_model, file)


    res2 = pd.DataFrame([res], columns=['AUC', 'Accuracy', 'Precision', 'Recall', 'f1-score'], index=['Random Forest with Optuna'])

    return res2

# How to evaluate a model with the best parameters
# best_model = RandomForestClassifier(
#     **study.best_params,
#       random_state=random_state,
#.      )


# XGBoost
def XGBoostKFold(k=k):
    from xgboost import XGBClassifier

    print('')
    print('■ XGBoost')
    print('')

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]

    # booster_option = 'gblinear' if linear ==1 else 'gbtree'
    booster_option = 'gbtree'

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    auc_scores, accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], [], []
    feature_importances = defaultdict(list)
    feature_importances_weight = defaultdict(list)
    feature_importances_cover = defaultdict(list)
    feature_importances_gain = defaultdict(list)


    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize the XGBoost model with default parameters
        model = XGBClassifier(booster=booster_option, use_label_encoder=False, eval_metric='logloss', random_state=random_state)
        model.fit(X_train, y_train)

        booster = model.get_booster()
        importance_weight = booster.get_score(importance_type='weight')
        importance_cover = booster.get_score(importance_type='cover')
        importance_gain = booster.get_score(importance_type='gain')

        for feature_name in feature_names:
            feature_importances_weight[feature_name].append(importance_weight.get(feature_name, 0))
            feature_importances_cover[feature_name].append(importance_cover.get(feature_name, 0))
            feature_importances_gain[feature_name].append(importance_gain.get(feature_name, 0))


        # Predict classes and probabilities for evaluation
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Compute metrics for the current fold
        auc = roc_auc_score(y_test, y_pred_prob)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Append scores
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # Getting feature importances
        for i, importance in enumerate(model.feature_importances_):
            feature_importances[feature_names[i]].append(importance)

    # Calculate and print the mean of each metric
    print(f'□ {k}-fold Cross Validation を用いた DEFAULT の hyperparameters での XGBoost による2値分類の検定結果')
    print(f'     The booster used: {booster_option}')
    print('')
    print(f'Mean AUC: {np.mean(auc_scores):.4f}')
    print(f'Mean Accuracy: {np.mean(accuracy_scores):.4f}')
    print(f'Mean Precision: {np.mean(precision_scores):.4f}')
    print(f'Mean Recall: {np.mean(recall_scores):.4f}')
    print(f'Mean F1 Score: {np.mean(f1_scores):.4f}')
    print('')

    # Average Importance
    avg_importance = {feature: np.mean(importances) for feature, importances in feature_importances.items()}
    sorted_avg_importance = dict(sorted(avg_importance.items(), key=lambda item: item[1], reverse=True))

    column_names = list(sorted_avg_importance.keys())
    feature_values = list(sorted_avg_importance.values())
    feature_importances_df = pd.DataFrame({'Feature': column_names, 'Importance': feature_values})
    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
    excel_filename = os.path.join(path_out, 'feature_importances_xgboost_kfold.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)


    # Visualization Functions
    def plot_importance(title, importance_dict, color, xlabel):
        plt.figure(figsize=(size_x, size_y))
        plt.title(title)
        keys = list(importance_dict.keys())[:10]
        values = list(importance_dict.values())[:10]
        plt.barh(keys, values, color=color)
        plt.xlabel(xlabel)
        plt.gca().invert_yaxis()
        plt.show()

    print('')
    print(f'◇ {k}-fold Cross Validation を用いた XGBoost による Feature Importance')
    print('')

    # Plot for Average Importance
    plot_importance(f"Top 10 Feature Importances in {k}-fold CV of XGBoost",  dict(list(sorted_avg_importance.items())[:10]), 'skyblue', "Average Importance")


    # Retrain the model on the full dataset
    model_full = XGBClassifier(booster=booster_option, use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    model_full.fit(X, y)
    # Ensure the directory exists
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_xgboost.pkl')
    # Save the best model
    with open(model_filename, 'wb') as file:
        pickle.dump(model_full, file)


    res = np.array([np.mean(auc_scores), np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)], dtype=object)
    res2 = pd.DataFrame([res], columns=['AUC', 'Accuracy', 'Precision', 'Recall', 'f1-score'], index=['XGBoost'])

    return res2



def XGBoostOptunaKFold(k=k):

    import xgboost as xgb
    import optuna

    print('')
    print('■　XGBoost (with Optuna)')
    print('')

    # Assuming df is your DataFrame loaded previously
    # df = pd.read_csv(str(path_in))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]

    # booster_option = 'gblinear' if linear ==1 else 'gbtree'
    booster_option = 'gbtree'

    def objective(trial):
        # Define the hyperparameter space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        }

        # Initialize and train the model
        model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', random_state=random_state, booster=booster_option)

        # KFold cross-validation
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

        auc_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)

            y_pred_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_prob)
            auc_scores.append(auc)

        return np.mean(auc_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trialsC)  # Adjust n_trials according to your computational resources


    print("◇ Best hyperparameters: ", study.best_params)


    # Final KFold cross-validation with the best model
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    final_scores = {'roc_auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    # Model evaluation with the best parameters
    feature_importances = defaultdict(list)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # best_params = study.best_params
        best_model = xgb.XGBClassifier(**study.best_params, booster=booster_option, use_label_encoder=False, eval_metric='logloss', random_state=random_state)
        best_model.fit(X_train, y_train)

        # Predict classes and probabilities for evaluation
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]

        # Compute metrics for the current fold
        final_scores['roc_auc'].append(roc_auc_score(y_test, y_pred_prob))
        final_scores['accuracy'].append(accuracy_score(y_test, y_pred))
        final_scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        final_scores['recall'].append(recall_score(y_test, y_pred))
        final_scores['f1'].append(f1_score(y_test, y_pred))

        # Getting feature importances
        for i, importance in enumerate(best_model.feature_importances_):
            feature_importances[feature_names[i]].append(importance)


    # Calculate and print the mean of each metric
    print('')
    print(f'□ {k}-fold Cross Validation を用いた XGBoost optimized with Optuna による2値分類の検定結果')
    print(f'     The booster used: {booster_option}')
    print('')
    for metric, scores in final_scores.items():
        print(f"{metric.capitalize()}: Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")
    print('')
    print('')
    print(f'◇ {k}-fold Cross Validation と Optunna で得られた XGBoost の Best Model における Feature Importance')
    print('')

    # Average Importance
    avg_importance = {feature: np.mean(importances) for feature, importances in feature_importances.items()}
    sorted_avg_importance = dict(sorted(avg_importance.items(), key=lambda item: item[1], reverse=True))

    column_names = list(sorted_avg_importance.keys())
    feature_values = list(sorted_avg_importance.values())
    feature_importances_df = pd.DataFrame({'Feature': column_names, 'Importance': feature_values})
    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
    excel_filename = os.path.join(path_out, 'feature_importances_xgboost_kfold_optuna.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)

    # Visualization Functions
    def plot_importance(title, importance_dict, color, xlabel):
        plt.figure(figsize=(size_x, size_y))
        plt.title(title)
        keys = list(importance_dict.keys())[:10]
        values = list(importance_dict.values())[:10]
        plt.barh(keys, values, color=color)
        plt.xlabel(xlabel)
        plt.gca().invert_yaxis()
        plt.show()

    # Plot for Average Importance
    plot_importance(f"Top 10 Feature Importances in {k}-fold CV of XGBoost optimized with Optuna",  dict(list(sorted_avg_importance.items())[:10]), 'skyblue', "Average Importance")


    # Train best model with full data
    best_model_full = xgb.XGBClassifier(**study.best_params, booster=booster_option, use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    best_model_full.fit(X, y)
    # Ensure the directory exists
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_xgboost_optuna.pkl')
    # Save the best model
    with open(model_filename, 'wb') as file:
        pickle.dump(best_model_full, file)


    res = [np.mean(scores) for scores in final_scores.values()]
    res2 = pd.DataFrame([res], columns=['AUC', 'Accuracy', 'Precision', 'Recall', 'f1-score'], index=['XGBoost with Optuna'])

    return res2

# LightGBM
def LightGBMKFold(k=k):
    from lightgbm import LGBMClassifier

    print('')
    print('■ LightGBM')
    print('')

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    auc_scores, accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], [], []
    feature_importances = defaultdict(list)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize the LightGBM model with default parameters
        model = LGBMClassifier(boosting_type='gbdt', random_state=random_state, verbose = -1)
        model.fit(X_train, y_train)

        # Predict classes and probabilities for evaluation
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Compute metrics for the current fold
        auc = roc_auc_score(y_test, y_pred_prob)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Append scores
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # Getting feature importances
        for i, importance in enumerate(model.feature_importances_):
            feature_importances[feature_names[i]].append(importance)

    # Calculate and print the mean of each metric
    print(f'□ {k}-fold Cross Validation を用いた DEFAULT の hyperparameters での LightGBM による2値分類の検定結果')
    print('')
    print(f'Mean AUC: {np.mean(auc_scores):.4f}')
    print(f'Mean Accuracy: {np.mean(accuracy_scores):.4f}')
    print(f'Mean Precision: {np.mean(precision_scores):.4f}')
    print(f'Mean Recall: {np.mean(recall_scores):.4f}')
    print(f'Mean F1 Score: {np.mean(f1_scores):.4f}')
    print('')

    # Average Importance
    avg_importance = {feature: np.mean(importances) for feature, importances in feature_importances.items()}
    sorted_avg_importance = dict(sorted(avg_importance.items(), key=lambda item: item[1], reverse=True))

    column_names = list(sorted_avg_importance.keys())
    feature_values = list(sorted_avg_importance.values())
    feature_importances_df = pd.DataFrame({'Feature': column_names, 'Importance': feature_values})
    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
    excel_filename = os.path.join(path_out, 'feature_importances_lightgbm_kfold.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)

    # print('')
    # print(f'◇ {k}-fold Cross Validation を用いた LightGBM による Feature Importance')
    # print('')

    # Visualization Functions
    def plot_importance(title, importance_dict, color, xlabel):
        plt.figure(figsize=(size_x, size_y))
        plt.title(title)
        keys = list(importance_dict.keys())[:10]
        values = list(importance_dict.values())[:10]
        plt.barh(keys, values, color=color)
        plt.xlabel(xlabel)
        plt.gca().invert_yaxis()
        plt.show()

    print('')
    print(f'◇ {k}-fold Cross Validation を用いた LightGBM による Feature Importance')
    print('')

    # Plot for Average Importance
    plot_importance(f"Top 10 Feature Importances in {k}-fold CV of LightGBM",dict(list(sorted_avg_importance.items())[:10]), 'skyblue', "Average Importance")

    # Retrain the model on the full dataset
    model_full = LGBMClassifier(boosting_type='gbdt', random_state=random_state, verbose = -1)
    model_full.fit(X, y)
    # Ensure the directory exists
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_lightgbm.pkl')
    # Save the best model
    with open(model_filename, 'wb') as file:
        pickle.dump(model_full, file)

    res = np.array([np.mean(auc_scores), np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)], dtype=object)
    res2 = pd.DataFrame([res], columns=['AUC', 'Accuracy', 'Precision', 'Recall', 'f1-score'], index=['LightGBM'])

    return res2



def LightGBMOptunaKFold(k=k):

    import lightgbm as lgb
    from lightgbm import early_stopping
    import optuna

    print('')
    print('■ LightGBM (with Optuna)')
    print('')

    # Assuming df is your DataFrame loaded previously
    # df = pd.read_csv(str(path_in))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]

    def objective(trial):

        # Define the hyperparameter space
        max_depth = trial.suggest_int('max_depth', 3, 9)
        num_leaves = trial.suggest_int('num_leaves', 2 ** max_depth / 2, 2 ** max_depth, log=True)

        # Define the hyperparameter space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        }

        # Initialize and train the model
        model = lgb.LGBMClassifier(**params, random_state=random_state, objective='binary', verbosity=-1)

        # KFold cross-validation
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

        auc_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train, eval_set=[(X_test, y_test)],  callbacks=[early_stopping(stopping_rounds=50)])

            y_pred_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_prob)
            auc_scores.append(auc)

        return np.mean(auc_scores)



    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trialsD)  # Adjust n_trials according to your computational resources
    # study.optimize(objective, n_trials=1)

    print('')
    print("◇ Best hyperparameters: ", study.best_params)


    # Final KFold cross-validation with the best model
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    final_scores = {'roc_auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    # Model evaluation with the best parameters
    feature_importances = defaultdict(list)


    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        best_model = lgb.LGBMClassifier(**study.best_params, random_state=random_state, objective='binary', verbosity=-1)
        best_model.fit(X_train, y_train)

        # Predict classes and probabilities for evaluation
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]

        # Compute metrics for the current fold
        final_scores['roc_auc'].append(roc_auc_score(y_test, y_pred_prob))
        final_scores['accuracy'].append(accuracy_score(y_test, y_pred))
        final_scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        final_scores['recall'].append(recall_score(y_test, y_pred))
        final_scores['f1'].append(f1_score(y_test, y_pred))

        # Getting feature importances
        for i, importance in enumerate(best_model.feature_importances_):
            feature_importances[feature_names[i]].append(importance)

    # Calculate and print the mean of each metric
    print('')
    print(f'□ {k}-fold Cross Validation を用いた LightGBM optimized with Optuna による2値分類の検定結果')
    print('')
    for metric, scores in final_scores.items():
        print(f"{metric.capitalize()}: Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")
    print('')
    print('')
    print(f'◇ {k}-fold Cross Validation と Optunna で得られた LightGBM の Best Model における Feature Importance')
    print('')

    # Average Importance
    avg_importance = {feature: np.mean(importances) for feature, importances in feature_importances.items()}
    sorted_avg_importance = dict(sorted(avg_importance.items(), key=lambda item: item[1], reverse=True))

    column_names = list(sorted_avg_importance.keys())
    feature_values = list(sorted_avg_importance.values())
    feature_importances_df = pd.DataFrame({'Feature': column_names, 'Importance': feature_values})
    # Sorting by absolute values for export
    feature_importances_df['Abs_Importance'] = feature_importances_df['Importance'].abs()
    feature_importances_df = feature_importances_df.sort_values(by='Abs_Importance', ascending=False).drop('Abs_Importance', axis=1)
    excel_filename = os.path.join(path_out, 'feature_importances_lightgbm_kfold_optuna.xlsx')
    feature_importances_df.to_excel(excel_filename, index=False)

    # Visualization Functions
    def plot_importance(title, importance_dict, color, xlabel):
        plt.figure(figsize=(size_x, size_y))
        plt.title(title)
        keys = list(importance_dict.keys())[:10]
        values = list(importance_dict.values())[:10]
        plt.barh(keys, values, color=color)
        plt.xlabel(xlabel)
        plt.gca().invert_yaxis()
        plt.show()

    # Plot for Average Importance
    plot_importance(f"Top 10 Feature Importances in {k}-fold CV of LightGBM optimized with Optuna", dict(list(sorted_avg_importance.items())[:10]), 'skyblue', "Average Importance")

    # Train best model with full data
    best_model_full = lgb.LGBMClassifier(**study.best_params, random_state=random_state, objective='binary', verbosity=-1)
    best_model_full.fit(X, y)
    # Ensure the directory exists
    os.makedirs(path_model, exist_ok=True)
    model_filename = os.path.join(path_model, 'kFCV_lightgbm_optuna.pkl')
    # Save the best model
    with open(model_filename, 'wb') as file:
        pickle.dump(best_model_full, file)

    res = [np.mean(scores) for scores in final_scores.values()]
    res2 = pd.DataFrame([res], columns=['AUC', 'Accuracy', 'Precision', 'Recall', 'f1-score'], index=['LightGBM with Optuna'])

    return res2

    # lassoはoptunaを使わない設定になっている。また、CatBoostは長時間かかるため外してある。
    results = pd.concat([res_linear, res_lasso, res_ridge_optuna, res_logistic_N, res_logistic_S, res_SVM_N, res_SVM_S, res_RandomForest,  res_RandomForest_optuna, res_XGBoost_optuna, res_LightGBM, res_LightGBM_optuna, res_MLPn, res_MLPs, res_Decision_Tree, res_Gradient_Boosting, res_NB, res_kNN_N, res_kNN_S], axis=0)
    results.to_excel(filepath)

    # 計算終了時の現在時刻
    end_time = datetime.now()
    formatted_end_time = end_time.strftime('%Y年%m月%d日 %H時%M分%S秒')
    print(f'◆ Present Time (End): {formatted_end_time}')

    # 計算に要した時間（終了時刻 - 開始時刻）
    elapsed_time = end_time - start_time

    # 時間、分、秒に分割して表示
    hours, remainder = divmod(elapsed_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'Elapsed Time: {hours} hours, {minutes} minutes, {seconds} seconds')
