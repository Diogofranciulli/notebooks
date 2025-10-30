"""
DATA SCIENCE AND STATISTICAL COMPUTING - Sprint 4

Arthur Cotrick Pagani - RM:554510
Diogo Leles Franciulli - RM:558487
Felipe Sousa De Oliveira - RM:559085
Ryan Brito Pereira Ramos - RM:554497
Victor Chave - RM:557067
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             mean_squared_error, r2_score, mean_absolute_error)
import warnings

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("DATA SCIENCE AND STATISTICAL COMPUTING - Sprint 4")
print("=" * 80)


# 1. PROBLEMA DE PREDIÇÃO

print("\n1. PROBLEMA DE PREDIÇÃO")
print("-" * 80)

from sklearn.datasets import make_classification, make_regression

X_class, y_class = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    random_state=42,
    class_sep=1.5
)

X_reg, y_reg = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=15,
    random_state=42
)

print("✓ Dataset de Classificação criado: {} amostras, {} features".format(
    X_class.shape[0], X_class.shape[1]))
print("✓ Dataset de Regressão criado: {} amostras, {} features".format(
    X_reg.shape[0], X_reg.shape[1]))
print("✓ Problema de predição binária definido (ex: churn, falha, ataque crítico)")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42, stratify=y_class)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)


scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)


# 2. KNN (K-Nearest Neighbors)

print("\n2. KNN - K-NEAREST NEIGHBORS")
print("-" * 80)

knn_class = KNeighborsClassifier(n_neighbors=5)
knn_class.fit(X_train_c_scaled, y_train_c)
y_pred_knn_c = knn_class.predict(X_test_c_scaled)

print("KNN Classificação (k=5):")
print(f"  Acurácia: {accuracy_score(y_test_c, y_pred_knn_c):.4f}")
print(f"  Precisão: {precision_score(y_test_c, y_pred_knn_c):.4f}")
print(f"  Recall: {recall_score(y_test_c, y_pred_knn_c):.4f}")
print(f"  F1-Score: {f1_score(y_test_c, y_pred_knn_c):.4f}")

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_r_scaled, y_train_r)
y_pred_knn_r = knn_reg.predict(X_test_r_scaled)

print("\nKNN Regressão (k=5):")
print(f"  MSE: {mean_squared_error(y_test_r, y_pred_knn_r):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_knn_r)):.4f}")
print(f"  R² Score: {r2_score(y_test_r, y_pred_knn_r):.4f}")

# 3. AVALIAÇÃO DO KNN

print("\n3. AVALIAÇÃO DO KNN")
print("-" * 80)

k_values = range(1, 21)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_c_scaled, y_train_c)
    train_scores.append(knn.score(X_train_c_scaled, y_train_c))
    test_scores.append(knn.score(X_test_c_scaled, y_test_c))

best_k = k_values[np.argmax(test_scores)]
print(f"Melhor valor de k: {best_k}")
print(f"Acurácia no treino (k={best_k}): {train_scores[best_k - 1]:.4f}")
print(f"Acurácia no teste (k={best_k}): {test_scores[best_k - 1]:.4f}")

# Validação cruzada

knn_best = KNeighborsClassifier(n_neighbors=best_k)
cv_scores = cross_val_score(knn_best, X_train_c_scaled, y_train_c, cv=5)
print(f"\nValidação Cruzada (5-fold):")
print(f"  Scores: {cv_scores}")
print(f"  Média: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 4. REGRESSÃO LOGÍSTICA

print("\n4. REGRESSÃO LOGÍSTICA")
print("-" * 80)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_c_scaled, y_train_c)
y_pred_logreg = log_reg.predict(X_test_c_scaled)

print("Regressão Logística:")
print(f"  Acurácia: {accuracy_score(y_test_c, y_pred_logreg):.4f}")
print(f"  Precisão: {precision_score(y_test_c, y_pred_logreg):.4f}")
print(f"  Recall: {recall_score(y_test_c, y_pred_logreg):.4f}")
print(f"  F1-Score: {f1_score(y_test_c, y_pred_logreg):.4f}")

print("\nInterpretação dos coeficientes:")
coef_df = pd.DataFrame({
    'Feature': [f'Feature_{i}' for i in range(X_class.shape[1])],
    'Coeficiente': log_reg.coef_[0]
}).sort_values('Coeficiente', key=abs, ascending=False)
print(coef_df.head())

# 5. COMPARAÇÃO DE RESULTADOS

print("\n5. COMPARAÇÃO DE RESULTADOS (KNN vs Regressão Logística)")
print("-" * 80)


knn_best.fit(X_train_c_scaled, y_train_c)
y_pred_knn_best = knn_best.predict(X_test_c_scaled)

comparison = pd.DataFrame({
    'Modelo': ['KNN', 'Regressão Logística'],
    'Acurácia': [
        accuracy_score(y_test_c, y_pred_knn_best),
        accuracy_score(y_test_c, y_pred_logreg)
    ],
    'Precisão': [
        precision_score(y_test_c, y_pred_knn_best),
        precision_score(y_test_c, y_pred_logreg)
    ],
    'Recall': [
        recall_score(y_test_c, y_pred_knn_best),
        recall_score(y_test_c, y_pred_logreg)
    ],
    'F1-Score': [
        f1_score(y_test_c, y_pred_knn_best),
        f1_score(y_test_c, y_pred_logreg)
    ]
})

print(comparison.to_string(index=False))

best_model_idx = comparison['F1-Score'].idxmax()
print(f"\n✓ Melhor modelo: {comparison.iloc[best_model_idx]['Modelo']}")

# 6. RIDGE REGRESSION

print("\n6. RIDGE REGRESSION")
print("-" * 80)

alphas = [0.01, 0.1, 1, 10, 100]
ridge_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_r_scaled, y_train_r)
    y_pred_ridge = ridge.predict(X_test_r_scaled)
    score = r2_score(y_test_r, y_pred_ridge)
    ridge_scores.append(score)
    print(f"Ridge (alpha={alpha:6.2f}): R² = {score:.4f}")

best_alpha_ridge = alphas[np.argmax(ridge_scores)]
print(f"\n✓ Melhor alpha para Ridge: {best_alpha_ridge}")

ridge_final = Ridge(alpha=best_alpha_ridge)
ridge_final.fit(X_train_r_scaled, y_train_r)
y_pred_ridge_final = ridge_final.predict(X_test_r_scaled)

print(f"Ridge Final:")
print(f"  R² Score: {r2_score(y_test_r, y_pred_ridge_final):.4f}")
print(f"  MSE: {mean_squared_error(y_test_r, y_pred_ridge_final):.4f}")
print(f"  MAE: {mean_absolute_error(y_test_r, y_pred_ridge_final):.4f}")


# 7. LASSO REGRESSION

print("\n7. LASSO REGRESSION")
print("-" * 80)

lasso_scores = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_r_scaled, y_train_r)
    y_pred_lasso = lasso.predict(X_test_r_scaled)
    score = r2_score(y_test_r, y_pred_lasso)
    lasso_scores.append(score)
    print(f"Lasso (alpha={alpha:6.2f}): R² = {score:.4f}")

best_alpha_lasso = alphas[np.argmax(lasso_scores)]
print(f"\n✓ Melhor alpha para Lasso: {best_alpha_lasso}")

lasso_final = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso_final.fit(X_train_r_scaled, y_train_r)
y_pred_lasso_final = lasso_final.predict(X_test_r_scaled)

print(f"Lasso Final:")
print(f"  R² Score: {r2_score(y_test_r, y_pred_lasso_final):.4f}")
print(f"  MSE: {mean_squared_error(y_test_r, y_pred_lasso_final):.4f}")
print(f"  MAE: {mean_absolute_error(y_test_r, y_pred_lasso_final):.4f}")

n_features_selected = np.sum(lasso_final.coef_ != 0)
print(f"  Features selecionadas: {n_features_selected}/{X_reg.shape[1]}")


# 8. REGRESSÃO POLINOMIAL

print("\n8. REGRESSÃO POLINOMIAL")
print("-" * 80)

degrees = [2, 3]

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_r_scaled)
    X_test_poly = poly.transform(X_test_r_scaled)

    ridge_poly = Ridge(alpha=1.0)
    ridge_poly.fit(X_train_poly, y_train_r)
    y_pred_poly = ridge_poly.predict(X_test_poly)

    print(f"Regressão Polinomial (grau {degree}):")
    print(f"  R² Score: {r2_score(y_test_r, y_pred_poly):.4f}")
    print(f"  MSE: {mean_squared_error(y_test_r, y_pred_poly):.4f}")
    print(f"  Features geradas: {X_train_poly.shape[1]}")
    print()


# 9. ÁRVORE DE DECISÃO E RANDOM FOREST

print("\n9. ÁRVORE DE DECISÃO E RANDOM FOREST")
print("-" * 80)

dt_class = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_class.fit(X_train_c_scaled, y_train_c)
y_pred_dt_c = dt_class.predict(X_test_c_scaled)

print("Árvore de Decisão (Classificação):")
print(f"  Acurácia: {accuracy_score(y_test_c, y_pred_dt_c):.4f}")
print(f"  F1-Score: {f1_score(y_test_c, y_pred_dt_c):.4f}")

rf_class = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_class.fit(X_train_c_scaled, y_train_c)
y_pred_rf_c = rf_class.predict(X_test_c_scaled)

print("\nRandom Forest (Classificação):")
print(f"  Acurácia: {accuracy_score(y_test_c, y_pred_rf_c):.4f}")
print(f"  F1-Score: {f1_score(y_test_c, y_pred_rf_c):.4f}")

feature_importance = pd.DataFrame({
    'Feature': [f'Feature_{i}' for i in range(X_class.shape[1])],
    'Importância': rf_class.feature_importances_
}).sort_values('Importância', ascending=False)

print("\nImportância das Features (Random Forest):")
print(feature_importance.head())

dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_reg.fit(X_train_r_scaled, y_train_r)
y_pred_dt_r = dt_reg.predict(X_test_r_scaled)

print("\nÁrvore de Decisão (Regressão):")
print(f"  R² Score: {r2_score(y_test_r, y_pred_dt_r):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_dt_r)):.4f}")

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_reg.fit(X_train_r_scaled, y_train_r)
y_pred_rf_r = rf_reg.predict(X_test_r_scaled)

print("\nRandom Forest (Regressão):")
print(f"  R² Score: {r2_score(y_test_r, y_pred_rf_r):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_rf_r)):.4f}")


# 10. RECOMENDAÇÃO FINAL

print("\n10. RECOMENDAÇÃO FINAL")
print("=" * 80)

print("\nRESULTADOS DE CLASSIFICAÇÃO:")
print("-" * 80)

classification_results = pd.DataFrame({
    'Modelo': ['KNN', 'Regressão Logística', 'Árvore de Decisão', 'Random Forest'],
    'Acurácia': [
        accuracy_score(y_test_c, y_pred_knn_best),
        accuracy_score(y_test_c, y_pred_logreg),
        accuracy_score(y_test_c, y_pred_dt_c),
        accuracy_score(y_test_c, y_pred_rf_c)
    ],
    'F1-Score': [
        f1_score(y_test_c, y_pred_knn_best),
        f1_score(y_test_c, y_pred_logreg),
        f1_score(y_test_c, y_pred_dt_c),
        f1_score(y_test_c, y_pred_rf_c)
    ],
    'Interpretabilidade': ['Baixa', 'Alta', 'Média', 'Baixa']
})

print(classification_results.to_string(index=False))

print("\nRESULTADOS DE REGRESSÃO:")
print("-" * 80)

regression_results = pd.DataFrame({
    'Modelo': ['Ridge', 'Lasso', 'Árvore de Decisão', 'Random Forest'],
    'R² Score': [
        r2_score(y_test_r, y_pred_ridge_final),
        r2_score(y_test_r, y_pred_lasso_final),
        r2_score(y_test_r, y_pred_dt_r),
        r2_score(y_test_r, y_pred_rf_r)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test_r, y_pred_ridge_final)),
        np.sqrt(mean_squared_error(y_test_r, y_pred_lasso_final)),
        np.sqrt(mean_squared_error(y_test_r, y_pred_dt_r)),
        np.sqrt(mean_squared_error(y_test_r, y_pred_rf_r))
    ],
    'Interpretabilidade': ['Média', 'Alta', 'Média', 'Baixa']
})

print(regression_results.to_string(index=False))

print("\n" + "=" * 80)
print("RECOMENDAÇÕES FINAIS:")
print("=" * 80)

best_class = classification_results.loc[classification_results['F1-Score'].idxmax()]
best_reg = regression_results.loc[regression_results['R² Score'].idxmax()]

print(f"""
PARA PROBLEMAS DE CLASSIFICAÇÃO:
  ✓ Melhor modelo: {best_class['Modelo']}
  ✓ F1-Score: {best_class['F1-Score']:.4f}
  ✓ Interpretabilidade: {best_class['Interpretabilidade']}

PARA PROBLEMAS DE REGRESSÃO:
  ✓ Melhor modelo: {best_reg['Modelo']}
  ✓ R² Score: {best_reg['R² Score']:.4f}
  ✓ Interpretabilidade: {best_reg['Interpretabilidade']}

CRITÉRIOS DE ESCOLHA:
  • Se interpretabilidade é crítica → Regressão Logística ou Lasso
  • Se performance é prioritária → Random Forest ou modelos ensemble
  • Se dados são simples e pequenos → KNN ou Árvore de Decisão
  • Se há muitas features irrelevantes → Lasso (seleção automática)
  • Se há relações não-lineares → Random Forest ou Regressão Polinomial

PRÓXIMOS PASSOS:
  1. Realizar feature engineering mais elaborado
  2. Otimizar hiperparâmetros com GridSearchCV
  3. Analisar curvas de aprendizado para detectar overfitting
  4. Validar modelos em dados de produção
  5. Implementar monitoramento contínuo de performance
""")

print("ANÁLISE COMPLETA FINALIZADA!")
