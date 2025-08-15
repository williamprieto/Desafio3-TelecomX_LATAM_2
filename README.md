# Challenge Telecom X – Predicción de Cancelación de Clientes (Churn Prediction)

## 📊 Descripción
Proyecto de análisis y modelado predictivo para anticipar la cancelación de clientes en una empresa de telecomunicaciones. Utiliza técnicas de machine learning, análisis exploratorio y herramientas de interpretabilidad como SHAP, con el objetivo de entender los factores que influyen en el abandono y proponer estrategias de retención efectivas.

## 🧠 Objetivo del Proyecto

Desarrollar un modelo capaz de predecir si un cliente cancelará su servicio, con base en sus características de contrato, historial de pagos y uso de servicios. Esto permite a la empresa tomar acciones proactivas y reducir su tasa de cancelación (churn rate).

## 🔍  Metodología

1. **Carga y limpieza de datos**
   - Eliminación de ID único
   - Codificación de variables categóricas con `get_dummies`
   - Balanceo de clases con **SMOTE**
   - Normalización con **StandardScaler**

2. **Análisis Exploratorio (EDA)**
   - Histogramas, boxplots y gráficos de correlación
   - Identificación de variables influyentes
   - Análisis de churn por tipo de contrato, servicios y método de pago

3. **Selección de características**
   - `SelectKBest` con `f_classif`
   - K óptimo = 29 variables seleccionadas por rendimiento

4. **Entrenamiento de modelos**
   - Regresión Logística (`LogisticRegression`)
   - Random Forest (`RandomForestClassifier`)
   - Gradient Boosting (`GradientBoostingClassifier`)
   - K-Nearest Neighbors (`KNN`)
   - Ensemble (VotingClassifier: LR + GB)

5. **Optimización de hiperparámetros**
   - `GridSearchCV` con validación cruzada estratificada

6. **Evaluación**
   - Métricas: Accuracy, Precision, Recall, F1 Score
   - Matrices de confusión
   - Curvas ROC y AUC
   - Interpretación con SHAP y coeficientes logísticos


## Herramientas Utilizadas

+ Python 3.8+
+ pandas
+ numpy
+ matplotlib
+ seaborn
+ scikit-learn
+ imblearn
+ shap
