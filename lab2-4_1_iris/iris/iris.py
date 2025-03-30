import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tpot import TPOTClassifier # Или TPOTRegressor для регрессии

# 1. Загрузка и визуальный анализ данных
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, names=column_names)

print(df.head())
print(df.describe())
print(df.info())
print(df['species'].value_counts())


# Pairplot
sns.pairplot(df, hue='species')
plt.show()

# 2. Бинарная классификация с LogisticRegression
# Выберем два класса для бинарной классификации (например, 'Iris-setosa' и 'Iris-versicolor')
df_binary = df[df['species'].isin(['Iris-setosa', 'Iris-versicolor'])].copy() # Используем copy(), чтобы избежать SettingWithCopyWarning
df_binary['species'] = df_binary['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})

X = df_binary.drop('species', axis=1)
y = df_binary['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Логистическая регрессия
model = LogisticRegression(random_state=42, solver='liblinear') # solver='liblinear' подходит для небольших датасетов
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 3. Переобучение линейной регрессии и борьба с ним
# Используем все данные и попробуем предсказать sepal_length на основе остальных признаков
X = df.drop('sepal_length', axis=1)
# Преобразуем категориальные данные в числовые, используя one-hot encoding
X = pd.get_dummies(X, columns=['species'], drop_first=True) # drop_first=True для избежания мультиколлинеарности
y = df['sepal_length']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Линейная регрессия с полиномиальными признаками
def train_and_evaluate(degree):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()), # Важно масштабировать данные перед полиномиальными признаками
        ('linear', LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f'Degree: {degree}')
    print(f'Train RMSE: {train_rmse}')
    print(f'Test RMSE: {test_rmse}')
    print("Cross-validation scores:")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(np.sqrt(-cv_scores)) # Convert to RMSE
    print(f"Mean CV RMSE: {np.sqrt(-cv_scores).mean()}")
    print("---")
    return pipeline

# Обучаем и оцениваем модели с разной степенью полинома
pipeline_1 = train_and_evaluate(1) # Линейная модель (без полиномиальных признаков)
pipeline_2 = train_and_evaluate(2)
pipeline_3 = train_and_evaluate(5) # Модель, склонная к переобучению

# Вывод: При увеличении степени полинома, Train RMSE уменьшается, а Test RMSE сначала уменьшается, потом увеличивается,
#        что говорит о переобучении. Cross-validation scores также показывают эту тенденцию.

# Борьба с переобучением: регуляризация, кросс-валидация, упрощение модели (уменьшение степени полинома).
# В данном примере, уменьшение степени полинома до 1 или 2 может помочь избежать переобучения.

# 4. Обогащение датасета сгенерированными данными

# Метод: Генерация синтетических данных на основе существующих, с небольшим добавлением случайного шума.
def generate_synthetic_data(df, num_samples=50, noise_level=0.05):
    """Генерирует синтетические данные на основе существующего датасета."""
    synthetic_data = []
    for i in range(num_samples):
        # Выбираем случайную строку из исходного датасета
        original_row = df.sample(n=1).iloc[0]
        
        # Добавляем небольшой случайный шум к числовым признакам
        new_row = original_row.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                noise = np.random.normal(0, noise_level * df[col].std())
                new_row[col] += noise

        synthetic_data.append(new_row)

    synthetic_df = pd.DataFrame(synthetic_data)
    return synthetic_df

# Генерируем 50 новых образцов
synthetic_df = generate_synthetic_data(df, num_samples=50, noise_level=0.05)

# Добавляем сгенерированные данные к исходному датасету
df_augmented = pd.concat([df, synthetic_df], ignore_index=True)
print(f'Размер исходного датасета: {len(df)}')
print(f'Размер обогащенного датасета: {len(df_augmented)}')

# Визуализируем, чтобы убедиться, что данные выглядят разумно
sns.pairplot(df_augmented, hue='species') # Теперь в dataset 200 экземпляров.
plt.show()

# 5. Эксперименты с AutoML (TPOT)
# TPOT (Tree-based Pipeline Optimization Tool) - это Python AutoML библиотека, которая автоматически проектирует и оптимизирует конвейеры машинного обучения.

# Разделение данных на обучающую и тестовую выборки
X = df_augmented.drop('species', axis=1)
y = df_augmented['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация и обучение TPOTClassifier
tpot = TPOTClassifier(generations=5, population_size=20, random_state=42, verbosity=2)
tpot.fit(X_train, y_train)

# Оценка производительности лучшей модели
print(tpot.score(X_test, y_test))

# Экспорт лучшего конвейера в виде Python-кода (полезно для дальнейшего использования)
tpot.export('tpot_iris_pipeline.py') # Он создаст Python-файл с автоматически сгенерированным кодом для лучшей модели

# Теперь вы можете импортировать эту модель из файла tpot_iris_pipeline.py и использовать её.

# Дополнительные шаги (после выполнения ноутбука):
# 1. Зафиксируйте изменения в Git:  git add . ; git commit -m "Added initial solution"
# 2. Создайте ветку для экспериментов: git checkout -b experiment_automl
# 3. Зафиксируйте изменения в ветке: git add . ; git commit -m "Experimented with TPOT AutoML"
# 4. Слейте изменения в основную ветку: git checkout main ; git merge experiment_automl
# 5. Запушьте изменения на GitHub: git push origin main
