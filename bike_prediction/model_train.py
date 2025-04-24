import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load your CSV
df = pd.read_csv("Used_Bikes.csv")

# Separate features and target
X = df.drop(columns='price')
y = df['price']

# Categorical and numerical columns
categorical_cols = ['bike_name', 'city', 'owner', 'brand']
numerical_cols = ['kms_driven', 'age', 'power']

# Preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # keep numeric features
)

# Pipeline: preprocessing + model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Save model
joblib.dump(model_pipeline, 'bike_price_predictor.pkl')

print("Model trained and saved as 'bike_price_predictor.pkl'")
