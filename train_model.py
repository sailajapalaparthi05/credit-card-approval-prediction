import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load datasets
app = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

# Create target
bad_status = ["1", "2", "3", "4", "5"]
credit["TARGET"] = credit["STATUS"].astype(str).apply(
    lambda x: 1 if x in bad_status else 0
)

target = credit.groupby("ID")["TARGET"].max().reset_index()

# Merge
df = app.merge(target, on="ID", how="inner")

# Drop ID
df.drop(columns=["ID"], inplace=True)

# Features & Target
X = df.drop("TARGET", axis=1)
y = df["TARGET"]

# Column types
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"

    ))
])

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(df["TARGET"].value_counts())

# Train
model.fit(X_train, y_train)

# Test
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save
joblib.dump(model, "credit_card_model.pkl")

print("credit_card_model.pkl created successfully!")