import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFromModel
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import hashlib

# 1. Read Data
data = pd.read_csv("C:\\Users\\HP\\Desktop\\ml project\\DoctorFeePrediction_Milestone2.csv")

if 'Fee Category' not in data.columns:
    raise ValueError("Column 'Fee Category' does not exist in the DataFrame. Please check the correct column name.")

data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

# Feature hashing for 'City'
def hash_city(city):
    hashed_value = int(hashlib.md5(city.encode()).hexdigest(), 16)
    return hashed_value % 100

data['City'] = data['City'].apply(hash_city)

# Extract qualifications and clean them
data['Doctor Qualification'] = data['Doctor Qualification'].apply(lambda x: re.findall(r"(MBBS|BDS|BAMS|BHMS|MD|MS|MDS|DNB|PhD)", x))
data['Doctor Qualification'] = data['Doctor Qualification'].apply(lambda x: ', '.join(sorted(set(x))))

# Encoding Functions
def encode_doctors_link(link):
    return 1 if link.startswith("https://") else 0 if link == "No Link Available" else -1

def encode_title(name):
    mapping = {'': 1, 'Dr.': 1, 'Asst. Prof. Dr.': 2, 'Prof. Dr.': 3, 'Assoc. Prof. Dr.': 4}
    titles = ' '.join(re.findall(r'(Dr\.|Prof\.|Asst\.|Assoc\.)', name))
    return mapping.get(titles, 1)

data['Doctors Link'] = data['Doctors Link'].apply(encode_doctors_link)
data['Doctor Name'] = data['Doctor Name'].apply(encode_title)

# Handle Missing Values
data.fillna(data.median(numeric_only=True), inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)

# Frequency Encoding for 'Hospital Address'
address_freq = data['Hospital Address'].value_counts().to_dict()
data['Hospital Address'] = data['Hospital Address'].map(address_freq)

# Label Encoding for 'Fee Category'
data['Fee Category'] = LabelEncoder().fit_transform(data['Fee Category'])

# Preprocessor for feature scaling and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('anova', SelectKBest(score_func=f_classif, k=4))
        ]), ['Experience(Years)', 'Total_Reviews', 'Patient Satisfaction Rate(%age)', 'Avg Time to Patients(mins)', 'Wait Time(mins)']),
        ('cat', Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
            ('chi2', SelectKBest(score_func=chi2, k=4))
        ]), ['City', 'Specialization', 'Doctor Qualification', 'Hospital Address', 'Doctor Name', 'Doctors Link'])
    ],
    remainder='drop'
)

# Models Configuration with best parameters
models = {
    'RandomForest': {
        'model': RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=300, random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [10], 'min_samples_split': [4,6], 'min_samples_leaf': [2,3]}
    },
    'SVC': {
        'model': SVC(kernel='linear', random_state=42),
        'params': {'C': [0.1]}
    }
}

# Training and Evaluation of Models
for name, config in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GridSearchCV(config['model'], config['params'], cv=5, scoring='accuracy'))
    ])
    pipeline.fit(data_train.drop('Fee Category', axis=1), data_train['Fee Category'])
    best_model = pipeline.named_steps['classifier'].best_estimator_

    y_train_pred = pipeline.predict(data_train.drop('Fee Category', axis=1))
    y_test_pred = pipeline.predict(data_test.drop('Fee Category', axis=1))

    train_accuracy = accuracy_score(data_train['Fee Category'], y_train_pred)
    test_accuracy = accuracy_score(data_test['Fee Category'], y_test_pred)

    print(f"{name} - Training Accuracy: {train_accuracy:.4f}, Testing Accuracy: {test_accuracy:.4f}")

# Additional Gradient Boosting Pipeline with Feature Selection
gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectFromModel(estimator=RandomForestClassifier(random_state=42))),
    ('classifier', GridSearchCV(GradientBoostingClassifier(random_state=42),
                                param_grid={'n_estimators': [50, 100, 150],
                                            'learning_rate': [0.01],
                                            'max_depth': [3, 4]},
                                cv=5,
                                scoring='accuracy'))
])

# Train and Evaluate Gradient Boosting with SelectFromModel
gb_pipeline.fit(data_train.drop('Fee Category', axis=1), data_train['Fee Category'])
best_model = gb_pipeline.named_steps['classifier'].best_estimator_

y_train_pred = gb_pipeline.predict(data_train.drop('Fee Category', axis=1))
y_test_pred = gb_pipeline.predict(data_test.drop('Fee Category', axis=1))

train_accuracy = accuracy_score(data_train['Fee Category'], y_train_pred)
test_accuracy = accuracy_score(data_test['Fee Category'], y_test_pred)

print(f"Gradient Boosting with SelectFromModel - Training Accuracy: {train_accuracy:.4f}, Testing Accuracy: {test_accuracy:.4f}")