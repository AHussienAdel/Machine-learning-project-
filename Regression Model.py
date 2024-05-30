import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import DBSCAN
import seaborn as sns


df = pd.read_csv("C:\\Users\\HP\\Desktop\\ml project\\DoctorFeePrediction.csv")
df.drop_duplicates(subset=['Doctor Name'], keep='first', inplace=True)

rows_removed = len(df)
print(f"Number of rows remaining from 2387: {rows_removed}")

num_rows = df.shape[0]

print(f"The DataFrame has {num_rows} rows before removing outliers.")

dbscan = DBSCAN(eps=0.5, min_samples=10)
clusters = dbscan.fit_predict(np.expand_dims(df['Fee(PKR)'], axis=1))
df['cluster'] = clusters
df = df[df['cluster'] != -1]  # -1 in cluster label is considered as noise or outlier
num_rows = df.shape[0]

print(f"The DataFrame has {num_rows} rows after removing outliers.")


'''from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df['Fee(PKR)'] = scaler.fit_transform(df[['Fee(PKR)']])'''

columns_to_drop = ['Doctor Name']
df.drop(columns=columns_to_drop, inplace=True)

cols_to_encode = ['City']
def feature_encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(X[c].astype(str))
        X[c] = lbl.transform(X[c].astype(str))
    return X

df_encoded = feature_encoder(df.copy(), cols_to_encode)

df_encoded_specialization = df['Specialization'].str.get_dummies(sep=', ')
df_encoded = pd.concat([df_encoded, df_encoded_specialization], axis=1)
df_encoded.drop(columns=['Specialization'], inplace=True)


df_encoded_qualifications = df['Doctor Qualification'].str.get_dummies(sep=', ')
df_encoded = pd.concat([df_encoded, df_encoded_qualifications], axis=1)

df_encoded.drop(columns=['Doctor Qualification'], inplace=True)


def encode_doctors_link(link):
    if link.startswith("https://"):
        return 1
    elif link == "No Link Available":
        return 0
    else:
        return -1

df_encoded['Doctors Link'] = df['Doctors Link'].apply(encode_doctors_link)

def encode_hospital_address(address):
    if address != "No Address Available":
        return 1
    else:
        return 0

df_encoded['Hospital Address'] = df['Hospital Address'].apply(encode_hospital_address)

all_columns = df_encoded.columns.tolist()
all_columns.remove("Fee(PKR)")
scaler = MinMaxScaler()
df_encoded[all_columns] = scaler.fit_transform(df_encoded[all_columns])

X = df_encoded.drop(columns=["Fee(PKR)"])
Y = df["Fee(PKR)"]

correlation_with_fee = df_encoded_specialization.corrwith(df_encoded['Fee(PKR)'])

#-----------------------------------
correlation = df_encoded.corr()['Fee(PKR)']
high_correlation_features = correlation[abs(correlation) > 0.3].index
print("Features with correlation greater than 0.3:")
print(high_correlation_features)
#------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.37, random_state=40)
#============================================================================================

#-----------------------------------------------------------------------------------------------
#select top 5 features with kBest
k = 5
selector = SelectKBest(score_func=f_regression, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_indices]
print("Selected Features:", selected_features)
#----------------------------------------------------------------------------
#Top 5 Correlation Dataframe
correlation_with_fee = X.corrwith(Y)
sorted_correlations = correlation_with_fee.sort_values(ascending=False)
top_5_features = sorted_correlations.head(3).index
new_df = X[top_5_features]
print("New DataFrame with top 3 features with the highest correlation with Fee(PKR):")
print(new_df.columns)
#-----------------------------------------------------------------------------
degree = 3
poly = PolynomialFeatures(degree=degree)
Xx_poly = poly.fit_transform(new_df)
Xx_train, Xx_test, yy_train, yy_test = train_test_split(Xx_poly, Y, test_size=0.37, random_state=40)
model = LinearRegression()
model.fit(Xx_train, yy_train)
yy_pred = model.predict(Xx_test)
mse = mean_squared_error(yy_test, yy_pred)
print("Mean Squared Error (Polynomial Regression):", mse)

#==========================================================================
#========================================================================
#random forest
# Initialize VarianceThreshold with a threshold
variance_threshold = VarianceThreshold(threshold=0.005)  # Adjust the threshold as needed

variance_threshold.fit(X_train)

X_train_selected = variance_threshold.transform(X_train)
X_test_selected = variance_threshold.transform(X_test)

rf_model = RandomForestRegressor(
    n_estimators=380,
    max_depth=18,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=35
)

rf_model.fit(X_train_selected, y_train)
rf_y_pred_test = rf_model.predict(X_test_selected)
rf_mse_test = mean_squared_error(y_test, rf_y_pred_test)

rf_y_pred_train = rf_model.predict(X_train_selected)

rf_mse_train = mean_squared_error(y_train, rf_y_pred_train)

print(f"Random Forest Regressor - MSE on Testing Data: {rf_mse_test}")
print(f"Random Forest Regressor - MSE on Training Data: {rf_mse_train}")
rf_rmse_test = np.sqrt(rf_mse_test)

rf_rmse_train = np.sqrt(rf_mse_train)

print(f"Random Forest Regressor - RMSE on Testing Data: {rf_rmse_test}")
print(f"Random Forest Regressor - RMSE on Training Data: {rf_rmse_train}")
rf_r2_test = rf_model.score(X_test_selected, y_test)
print(f"Random Forest Regressor - R^2 (Testing): {rf_r2_test}")

rf_r2_train = rf_model.score(X_train_selected, y_train)
print(f"Random Forest Regressor - R^2 (Training): {rf_r2_train}")
train_sizes, train_scores, test_scores = learning_curve(
    rf_model,
    X_train_selected,
    y_train,
    cv=5,
    scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Learning Curve (Random Forest Regressor)")
plt.xlabel("Training Examples")
plt.ylabel("Mean Squared Error")

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training error")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Testing error")

plt.legend(loc="best")
plt.show()
#=================================================================================
#gradient descent booster

estimator_model = GradientBoostingRegressor(
        random_state=38,
        learning_rate=0.046,
        n_estimators=195,
        max_depth=4,
        min_samples_leaf=5
    )
sfm = SelectFromModel(estimator=estimator_model, threshold='mean')
sfm.fit(X_train, y_train)
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)
gb_model = GradientBoostingRegressor(
    random_state=38,
    learning_rate=0.046,
    n_estimators=195,
    max_depth=4,
    min_samples_leaf=5
)
gb_model.fit(X_train_selected, y_train)
gb_y_pred_test = gb_model.predict(X_test_selected)
gb_mse_test = mean_squared_error(y_test, gb_y_pred_test)
print(f"Gradient Boosting Regressor - R^2 (Testing): {gb_model.score(X_test_selected, y_test)}")
print(f"Gradient Boosting Regressor - Mean Squared Error (Testing): {gb_mse_test}")
gb_y_pred_train = gb_model.predict(X_train_selected)
gb_mse_train = mean_squared_error(y_train, gb_y_pred_train)

print(f"Gradient Boosting Regressor - R^2 (Training): {gb_model.score(X_train_selected, y_train)}")
print(f"Gradient Boosting Regressor - Mean Squared Error (Training): {gb_mse_train}")

gb_rmse_test = np.sqrt(gb_mse_test)

gb_rmse_train = np.sqrt(gb_mse_train)

print(f"Gradient Boosting Regressor - RMSE (Testing): {gb_rmse_test}")
print(f"Gradient Boosting Regressor - RMSE (Training): {gb_rmse_train}")
train_sizes, train_scores, test_scores = learning_curve(
    gb_model,
    X_train_selected,
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1
)

train_mean = -np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = -np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label='Training Error', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
plt.plot(train_sizes, test_mean, label='Testing Error', color='red')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='red')
plt.xlabel('Number of Training Samples')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve - Gradient Boosting Regressor')
plt.legend()
plt.show()
#================================================================================
# Perform feature engineering
feature1 = 'Experience(Years)'
feature2 = 'Total_Reviews'
new_feature = df_encoded[feature1] * df_encoded[feature2]
df_encoded = pd.concat([df_encoded, new_feature.rename('Experience_Total_Reviews')], axis=1)
correlation_engineered_feature = df_encoded['Experience_Total_Reviews'].corr(df_encoded['Fee(PKR)'])
print(f"Correlation between '{feature1} * {feature2}' and 'Fee(PKR)': {correlation_engineered_feature}")
feature3 = 'Total_Reviews'
feature4 = 'Doctors Link'
new_feature_name = f'{feature3}_{feature4}_interaction'
new_feature = df_encoded[feature3] + df_encoded[feature4]
df_encoded[new_feature_name] = new_feature  # Add the new feature directly
correlation_engineered_feature = df_encoded[new_feature_name].corr(df_encoded['Fee(PKR)'])
print(f"Correlation between '{feature3} + {feature4}' and 'Fee(PKR)': {correlation_engineered_feature}")
feature5 = 'Experience(Years)'
feature6 = 'Doctors Link'
new_feature_name = f'{feature5}_{feature6}_product'
new_feature = df_encoded[feature5] * df_encoded[feature6]
df_encoded[new_feature_name] = new_feature  # Add the new feature directly
correlation_engineered_feature = df_encoded[new_feature_name].corr(df_encoded['Fee(PKR)'])
print(f"Correlation between '{feature5} * {feature6}' and 'Fee(PKR)': {correlation_engineered_feature}")
print("Engineered features added:", [new_feature_name, f'{feature3}_{feature4}_interaction'])
engineered_features_df = df_encoded[['Experience_Total_Reviews', f'{feature3}_{feature4}_interaction', f'{feature5}_{feature6}_product']]
degree = 3
poly = PolynomialFeatures(degree=degree)
engineered_features_poly = poly.fit_transform(engineered_features_df)
X_train, X_test, y_train, y_test = train_test_split(engineered_features_poly, df_encoded['Fee(PKR)'], test_size=0.37, random_state=40)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (Polynomial Regression with degree 3):", mse)
