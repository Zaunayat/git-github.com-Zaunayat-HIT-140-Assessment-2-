# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy.stats import boxcox

# Setting environment to ignore future warnings
import warnings
warnings.simplefilter('ignore')

# Loading the Dataset
parkinson_data = pd.read_csv('po2_data.csv')
parkinson_data.head()


# ## Data Exploration

# --> Checking for missing values <br>
# --> Obtaining statistical summaries for the dataset <br>
# --> Visualizing distributions of key variables

# Function to perform all EDA
def perform_eda(df, name=""):
    # Printing basic detail of data like name, size, shape
    print(f"EDA of {str(name)} Data....")
    print(f"Size {df.size}")
    print(f"Columns {df.shape[1]}")
    print(f"Records {df.shape[0]}")
    print("="*50)
    
    # Printing top 5 records of data
    print("First Look of Data....")
    print(df.head())
    print("="*50)
    
    # Getting Numerical and Categorical columns Separately
    cat_cols = df.select_dtypes(object).columns
    num_cols = df.select_dtypes(np.number).columns

    # Printing the Numerical columns
    print("Dataset has following Numerical columns...")
    for i, j in enumerate(num_cols):
        print(f" {i+1}) {j}")

    # Printing the Categorical columns
    print("\n\nDataset has following Categorical columns...")
    for i, j in enumerate(cat_cols):
        print(f" {i+1}) {j}")
    
    # Printing info of data like data type, non null values
    print("="*50)
    print("Information of Data....")
    print(df.info())
    print("="*50)
    
    # Displaying statistical properties of data like mean, median, max, min
    print("Statistical Properties of Data....")
    print(df.describe(include="all"))
    print("="*50)


perform_eda(parkinson_data, "Parkinson Disease")


# Set up the figure and axes
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

# Plot the distribution of age
sns.histplot(parkinson_data['age'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Age')

# Plot the distribution of motor_updrs
sns.histplot(parkinson_data['motor_updrs'], kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Motor UPDRS Score')

# Plot the distribution of total_updrs
sns.histplot(parkinson_data['total_updrs'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Total UPDRS Score')

# Plot the distribution of a voice measure (e.g., jitter(%))
sns.histplot(parkinson_data['jitter(%)'], kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Jitter(%)')

plt.tight_layout()
plt.show()


# __Distribution of Age__: The age distribution appears to be somewhat uniform, with a slight increase in the number of subjects around the ages of 60-70.<br>
# __Distribution of Motor UPDRS Score__: The motor UPDRS scores seem to have a relatively normal distribution, centered around 20.<br>
# __Distribution of Total UPDRS Score__: The total UPDRS scores are slightly right-skewed, with a mode around the 25-30 range.<br>
# __Distribution of Jitter(%)__: The jitter distribution is right-skewed, with most values close to 0. A few outliers have high jitter values.

# ## Data Visualization

# - __Plot the Relationships Between Predictors and Target Variables__

# Set up the figure and axes
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 15))

# Voice measures to visualize
voice_measures = ['jitter(%)', 'shimmer(%)', 'nhr']

# Plot scatter plots for voice measures against motor_updrs and total_updrs
for i, measure in enumerate(voice_measures):
    sns.scatterplot(x=measure, y='motor_updrs', data=parkinson_data, ax=axes[i, 0], alpha=0.5)
    axes[i, 0].set_title(f'Motor UPDRS vs. {measure}')
    
    sns.scatterplot(x=measure, y='total_updrs', data=parkinson_data, ax=axes[i, 1], alpha=0.5)
    axes[i, 1].set_title(f'Total UPDRS vs. {measure}')

plt.tight_layout()
plt.show()


# __Jitter(%):__ Higher values of jitter seem to be associated with higher UPDRS scores, but the relationship is not strongly linear.<br>
# __Shimmer(%):__ Similarly, higher shimmer values tend to be associated with higher UPDRS scores. This relationship seems to be more pronounced for the total UPDRS scores.<br>
# __NHR (Noise-to-Harmonics Ratio):__ The majority of data points are clustered at the lower end, with a few outliers at the higher end. It's harder to determine a clear relationship with UPDRS scores from this visualization.

# - __Correlation Matrix__

# Compute the correlation matrix
correlation_matrix = parkinson_data.corr()

# Focus on correlations of voice measures with motor_updrs and total_updrs
correlations_with_updrs = correlation_matrix[['motor_updrs', 'total_updrs']].sort_values(by='total_updrs', ascending=False)

# Display the correlations
correlations_with_updrs


# Set up the figure for motor_updrs
plt.figure(figsize=(18, 1))

# Plot horizontal heatmap for correlations with motor_updrs
sns.heatmap(correlation_matrix[['motor_updrs']].sort_values(by='motor_updrs', ascending=False).T, fmt='.1f',
            annot=True, cmap='rocket', vmin=-1, vmax=1, cbar_kws={"orientation": "vertical"})

# Set the title
plt.title('Correlations with Motor UPDRS')
plt.show()


# Set up the figure for total_updrs
plt.figure(figsize=(18, 1))

# Plot vertical heatmap for correlations with total_updrs
sns.heatmap(correlation_matrix[['total_updrs']].sort_values(by='total_updrs', ascending=False).T, fmt='.1f',
            annot=True, cmap='rocket', vmin=-1, vmax=1, cbar_kws={"orientation": "vertical"})

# Set the title
plt.title('Correlations with Total UPDRS')
plt.show()


# ## Data Preparation & Modeling

def data_preparation(data, target, test_size):
    # Splitting the data into training and test sets for motor_updrs
    X_train_motor, X_test_motor, y_train_motor, y_test_motor = train_test_split(
        data, target, test_size=test_size, random_state=42)
    
    # Initialize the standard scaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and test data for motor_updrs
    X_train_scaled = scaler.fit_transform(X_train_motor)
    X_test_scaled = scaler.transform(X_test_motor)


    # Checking the mean and standard deviation of the scaled training data for motor_updrs
    X_train_scaled.mean(axis=0)

    return X_train_scaled, X_test_scaled, y_train_motor, y_test_motor


def train_Linear_Regression(xTrain, xTest, yTrain, yTest):
    # Initialize the linear regression model
    lr = LinearRegression()

    # Train the model on the scaled training data for motor_updrs
    lr.fit(X_train, y_train)

    # Predict on the test data
    y_pred = lr.predict(X_test)

    # Evaluate the model's performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    nrmse = rmse / (y_test.max() - y_test.min())
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

    return mae, mse, rmse, nrmse, r2, adjusted_r2


# Features (excluding target variables and identifiers)
features = parkinson_data.drop(columns=['subject#', 'motor_updrs', 'total_updrs'])

# Targets
motor_updrs_target = parkinson_data['motor_updrs']
total_updrs_target = parkinson_data['total_updrs']


# ## Task 1

# - __Motor UPDRS__

X_train, X_test, y_train, y_test = data_preparation(features, motor_updrs_target, 0.2)


mae, mse, rmse, nrmse, r2, adjusted_r2 = train_Linear_Regression(X_train, X_test, y_train, y_test)


print("*"*50)
print("Mean Absolute Error: ", round(mae, 2))
print("Mean Squre Error: ", round(mse ,2))
print("Root Mean Square Error: ", round(rmse, 2))
print("Normalizaed Root Mean Square Error: ", round(nrmse, 2))
print("R2 Score: ", round(r2, 2))
print("Adjusted R2 Score: ", round(adjusted_r2, 2))
print("*"*50)


# - __Total UPDRs__

X_train, X_test, y_train, y_test = data_preparation(features, total_updrs_target, 0.2)


mae, mse, rmse, nrmse, r2, adjusted_r2 = train_Linear_Regression(X_train, X_test, y_train, y_test)


print("*"*50)
print("Mean Absolute Error: ", round(mae, 2))
print("Mean Squre Error: ", round(mse ,2))
print("Root Mean Square Error: ", round(rmse, 2))
print("Normalizaed Root Mean Square Error: ", round(nrmse, 2))
print("R2 Score: ", round(r2, 2))
print("Adjusted R2 Score: ", round(adjusted_r2, 2))
print("*"*50)


# ## Task 2

# - __Motor UPDRS__

test_sizes = [0.5, 0.4, 0.3, 0.2]
for size in test_sizes:
    X_train, X_test, y_train, y_test = data_preparation(features, motor_updrs_target, size)
    mae, mse, rmse, nrmse, r2, adjusted_r2 = train_Linear_Regression(X_train, X_test, y_train, y_test)
    print("*"*25, f'Results for Test Size of {size*100}%', "*"*25)
    print("Mean Absolute Error: ", round(mae, 2))
    print("Mean Squre Error: ", round(mse ,2))
    print("Root Mean Square Error: ", round(rmse, 2))
    print("Normalizaed Root Mean Square Error: ", round(nrmse, 2))
    print("R2 Score: ", round(r2, 2))
    print("Adjusted R2 Score: ", round(adjusted_r2, 2))
    print("*"*80)
    print()


# - __Total UPDRS__

test_sizes = [0.5, 0.4, 0.3, 0.2]
for size in test_sizes:
    X_train, X_test, y_train, y_test = data_preparation(features, total_updrs_target, size)
    mae, mse, rmse, nrmse, r2, adjusted_r2 = train_Linear_Regression(X_train, X_test, y_train, y_test)
    print("*"*25, f'Results for Test Size of {size*100}%', "*"*25)
    print("Mean Absolute Error: ", round(mae, 2))
    print("Mean Squre Error: ", round(mse ,2))
    print("Root Mean Square Error: ", round(rmse, 2))
    print("Normalizaed Root Mean Square Error: ", round(nrmse, 2))
    print("R2 Score: ", round(r2, 2))
    print("Adjusted R2 Score: ", round(adjusted_r2, 2))
    print("*"*80)
    print()


# ## Task 3

# - __Log Transform__ & __Collinearity Analysis__ for __Motor UPDRS__

X_train, X_test, y_train, y_test = data_preparation(features, motor_updrs_target, 0.2)

# Apply log-transform to the target variables
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

# Visualize the distributions after log-transform
plt.figure(figsize=(15, 4))

# Plot the distribution of log-transformed motor_updrs
plt.subplot(1, 2, 1)
sns.histplot(y_train, kde=True, bins=30)
plt.title('Distribution of log-transformed motor_updrs')

plt.tight_layout()
plt.show()


# Calculate VIF for each feature
X_with_const = sm.add_constant(X_train)  # Adding a constant for the intercept term
# Correcting the VIF computation
vif_data = pd.DataFrame()
vif_data["Feature"] = features.columns
vif_data["VIF"] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]

vif_data.sort_values(by="VIF", ascending=False)


# Features to be removed based on high VIF values
features_to_remove = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()

# Removing these features from our datasets
X_train = pd.DataFrame(X_train, columns=features.columns).drop(columns=features_to_remove)
X_test = pd.DataFrame(X_test, columns=features.columns).drop(columns=features_to_remove)

mae, mse, rmse, nrmse, r2, adjusted_r2 = train_Linear_Regression(X_train, X_test, y_train, y_test)


print("*"*50)
print("Mean Absolute Error: ", round(mae, 2))
print("Mean Squre Error: ", round(mse ,2))
print("Root Mean Square Error: ", round(rmse, 2))
print("Normalizaed Root Mean Square Error: ", round(nrmse, 2))
print("R2 Score: ", round(r2, 2))
print("Adjusted R2 Score: ", round(adjusted_r2, 2))
print("*"*50)


# - __Log Transform__ & __Collinearity Analysis__ for __Total UPDRS__

X_train, X_test, y_train, y_test = data_preparation(features, total_updrs_target, 0.2)

# Apply log-transform to the target variables
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

# Visualize the distributions after log-transform
plt.figure(figsize=(15, 4))

# Plot the distribution of log-transformed motor_updrs
plt.subplot(1, 2, 1)
sns.histplot(y_train, kde=True, bins=30)
plt.title('Distribution of log-transformed total_updrs')

plt.tight_layout()
plt.show()

# Calculate VIF for each feature
X_with_const = sm.add_constant(X_train)  # Adding a constant for the intercept term
# Correcting the VIF computation
vif_data = pd.DataFrame()
vif_data["Feature"] = features.columns
vif_data["VIF"] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]

vif_data.sort_values(by="VIF", ascending=False)


# Features to be removed based on high VIF values
features_to_remove = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()

# Removing these features from our datasets
X_train = pd.DataFrame(X_train, columns=features.columns).drop(columns=features_to_remove)
X_test = pd.DataFrame(X_test, columns=features.columns).drop(columns=features_to_remove)


print("*"*50)
print("Mean Absolute Error: ", round(mae, 2))
print("Mean Squre Error: ", round(mse ,2))
print("Root Mean Square Error: ", round(rmse, 2))
print("Normalizaed Root Mean Square Error: ", round(nrmse, 2))
print("R2 Score: ", round(r2, 2))
print("Adjusted R2 Score: ", round(adjusted_r2, 2))
print("*"*50)


# ## Task 4

# - __Gaussian Transformation__ for __Motor UpDrs__

X_train, X_test, y_train, y_test = data_preparation(features, motor_updrs_target, 0.2)


# Applying Box-Cox transformation
y_train, motor_lambda = boxcox(y_train)
y_test = boxcox(y_test, lmbda=motor_lambda)

# Visualize the distributions after Box-Cox transformation
plt.figure(figsize=(15, 4))

# Plot the distribution of Box-Cox transformed motor_updrs
plt.subplot(1, 2, 1)
sns.histplot(y_train, kde=True, bins=30)
plt.title('Distribution of Box-Cox transformed motor_updrs')

plt.tight_layout()
plt.show()


mae, mse, rmse, nrmse, r2, adjusted_r2 = train_Linear_Regression(X_train, X_test, y_train, y_test)


print("*"*50)
print("Mean Absolute Error: ", round(mae, 2))
print("Mean Squre Error: ", round(mse ,2))
print("Root Mean Square Error: ", round(rmse, 2))
print("Normalizaed Root Mean Square Error: ", round(nrmse, 2))
print("R2 Score: ", round(r2, 2))
print("Adjusted R2 Score: ", round(adjusted_r2, 2))
print("*"*50)


# - __Gaussian Transformation__ for __Total UpDrs__

X_train, X_test, y_train, y_test = data_preparation(features, total_updrs_target, 0.2)


# Applying Box-Cox transformation
y_train, motor_lambda = boxcox(y_train)
y_test = boxcox(y_test, lmbda=motor_lambda)

# Visualize the distributions after Box-Cox transformation
plt.figure(figsize=(15, 4))

# Plot the distribution of Box-Cox transformed motor_updrs
plt.subplot(1, 2, 1)
sns.histplot(y_train, kde=True, bins=30)
plt.title('Distribution of Box-Cox transformed motor_updrs')

plt.tight_layout()
plt.show()


mae, mse, rmse, nrmse, r2, adjusted_r2 = train_Linear_Regression(X_train, X_test, y_train, y_test)


print("*"*50)
print("Mean Absolute Error: ", round(mae, 2))
print("Mean Squre Error: ", round(mse ,2))
print("Root Mean Square Error: ", round(rmse, 2))
print("Normalizaed Root Mean Square Error: ", round(nrmse, 2))
print("R2 Score: ", round(r2, 2))
print("Adjusted R2 Score: ", round(adjusted_r2, 2))
print("*"*50)