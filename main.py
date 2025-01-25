# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
df = pd.read_csv('movies.csv')  # Replace with your own dataset path

# Display the first few rows of the dataframe
print("First few rows of the dataset:")
print(df.head())

# Step 2: Data Cleaning
# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Convert columns to the appropriate data types
df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')
df['BoxOffice'] = pd.to_numeric(df['BoxOffice'], errors='coerce')

# Verify data types
print("\nData types after conversion:")
print(df.dtypes)

# Step 3: Exploratory Data Analysis (EDA)

# 3.1: Top 10 movies by box office earnings
top_movies = df.sort_values(by='BoxOffice', ascending=False).head(10)
print("\nTop 10 movies by Box Office earnings:")
print(top_movies[['Title', 'BoxOffice']])

# 3.2: Movie Rating Distribution
sns.histplot(df['Rating'], kde=True)
plt.title('Movie Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# 3.3: Genres Popularity
genre_counts = df['Genre'].value_counts()
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Movie Genre Popularity')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 3.4: Budget vs Box Office Earnings
sns.scatterplot(data=df, x='Budget', y='BoxOffice')
plt.title('Budget vs Box Office Earnings')
plt.xlabel('Budget ($)')
plt.ylabel('Box Office Earnings ($)')
plt.show()

# Step 4: Advanced Analysis - Regression Model

# Prepare data for regression model
X = df[['Budget']]  # Features (Budget)
y = df['BoxOffice']  # Target (BoxOffice)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict box office earnings on the test data
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error of the model:", mse)

# Step 5: Visualizing the Regression Line
sns.scatterplot(x=X_test['Budget'], y=y_test, color='blue', label='Actual')
sns.lineplot(x=X_test['Budget'], y=y_pred, color='red', label='Predicted')
plt.title('Budget vs Box Office Regression Line')
plt.xlabel('Budget ($)')
plt.ylabel('Box Office Earnings ($)')
plt.legend()
plt.show()

# Step 6: Conclusion
# Based on the analysis, you can conclude things like:
# - How much does the budget affect box office earnings?
# - What genre is most popular among the movies?
# - How are movie ratings distributed?

