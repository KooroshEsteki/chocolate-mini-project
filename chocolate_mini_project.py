import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Step 1 and 2: Read the dataset
df = pd.read_csv("flavors_of_cacao.csv")

# Clean column names
df.columns = (
    df.columns
    .str.replace("\xa0", " ", regex=False)
    .str.replace("\n", " ", regex=False)
    .str.strip()
)

print("First five rows:")
print(df.head())

# Step 2: Check missing values
df = df.replace(r'^\s*$', np.nan, regex=True)

print("\nMissing values before deleting:")
print(df.isnull().sum())

# Delete rows with missing values
df_clean = df.dropna()

print("\nOriginal dataset shape:", df.shape)
print("Cleaned dataset shape:", df_clean.shape)

# Step 3: Exploring the dataset
print("\nNumber of tuples in the dataset:", df.shape[0])

unique_companies = df["Company (Maker-if known)"].nunique()
print("Number of unique company names:", unique_companies)

reviews_2013 = df[df["Review Date"] == 2013].shape[0]
print("Number of reviews made in 2013:", reviews_2013)

missing_bean_type = df["Bean Type"].isnull().sum()
print("Missing values in Bean Type column:", missing_bean_type)

# Step 4: Histogram of Rating
plt.figure(figsize=(8, 5))
plt.hist(df_clean["Rating"], bins=20, edgecolor="black")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.title("Distribution of Chocolate Bar Ratings")
plt.show()

print("""
Comment on histogram:
Most chocolate ratings are concentrated between 3.0 and 3.75.
Very low ratings are rare, and very high ratings above 4.0 are also less common.
This means most chocolate bars received moderate to good ratings.
""")

# Step 4: Convert Cocoa Percent column
df_clean = df_clean.copy()

df_clean["Cocoa Percent Numeric"] = (
    df_clean["Cocoa Percent"]
    .str.replace("%", "", regex=False)
    .astype(float)
)

print("\nConverted Cocoa Percent values:")
print(df_clean[["Cocoa Percent", "Cocoa Percent Numeric"]].head())

# Step 5: Cocoa Percent vs Rating
plt.figure(figsize=(8, 5))
plt.scatter(
    df_clean["Cocoa Percent Numeric"],
    df_clean["Rating"],
    alpha=0.1
)
plt.xlabel("Cocoa Percent")
plt.ylabel("Rating")
plt.title("Cocoa Percent vs Rating")
plt.show()

correlation = df_clean["Cocoa Percent Numeric"].corr(df_clean["Rating"])
print("\nCorrelation between Cocoa Percent and Rating:", correlation)

print("""
Comment on scatter plot:
Higher cocoa percentage does not clearly guarantee a higher rating.
Most highly rated chocolates are around the middle cocoa range, especially near 65% to 75%.
The relationship appears weak rather than strongly positive.
""")

# Step 6: Normalize Rating column
scaler = MinMaxScaler()
df_clean["Rating Normalized"] = scaler.fit_transform(df_clean[["Rating"]])

print("\nNormalized rating values:")
print(df_clean[["Rating", "Rating Normalized"]].head())

# Step 7: Average rating by company
company_avg_rating = (
    df_clean
    .groupby("Company (Maker-if known)")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

print("\nCompanies ordered by average rating:")
print(company_avg_rating)

# Step 8: Encoding categorical columns
encoded_df = pd.get_dummies(
    df_clean,
    columns=["Company (Maker-if known)", "Company Location"],
    drop_first=False
)

print("\nOriginal cleaned dataset shape:", df_clean.shape)
print("Encoded dataset shape:", encoded_df.shape)

# Save output files
df_clean.to_csv("cleaned_chocolate_dataset.csv", index=False)
encoded_df.to_csv("encoded_chocolate_dataset.csv", index=False)

print("\nCleaned and encoded datasets saved successfully.")
