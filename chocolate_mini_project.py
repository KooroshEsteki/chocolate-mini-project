import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("flavors_of_cacao.csv")

df.columns = (
    df.columns
    .str.replace("\xa0", " ", regex=False)
    .str.replace("\n", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("First five rows:")
print(df.head())

df = df.replace(r"^\s*$", np.nan, regex=True)

print("\nMissing values before deleting:")
print(df.isnull().sum())

df_clean = df.dropna().copy()

print("\nOriginal dataset shape:", df.shape)
print("Cleaned dataset shape:", df_clean.shape)

print("\nNumber of tuples in the dataset:", df.shape[0])

unique_companies = df["Company (Maker-if known)"].nunique()
print("Number of unique company names:", unique_companies)

reviews_2013 = df[df["Review Date"] == 2013].shape[0]
print("Number of reviews made in 2013:", reviews_2013)

missing_bean_type = df["Bean Type"].isnull().sum()
print("Missing values in Bean Type column:", missing_bean_type)

plt.figure(figsize=(8, 5))
plt.hist(df_clean["Rating"], bins=20, edgecolor="black")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.title("Distribution of Chocolate Bar Ratings")
plt.show()

df_clean["Cocoa Percent Numeric"] = (
    df_clean["Cocoa Percent"]
    .str.replace("%", "", regex=False)
    .astype(float)
)

print("\nConverted Cocoa Percent values:")
print(df_clean[["Cocoa Percent", "Cocoa Percent Numeric"]].head())

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

scaler = MinMaxScaler()
df_clean["Rating Normalized"] = scaler.fit_transform(df_clean[["Rating"]])

print("\nNormalized rating values:")
print(df_clean[["Rating", "Rating Normalized"]].head())

company_avg_rating = (
    df_clean
    .groupby("Company (Maker-if known)")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

print("\nCompanies ordered by average rating:")
print(company_avg_rating)

encoded_df = pd.get_dummies(
    df_clean,
    columns=["Company (Maker-if known)", "Company Location"],
    drop_first=False
)

print("\nOriginal cleaned dataset shape:", df_clean.shape)
print("Encoded dataset shape:", encoded_df.shape)

df_clean.to_csv("cleaned_chocolate_dataset.csv", index=False)
encoded_df.to_csv("encoded_chocolate_dataset.csv", index=False)

print("\nCleaned and encoded datasets saved successfully.")
