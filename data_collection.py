import pandas as pd
data = pd.read_csv(
    "bbc-news-data.csv",
     sep="\t"
)
filtered_data = data[data["category"].isin(["sport", "politics"])]
filtered_data["text"] = filtered_data["title"] + " " + filtered_data["content"]
final_data = filtered_data[["category", "text"]]
final_data.to_csv("sports_politics_dataset.csv", index=False)
print("\nCleaned dataset saved as 'sports_politics_dataset.csv'")