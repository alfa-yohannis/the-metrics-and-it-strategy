import pandas as pd
from scipy.stats import theilslopes
import pymannkendall as mk

# Load dataset
df = pd.read_csv("universities2.csv")
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

# Trend classifier
def classify_trend(slope, mk_trend, p_value, alpha=0.05):
    if mk_trend == 'increasing' and p_value < alpha and slope > 0:
        return 'Declining'
    elif mk_trend == 'decreasing' and p_value < alpha and slope < 0:
        return 'Improving'
    else:
        return 'Stable'

# Analyze trend per university (unique per name)
results = []
for name, group in df.groupby('name'):
    group = group.dropna(subset=['year', 'rank'])
    if group['year'].nunique() >= 10:
        group = group.sort_values(by='year')
        slope, _, _, _ = theilslopes(group['rank'], group['year'])
        mk_result = mk.original_test(group['rank'])
        category = classify_trend(slope, mk_result.trend, mk_result.p)
        results.append({'name': name, 'trend_category': category})

# Create DataFrame with 1 row per university
trend_df = pd.DataFrame(results)

# Filter the original data to only include universities that have trend classification
filtered_df = df[df['name'].isin(trend_df['name'])]

# Merge to attach trend category to every year row
merged_df = filtered_df.merge(trend_df, on='name', how='left')

# Export to CSV by category
merged_df[merged_df['trend_category'] == 'Improving'].to_csv('universities_improving.csv', index=False)
merged_df[merged_df['trend_category'] == 'Declining'].to_csv('universities_declining.csv', index=False)
merged_df[merged_df['trend_category'] == 'Stable'].to_csv('universities_stable.csv', index=False)

# Sanity check: should match classification counts
print("Universities per trend category (unique names):")
print(trend_df['trend_category'].value_counts())

print("\nRows in output CSVs (multi-year rows per group):")
print(merged_df['trend_category'].value_counts())

# Display unique university name counts per category
print("\nUnique university counts per trend category:")
print(merged_df.groupby('trend_category')['name'].nunique())