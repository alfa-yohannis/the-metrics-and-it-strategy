import pandas as pd
from scipy.stats import theilslopes
import pymannkendall as mk

# Load dataset
df = pd.read_csv("universities2.csv")

# Ensure numeric types
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

# Classification function
def classify_trend(slope, mk_trend, p_value, alpha=0.05):
    if mk_trend == 'increasing' and p_value < alpha and slope > 0:
        return 'Declining'
    elif mk_trend == 'decreasing' and p_value < alpha and slope < 0:
        return 'Improving'
    else:
        return 'Stable'

# Container for results
results = []

for name, group in df.groupby('name'):
    group = group.dropna(subset=['year', 'rank'])
    if group['year'].nunique() >= 10:
        group = group.sort_values(by='year')

        slope, intercept, lower, upper = theilslopes(group['rank'], group['year'])
        mk_result = mk.original_test(group['rank'])
        trend_category = classify_trend(slope, mk_result.trend, mk_result.p)

        results.append({
            'name': name,
            'n_years': group['year'].nunique(),
            'theil_sen_slope_rank': slope,
            'mk_trend': mk_result.trend,
            'mk_p_value': mk_result.p,
            'trend_category': trend_category
        })

# Compile DataFrame
result_df = pd.DataFrame(results)
result_df.sort_values(by='theil_sen_slope_rank', inplace=True)

# Print counts per category
category_counts = result_df['trend_category'].value_counts()
print("\nTrend category counts:")
for cat in ['Improving', 'Declining', 'Stable']:
    print(f"{cat}: {category_counts.get(cat, 0)}")

# Show top 10 examples for each
print("\nTop 10 improving universities:")
print(result_df[result_df['trend_category'] == 'Improving'].head(10))

print("\nTop 10 declining universities:")
print(result_df[result_df['trend_category'] == 'Declining'].tail(10))

print("\nTop 10 stable universities (lowest absolute slope):")
print(result_df[result_df['trend_category'] == 'Stable']
      .reindex(result_df[result_df['trend_category'] == 'Stable']
      .theil_sen_slope_rank.abs().sort_values().index).head(10))
