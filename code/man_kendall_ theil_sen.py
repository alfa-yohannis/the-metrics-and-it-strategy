import pandas as pd
from scipy.stats import theilslopes
import pymannkendall as mk

# Load dataset
df = pd.read_csv("universities2.csv")

# Ensure numeric types
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

# Container for results
results = []

for name, group in df.groupby('name'):
    group = group.dropna(subset=['year', 'rank'])
    if group['year'].nunique() >= 10:
        # Sort by year
        group = group.sort_values(by='year')

        # Theil-Sen slope
        slope, intercept, lower, upper = theilslopes(group['rank'], group['year'])

        # Mann-Kendall test
        mk_result = mk.original_test(group['rank'])

        results.append({
    'name': name,
    'n_years': group['year'].nunique(),
    'theil_sen_slope_rank': slope,
    'intercept': intercept,
    'lower_slope': lower,
    'upper_slope': upper,
    'mk_trend': mk_result.trend,
    'mk_p_value': mk_result.p,
    'mk_tau': mk_result.Tau,
    'mk_s': mk_result.s,
    'mk_var_s': mk_result.var_s,
    'mk_z': mk_result.z
})


# Convert to DataFrame and sort
result_df = pd.DataFrame(results)
result_df.sort_values(by='theil_sen_slope_rank', inplace=True)

# Show top and bottom results
print("Top 10 improving universities:")
print(result_df.head(10))

print("\nTop 10 declining universities:")
print(result_df.tail(10))
