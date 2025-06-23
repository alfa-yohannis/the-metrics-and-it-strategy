import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import networkx as nx

# ------------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------------
file_path = 'universities2.csv'         # <- path to your CSV file
# file_path = 'universities_declining.csv'
# file_path = 'universities_improving.csv'
# file_path = 'universities_stable.csv'

data = pd.read_csv(file_path)

# ------------------------------------------------------------------
# 2. Keep only the columns we care about
# ------------------------------------------------------------------
columns_of_interest = [
    'rank', 'name', 'overall_score', 'teaching_score', 'research_score',
    'citations_score', 'industry_income_score', 'international_outlook_score',
    'number_students', 'student_staff_ratio', 'intl_students', 'year'
]
data = data[columns_of_interest]

# ------------------------------------------------------------------
# 3. NEW LINE: limit the analysis to University of Cambridge only
# ------------------------------------------------------------------
data = data[data['name'] == 'Wuhan University']

# ------------------------------------------------------------------
# 4. Correlation helpers
# ------------------------------------------------------------------
def calculate_correlations(df):
    """Return {(col1, col2): {'correlation': rho, 'p_value': p}} for numeric pairs."""
    correlations = {}
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col1 in numeric_cols:
        for col2 in numeric_cols:
            if col1 != col2:
                rho, p = spearmanr(df[col1].dropna(), df[col2].dropna())
                correlations[(col1, col2)] = {'correlation': rho, 'p_value': p}
    return correlations

def format_significance(p):
    return '***' if p <= 0.01 else '**' if p <= 0.05 else '*' if p <= 0.1 else ''

# ------------------------------------------------------------------
# 5. Compute and print correlations grouped by variable
# ------------------------------------------------------------------
correlations = calculate_correlations(data)
grouped = {}
for (c1, c2), vals in correlations.items():
    grouped.setdefault(c1, []).append((c2, vals['correlation'], vals['p_value']))

for var, lst in grouped.items():
    print(f"\nCorrelations for {var}:")
    for c2, rho, p in sorted(lst, key=lambda x: abs(x[1]), reverse=True):
        print(f"  {c2}: correlation = {rho:.2f}, p-value = {p:.2f} {format_significance(p)}")

# ------------------------------------------------------------------
# 6. Build graph and compute node influence
# ------------------------------------------------------------------
threshold = 0.1  # absolute rho threshold
G = nx.Graph()
influence = {}

for (c1, c2), vals in correlations.items():
    if abs(vals['correlation']) >= threshold and vals['p_value'] <= 0.05:
        G.add_edge(c1, c2, weight=abs(vals['correlation']))
        influence[c1] = influence.get(c1, 0) + abs(vals['correlation'])
        influence[c2] = influence.get(c2, 0) + abs(vals['correlation'])

if influence:  # avoid zero‐division if the graph is empty
    max_inf, min_inf = max(influence.values()), min(influence.values())
    norm_inf = {n: (v - min_inf) / (max_inf - min_inf) for n, v in influence.items()}
else:
    norm_inf = {}

print("\nVariables ordered by total influence:")
for var, inf in sorted(influence.items(), key=lambda x: x[1], reverse=True):
    print(f"{var}: unnormalized influence = {inf:.2f}, normalized influence = {norm_inf[var]:.2f}")
