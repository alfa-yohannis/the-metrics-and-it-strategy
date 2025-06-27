import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import networkx as nx

# ------------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------------
file_path = 'universities_le_10_year.csv'         # <- path to your CSV file
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
# 3. Limit the analysis to one university
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
# 6. Build graph and compute node influence (no double-counting)
# ------------------------------------------------------------------
# Calculate total influence for node sizes
threshold = 0.0 # Minimum absolute correlation value to draw an edge
graph = nx.Graph()
node_influence = {}

# Add nodes and edges based on correlations
for (col1, col2), values in correlations.items():
    if abs(values['correlation']) >= threshold and values['p_value'] <= 0.1:  # Only significant correlations
        weight = abs(values['correlation'])
        print(f"Adding edge: {col1} - {col2} with weight {abs(weight):.2f}")
        graph.add_edge(col1, col2, weight=weight)
        node_influence[col1] = node_influence.get(col1, 0) + weight 
        # node_influence[col2] = node_influence.get(col2, 0) + weight

# add self-influence to each node
for node in node_influence:
    node_influence[node] += 1
    
# Normalize node influence for sizes
max_influence = max(node_influence.values())
min_influence = min(node_influence.values())
normalized_influence = {node: (influence - min_influence) / (max_influence - min_influence) for node, influence in node_influence.items()}

# Print variables ordered by total influence
print("\nVariables ordered by total influence:")
sorted_influence = sorted(node_influence.items(), key=lambda x: x[1], reverse=True)
for variable, influence in sorted_influence:
    influence_with_self = influence + normalized_influence.get(variable, 0)  # Add self-influence for clarity
    print(f"{variable}: unnormalized influence = {influence:.2f}, normalized influence = {normalized_influence[variable]:.2f}")
