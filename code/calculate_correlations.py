import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import networkx as nx

# Load the dataset
file_path = 'universities_le_10_year.csv'  # Replace with the path to your CSV file
# file_path = 'universities_declining.csv'  
# file_path = 'universities_improving.csv'  
# file_path = 'universities_stable.csv'  

data = pd.read_csv(file_path)

# Select only the specified columns
columns_of_interest = ['rank', 'name', 'overall_score', 'teaching_score', 'research_score', 'citations_score',
                       'industry_income_score', 'international_outlook_score', 'number_students',
                       'student_staff_ratio', 'intl_students', 'year']
data = data[columns_of_interest]

# Function to calculate correlation and p-value

def calculate_correlations(data):
    correlations = {}
    columns = data.select_dtypes(include=['number']).columns
    for col1 in columns:
        for col2 in columns:
            if col1 != col2:
                corr, p_value = spearmanr(data[col1].dropna(), data[col2].dropna())
                correlations[(col1, col2)] = {'correlation': corr, 'p_value': p_value}
    return correlations

# Function to format the significance level
def format_significance(p_value):
    if p_value <= 0.01:
        return '***'
    elif p_value <= 0.05:
        return '**'
    elif p_value <= 0.1:
        return '*'
    else:
        return ''

# Calculate correlations
correlations = calculate_correlations(data)

# Group and print correlations per variable, ordered by strength
grouped_correlations = {}
for (col1, col2), values in correlations.items():
    if col1 not in grouped_correlations:
        grouped_correlations[col1] = []
    grouped_correlations[col1].append((col2, values['correlation'], values['p_value']))

# Sort and display grouped correlations
for variable, corr_list in grouped_correlations.items():
    print(f"\nCorrelations for {variable}:")
    sorted_corr_list = sorted(corr_list, key=lambda x: abs(x[1]), reverse=True)
    for col2, corr, p_value in sorted_corr_list:
        significance = format_significance(p_value)
        print(f"  {col2}: correlation = {corr:.2f}, p-value = {p_value:.2f} {significance}")

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
