import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


file_path = '/Users/aneeshkrishna/Downloads/Hiroshima Data Copy.xlsx'

df = pd.read_excel(file_path, engine='openpyxl')

# Convert FinishOnUtc to datetime if not already
df['FinishOnUtc'] = pd.to_datetime(df['FinishOnUtc'])

# Function to label COVID period
def covid_period(date):
    if date.year < 2020:
        return 'Pre-COVID'
    elif date.year in [2020, 2021]:
        return 'COVID'
    else:
        return 'Post-COVID'

# Apply function to create new column
df['COVID_Period'] = df['FinishOnUtc'].apply(covid_period)

# Check distribution
print(df['COVID_Period'].value_counts())

summary = df.groupby('COVID_Period')[
    ['Age', 'TimeInAnotherCountry', 'InterestInInternationalOrInterculturalLearning']
].mean().round(2)

print(summary)

gender_summary = df.groupby(['COVID_Period', 'Gender'])[
    ['Age', 'TimeInAnotherCountry', 'InterestInInternationalOrInterculturalLearning']
].mean().round(2)

print(gender_summary)

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='InterestInInternationalOrInterculturalLearning', hue='COVID_Period', bins=10, kde=True, multiple='stack')
plt.title('Distribution of Interest in International/Intercultural Learning by COVID Period')
plt.xlabel('Interest Level (1-10)')
plt.ylabel('Number of Students')
plt.tight_layout()
#plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='TimeInAnotherCountry', hue='COVID_Period', bins=15, kde=True, multiple='stack')
plt.title('Distribution of Time Spent Abroad by COVID Period')
plt.xlabel('Time Abroad (Months)')
plt.ylabel('Number of Students')
plt.tight_layout()
#plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='COVID_Period', bins=12, kde=True, multiple='stack')
plt.title('Age Distribution by COVID Period')
plt.xlabel('Age')
plt.ylabel('Number of Students')
plt.tight_layout()
#plt.show()

# Interest in International or Intercultural Learning
interest = df["InterestInInternationalOrInterculturalLearning"]
print("Interest in International/Intercultural Learning:")
print(f"  Mean: {interest.mean():.2f}")
print(f"  Median: {interest.median():.2f}")
print(f"  Std Dev: {interest.std():.2f}")
print(f"  Min: {interest.min():.2f}, Max: {interest.max():.2f}\n")

# Time in Another Country
time_abroad = df["TimeInAnotherCountry"]
print("Time Spent in Another Country (Years):")
print(f"  Mean: {time_abroad.mean():.2f}")
print(f"  Median: {time_abroad.median():.2f}")
print(f"  Std Dev: {time_abroad.std():.2f}")
print(f"  Min: {time_abroad.min():.2f}, Max: {time_abroad.max():.2f}\n")

# Age
age = df["Age"]
print("Age:")
print(f"  Mean: {age.mean():.2f}")
print(f"  Median: {age.median():.2f}")
print(f"  Std Dev: {age.std():.2f}")
print(f"  Min: {age.min():.2f}, Max: {age.max():.2f}\n")

# Correlation matrix
correlations = df[['Age', 'TimeInAnotherCountry', 'InterestInInternationalOrInterculturalLearning']].corr()
print(correlations)

# Set the style
sns.set(style="whitegrid")

# Define gender labels for legend
gender_labels = {1.0: 'Male', 2.0: 'Female', 3.0: 'Other'}

# Set up variables to analyze
plot_pairs = [
    ('Age', 'InterestInInternationalOrInterculturalLearning'),
    ('TimeInAnotherCountry', 'InterestInInternationalOrInterculturalLearning'),
    ('Age', 'TimeInAnotherCountry')
]

# Create scatterplots with regression lines, faceted by COVID_Period
for x, y in plot_pairs:
    g = sns.lmplot(
        data=df,
        x=x, y=y,
        col='COVID_Period',
        hue='Gender',
        palette='Set1',
        scatter_kws={'alpha':0.5},
        height=4, aspect=1,
        markers=["o", "s", "D"]
    )
    
    g.set_axis_labels(x, y)
    g.set_titles("{col_name}")
    g.add_legend(title="Gender", label_order=[1.0, 2.0, 3.0])
    for t in g._legend.texts:
        t.set_text(gender_labels.get(float(t.get_text()), t.get_text()))

    plt.subplots_adjust(top=0.85)
    plt.suptitle(f"{y} vs {x} by COVID Period and Gender", fontsize=14)
    plt.show()
    
    # Variables to analyze
analysis_pairs = [
    ('Age', 'InterestInInternationalOrInterculturalLearning'),
    ('TimeInAnotherCountry', 'InterestInInternationalOrInterculturalLearning'),
    ('Age', 'TimeInAnotherCountry')
]

# Gender labels
gender_labels = {1.0: 'Male', 2.0: 'Female', 3.0: 'Other'}

# Function to describe strength of correlation
def describe_corr(r):
    if np.isnan(r):
        return "insufficient data"
    abs_r = abs(r)
    if abs_r < 0.1:
        return "no or negligible correlation"
    elif abs_r < 0.3:
        return "weak correlation"
    elif abs_r < 0.6:
        return "moderate correlation"
    else:
        return "strong correlation"

# Analyze and describe
for x, y in analysis_pairs:
    print(f"\n### Relationship: {x} vs. {y}\n")
    for period in df['COVID_Period'].unique():
        for gender in df['Gender'].unique():
            subset = df[(df['COVID_Period'] == period) & (df['Gender'] == gender)]
            if len(subset) >= 10:
                r, p = np.corrcoef(subset[x], subset[y])[0, 1], None
                slope, intercept, r_val, p_val, std_err = linregress(subset[x], subset[y])
                desc = describe_corr(r_val)
                print(f"- {period}, {gender_labels.get(gender, 'Unknown')}:")
                print(f"  Pearson r = {r_val:.2f} ({desc}), slope = {slope:.2f}, p = {p_val:.4f}")
            else:
                print(f"- {period}, {gender_labels.get(gender, 'Unknown')}: Not enough data")

