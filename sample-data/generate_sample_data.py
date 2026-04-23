"""
Generate a sample hiring dataset with intentional bias for demo purposes.
Run: python generate_sample_data.py
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

genders = np.random.choice(['Male', 'Female', 'Non-binary'], n, p=[0.52, 0.44, 0.04])
ages = np.random.randint(22, 60, n)
experience = np.random.randint(0, 20, n)
education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.2, 0.45, 0.25, 0.1])
interview_score = np.random.randint(40, 100, n)
gpa = np.round(np.random.uniform(2.5, 4.0, n), 2)

# Intentionally biased hiring — males favored
base_prob = (
    0.3
    + (experience / 20) * 0.25
    + (interview_score / 100) * 0.25
    + (gpa - 2.5) / 1.5 * 0.1
    + np.where(education == 'PhD', 0.1,
      np.where(education == 'Master', 0.07,
      np.where(education == 'Bachelor', 0.04, 0.0)))
)

# Add gender bias
bias = np.where(genders == 'Male', 0.12,
       np.where(genders == 'Female', -0.08, -0.04))
prob = np.clip(base_prob + bias + np.random.normal(0, 0.05, n), 0.05, 0.95)
hired = (np.random.uniform(0, 1, n) < prob).astype(int)

df = pd.DataFrame({
    'age': ages,
    'gender': genders,
    'education': education,
    'years_experience': experience,
    'gpa': gpa,
    'interview_score': interview_score,
    'hired': hired
})

df.to_csv('sample_hiring.csv', index=False)
print(f"Generated sample_hiring.csv with {n} rows")
print(f"Hire rate by gender:")
print(df.groupby('gender')['hired'].mean().round(3))
print(f"\nOverall hire rate: {df['hired'].mean():.3f}")