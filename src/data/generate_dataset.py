"""
generate_dataset.py
-------------------
Generates a synthetic 2M row banking churn dataset
simulating real-world data from multiple source systems.

Run this ONCE to generate the raw dataset.
Output: data/raw/bank_churn_2M.csv

Author: Sonita
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ───────────────────────────────────────────────────────────
# Setting seeds ensures same dataset is generated every time
# Critical in prod for reproducibility and debugging
np.random.seed(42)
random.seed(42)

N = 2_000_000  # Number of customer records

print("🏦 Generating 2M row Bank Churn Dataset...")
print("=" * 50)

# ── 1. IDs & PII ──────────────────────────────────────────────────────────────
# In prod, PII columns are masked/tokenized before model training
# We include them here to practice data masking techniques
print("Step 1/11: Generating IDs & PII...")
customer_ids = np.arange(10_000_001, 10_000_001 + N)

surnames = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller",
            "Davis","Wilson","Moore","Taylor","Anderson","Thomas","Jackson",
            "White","Harris","Martin","Thompson","Young","King"]
surnames_arr = np.random.choice(surnames, N)

# Phone numbers in Indian format (simulating Indian bank)
phones = [f"+91-{np.random.randint(7000000000, 9999999999)}" for _ in range(N)]

# Email domains split realistically
emails = [f"cust{cid}@{'gmail.com' if r<0.5 else 'yahoo.com' if r<0.8 else 'outlook.com'}"
          for cid, r in zip(customer_ids, np.random.rand(N))]

# Partial Aadhaar — only last 4 digits stored (GDPR/compliance best practice)
aadhaar_last4 = np.random.randint(1000, 9999, N)

# ── 2. Geography ──────────────────────────────────────────────────────────────
# Geography affects churn — German customers churn more in this dataset
# This creates a meaningful feature for the model to learn
print("Step 2/11: Generating geography...")
geographies = ["France", "Germany", "Spain"]
geography = np.random.choice(geographies, N, p=[0.5, 0.25, 0.25])

cities_by_geo = {
    "France":  ["Paris","Lyon","Marseille","Toulouse","Nice","Nantes","Strasbourg",
                "Bordeaux","Lille","Rennes","Reims","Saint-Etienne","Toulon","Grenoble","Dijon"],
    "Germany": ["Berlin","Hamburg","Munich","Cologne","Frankfurt","Stuttgart","Dusseldorf",
                "Leipzig","Dortmund","Essen","Bremen","Dresden","Hanover","Nuremberg","Duisburg"],
    "Spain":   ["Madrid","Barcelona","Valencia","Seville","Zaragoza","Malaga","Murcia",
                "Palma","Las Palmas","Bilbao","Alicante","Cordoba","Valladolid","Vigo","Gijon"]
}
city = [np.random.choice(cities_by_geo[g]) for g in geography]

region_map = {"France": 1, "Germany": 2, "Spain": 3}
region_code = [region_map[g] for g in geography]

# ── 3. Demographics ───────────────────────────────────────────────────────────
# Gender has intentional inconsistencies (~2%) to simulate dirty real-world data
# Interview topic: how do you standardize categorical columns?
print("Step 3/11: Generating demographics...")
gender_raw = np.random.choice(["Male", "Female"], N, p=[0.55, 0.45])
gender = []
for g in gender_raw:
    r = random.random()
    if r < 0.01:   gender.append("male")   # lowercase inconsistency
    elif r < 0.02: gender.append("M")      # abbreviated inconsistency
    else:          gender.append(g)

# Age follows normal distribution; impossible values injected (~0.1%)
# Interview topic: business rule validation vs statistical validation
age = np.random.normal(38, 12, N).clip(18, 80).astype(int)
impossible_mask = np.random.rand(N) < 0.001
age[impossible_mask] = np.random.choice([10, 11, 12, 95, 100], impossible_mask.sum())

# Occupation — 25 categories (high cardinality)
# Interview topic: target encoding vs one-hot encoding
occupations = [
    "Engineer","Doctor","Teacher","Lawyer","Accountant","Nurse","Architect",
    "Pharmacist","Pilot","Manager","Banker","Consultant","Analyst","Designer",
    "Developer","Salesperson","Entrepreneur","Professor","Scientist","Journalist",
    "Retired","Student","Homemaker","Government Employee","Self-Employed"
]
occupation = np.random.choice(occupations, N)

# ── 4. Financial ──────────────────────────────────────────────────────────────
# Outliers injected in CreditScore and Salary (~0.5%)
# Interview topic: IQR vs Z-score for outlier detection
print("Step 4/11: Generating financial data...")
credit_score = np.random.normal(650, 100, N).clip(300, 900).astype(int)
outlier_mask = np.random.rand(N) < 0.005
credit_score[outlier_mask] = np.random.choice([150, 200, 950, 1000], outlier_mask.sum())

# 25% customers have zero balance — realistic for dormant accounts
balance = np.random.exponential(60000, N).clip(0, 500000).round(2)
balance[np.random.rand(N) < 0.25] = 0.0

estimated_salary = np.random.normal(80000, 30000, N).clip(10000, 300000).round(2)
outlier_sal = np.random.rand(N) < 0.005
estimated_salary[outlier_sal] = np.random.choice([1000, 2000, 800000, 900000], outlier_sal.sum())

savings_balance     = (balance * np.random.uniform(0.1, 0.5, N)).round(2)
investment_balance  = np.where(np.random.rand(N) < 0.4,
                               np.random.exponential(20000, N).clip(0, 200000).round(2), 0.0)
total_product_value = (balance + savings_balance + investment_balance).round(2)
reward_points       = np.random.randint(0, 50000, N)
overdraft_count     = np.random.poisson(0.5, N).clip(0, 10)
missed_payments     = np.random.poisson(0.3, N).clip(0, 8)

# ── 5. Banking Relationship ───────────────────────────────────────────────────
# These features are the strongest predictors of churn
# Interview topic: which features matter most and why?
print("Step 5/11: Generating banking relationship data...")
tenure         = np.random.randint(0, 11, N)
num_products   = np.random.choice([1,2,3,4], N, p=[0.4, 0.45, 0.10, 0.05])
has_cr_card    = np.random.choice([0,1], N, p=[0.3, 0.7])
is_active      = np.random.choice([0,1], N, p=[0.35, 0.65])
num_complaints = np.random.poisson(0.4, N).clip(0, 10)
num_trans_3m   = np.random.poisson(8, N).clip(0, 50)
avg_monthly_bal= (balance * np.random.uniform(0.8, 1.2, N)).round(2)
num_logins     = np.random.poisson(15, N).clip(0, 100)

loan_types = ["None","Home","Auto","Personal","Education","Business"]
loan_type  = np.random.choice(loan_types, N, p=[0.35, 0.20, 0.15, 0.15, 0.10, 0.05])

# Channel — how customer interacts with bank
channels = ["Mobile","Branch","Online","ATM","Phone"]
channel  = np.random.choice(channels, N, p=[0.35, 0.20, 0.25, 0.15, 0.05])

# Segment derived from salary — ordinal feature
# Interview topic: ordinal encoding vs one-hot for ordered categories
customer_segment = np.where(estimated_salary > 120000, "Premium",
                   np.where(estimated_salary > 50000,  "Regular", "Basic"))

contact_methods   = ["Email","SMS","Phone","App"]
preferred_contact = np.random.choice(contact_methods, N, p=[0.30, 0.25, 0.20, 0.25])

# ── 6. Temporal Features ──────────────────────────────────────────────────────
# Date features enable RFM analysis and time-based feature engineering
# Interview topic: why random train/test split is wrong for temporal data
print("Step 6/11: Generating temporal data...")
base_date = datetime(2024, 1, 1)

def rand_dates(start_back, end_back, size):
    """Generate random dates between start_back and end_back days ago."""
    offsets = np.random.randint(end_back, start_back, size)
    return [(base_date - timedelta(days=int(o))).strftime("%Y-%m-%d") for o in offsets]

account_open_date   = rand_dates(365*12, 365*1, N)
last_transaction_dt = rand_dates(365,    1,      N)
last_login_date     = rand_dates(180,    1,      N)
last_complaint_date = rand_dates(730,    30,     N)
record_timestamp    = rand_dates(30,     0,      N)

# Inject future dates (~0.3%) — data validation issue
# Interview topic: how do you catch and handle future-dated records?
last_transaction_dt = list(last_transaction_dt)
for i in np.where(np.random.rand(N) < 0.003)[0]:
    last_transaction_dt[i] = (base_date + timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")

# Inject mixed date formats (~1%) — common in real data pipelines
# Interview topic: robust date parsing strategies
account_open_date = list(account_open_date)
for i in np.where(np.random.rand(N) < 0.01)[0]:
    d = datetime.strptime(account_open_date[i], "%Y-%m-%d")
    account_open_date[i] = d.strftime("%d/%m/%Y")

# ── 7. Production Metadata ────────────────────────────────────────────────────
# In prod, data comes from multiple source systems
# Interview topic: how do you design a feature store for multi-source data?
print("Step 7/11: Generating production metadata...")
data_source          = np.random.choice(["CRM","Mobile","Branch","API"], N, p=[0.30,0.35,0.20,0.15])
data_quality_flag    = (np.random.rand(N) < 0.05).astype(int)
model_version        = np.random.choice(["v1.0","v1.1","v2.0"], N, p=[0.20,0.30,0.50])
previous_model_score = np.random.uniform(0, 1, N).round(4)

# CampaignResponseHistory — potential data leakage feature
# Interview topic: how do you detect and handle data leakage?
campaign_response    = np.random.choice([0,1], N, p=[0.7, 0.3])

risk_category = np.where(credit_score < 500, "High",
                np.where(credit_score < 650, "Medium", "Low"))

# ── 8. Target Variable: Churn ─────────────────────────────────────────────────
# Churn probability is driven by realistic business logic
# Interview topic: what features drive churn and why?
print("Step 8/11: Generating churn labels...")
churn_prob = (
    0.05                                                              # base rate
    + 0.15 * (age > 55).astype(float)                               # older customers churn more
    + 0.10 * (balance == 0).astype(float)                           # zero balance = disengaged
    + 0.12 * (num_complaints > 2).astype(float)                     # complaints predict churn
    + 0.08 * (is_active == 0).astype(float)                         # inactive customers churn more
    + 0.10 * (num_products == 1).astype(float)                      # single product = low loyalty
    - 0.05 * (tenure > 7).astype(float)                             # long tenure = loyal
    - 0.04 * (has_cr_card == 1).astype(float)                       # credit card = engaged
    + 0.06 * np.array([g == "Germany" for g in geography], dtype=float)  # geographic effect
    + 0.05 * (missed_payments > 2).astype(float)                    # financial stress
).clip(0.02, 0.85)

exited = (np.random.rand(N) < churn_prob).astype(int)

# Inject ~2% label noise — real world labels are never perfect
# Interview topic: how do you handle noisy labels?
noise_mask = np.random.rand(N) < 0.02
exited[noise_mask] = 1 - exited[noise_mask]

# ── 9. Assemble DataFrame ─────────────────────────────────────────────────────
print("Step 9/11: Assembling DataFrame...")
df = pd.DataFrame({
    # IDs & PII
    "CustomerId":                 customer_ids,
    "Surname":                    surnames_arr,
    "Phone":                      phones,
    "Email":                      emails,
    "AadhaarLast4":               aadhaar_last4,
    # Geography
    "Geography":                  geography,
    "City":                       city,
    "RegionCode":                 region_code,
    # Demographics
    "Gender":                     gender,
    "Age":                        age,
    "Occupation":                 occupation,
    # Financial
    "CreditScore":                credit_score,
    "Balance":                    balance,
    "SavingsAccountBalance":      savings_balance,
    "InvestmentAccountBalance":   investment_balance,
    "TotalProductValue":          total_product_value,
    "EstimatedSalary":            estimated_salary,
    "RewardPoints":               reward_points,
    "OverdraftCount":             overdraft_count,
    "MissedPayments":             missed_payments,
    # Banking relationship
    "CustomerSegment":            customer_segment,
    "RiskCategory":               risk_category,
    "Tenure":                     tenure,
    "NumOfProducts":              num_products,
    "HasCrCard":                  has_cr_card,
    "IsActiveMember":             is_active,
    "LoanType":                   loan_type,
    "Channel":                    channel,
    "PreferredContactMethod":     preferred_contact,
    "NumComplaints":              num_complaints,
    "NumTransactionsLast3Months": num_trans_3m,
    "AvgMonthlyBalance":          avg_monthly_bal,
    "NumLogins":                  num_logins,
    # Temporal
    "AccountOpenDate":            account_open_date,
    "LastTransactionDate":        last_transaction_dt,
    "LastLoginDate":              last_login_date,
    "LastComplaintDate":          last_complaint_date,
    "RecordTimestamp":            record_timestamp,
    # Production metadata
    "DataSource":                 data_source,
    "DataQualityFlag":            data_quality_flag,
    "ModelVersion":               model_version,
    "PreviousModelScore":         previous_model_score,
    "CampaignResponseHistory":    campaign_response,
    # Target
    "Exited":                     exited,
})

# ── 10. Inject Missing Values ─────────────────────────────────────────────────
# Missing values injected at different rates per column
# Interview topic: MCAR vs MAR vs MNAR — which is this?
print("Step 10/11: Injecting missing values...")
missing_config = {
    "Balance":               0.03,  # 3% missing
    "Occupation":            0.05,  # 5% missing — highest, common in CRM data
    "LastTransactionDate":   0.04,  # 4% missing
    "CreditScore":           0.02,  # 2% missing
    "City":                  0.03,  # 3% missing
    "EstimatedSalary":       0.02,  # 2% missing
    "NumLogins":             0.03,  # 3% missing
}
for col, rate in missing_config.items():
    df.loc[np.random.rand(len(df)) < rate, col] = np.nan

# ── 11. Inject Duplicates & Save ──────────────────────────────────────────────
# 500 duplicate rows simulate real-world ETL issues
# Interview topic: how do you detect and remove duplicates in prod pipeline?
print("Step 11/11: Injecting duplicates & saving...")
dup_rows = df.sample(500, random_state=42)
df = pd.concat([df, dup_rows], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to data/raw — raw data is never modified after saving
output_path = "data/raw/bank_churn_2M.csv"
df.to_csv(output_path, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("✅ DATASET GENERATED SUCCESSFULLY!")
print("=" * 50)
print(f"📁 Saved to: {output_path}")
print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"🎯 Churn rate: {df['Exited'].mean():.2%}")
print(f"\n📋 Columns: {df.shape[1]} total")
print(f"   • Numerical:   {df.select_dtypes(include=np.number).shape[1]}")
print(f"   • Categorical: {df.select_dtypes(include='object').shape[1]}")
print(f"\n⚠️  Data Quality Issues Injected:")
print(f"   • Missing values in 7 columns")
print(f"   • 500 duplicate rows")
print(f"   • Outliers in CreditScore & EstimatedSalary")
print(f"   • Impossible ages (~0.1%)")
print(f"   • Inconsistent gender labels (~2%)")
print(f"   • Future dated records (~0.3%)")
print(f"   • Mixed date formats (~1%)")
print(f"   • Label noise (~2%)")
print(f"\n🏦 Key Cardinalities:")
for col in ["Geography","City","Occupation","LoanType","Channel","CustomerSegment"]:
    print(f"   • {col}: {df[col].nunique()} unique values")