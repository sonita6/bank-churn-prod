"""
clean_data.py
-------------
Cleans the raw bank churn dataset based on EDA findings.

Input:  data/raw/bank_churn_2M.csv
Output: data/processed/bank_churn_cleaned.csv

Run this script inside Docker via terminal:
docker exec -it bank_churn_jupyter python src/data/clean_data.py

Author: Sonita
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

print("🧹 Starting Data Cleaning Pipeline...")
print("=" * 50)

# ── Step 1: Load Raw Data ─────────────────────────────────────────────────────
# Always load from data/raw — raw data is NEVER modified
# Cleaned data goes to data/processed
print("\nStep 1: Loading raw data...")
start = time.time()
df = pd.read_csv('/app/data/raw/bank_churn_2M.csv')
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Memory before cleaning: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB")

# ── Step 2: Remove Duplicates ─────────────────────────────────────────────────
# EDA found 500 exact duplicate rows
# Keep first occurrence, drop the rest
# Interview topic: in prod, dedup should happen at pipeline level, not model level
print("\nStep 2: Removing duplicates...")
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"✅ Removed {before - after:,} duplicate rows")
print(f"   Rows remaining: {after:,}")

# ── Step 3: Fix Impossible Age Values ─────────────────────────────────────────
# EDA found Age values like 10, 11, 12 (too young for bank account)
# and 95, 100 (extremely unlikely)
# Business rule: valid age range is 18-85
# Interview topic: business rule validation vs statistical validation
print("\nStep 3: Fixing impossible Age values...")
invalid_age_mask = (df['Age'] < 18) | (df['Age'] > 85)
invalid_count = invalid_age_mask.sum()

# Replace with median age — safe imputation for small number of invalid values
median_age = df.loc[~invalid_age_mask, 'Age'].median()
df.loc[invalid_age_mask, 'Age'] = median_age
print(f"✅ Fixed {invalid_count:,} invalid age values → replaced with median ({median_age:.0f})")

# ── Step 4: Standardize Gender ────────────────────────────────────────────────
# EDA found 4 unique values: Male, Female, male, M
# Standardize to Male/Female only
# Interview topic: always lowercase + strip before mapping to catch edge cases
print("\nStep 4: Standardizing Gender column...")
before_unique = df['Gender'].nunique()
df['Gender'] = df['Gender'].str.strip().str.lower()
df['Gender'] = df['Gender'].map({'male': 'Male', 'm': 'Male', 'female': 'Female'})
after_unique = df['Gender'].nunique()
print(f"✅ Gender standardized: {before_unique} → {after_unique} unique values")
print(f"   Distribution: {df['Gender'].value_counts().to_dict()}")

# ── Step 5: Handle Missing Values ─────────────────────────────────────────────
# Different strategy per column based on EDA findings
# Interview topic: MCAR vs MAR vs MNAR affects imputation strategy
print("\nStep 5: Handling missing values...")

# 5a. LoanType — 34.97% missing means customer has no loan
# Treat NaN as 'No Loan' category — domain knowledge driven
df['LoanType'] = df['LoanType'].fillna('No Loan')
print(f"   ✅ LoanType: NaN → 'No Loan' (domain knowledge)")

# 5b. Occupation — 4.97% missing
# Use 'Unknown' — we don't know occupation, don't want to assume
df['Occupation'] = df['Occupation'].fillna('Unknown')
print(f"   ✅ Occupation: NaN → 'Unknown'")

# 5c. City — 3% missing
# Use mode (most frequent city per geography)
# Interview topic: group-wise imputation is more accurate than global mode
for geo in df['Geography'].unique():
    mode_city = df.loc[df['Geography'] == geo, 'City'].mode()[0]
    mask = (df['Geography'] == geo) & (df['City'].isna())
    df.loc[mask, 'City'] = mode_city
print(f"   ✅ City: NaN → mode city per Geography")

# 5d. CreditScore — 2.01% missing
# Use median — credit score is slightly skewed, median is more robust
median_credit = df['CreditScore'].median()
df['CreditScore'] = df['CreditScore'].fillna(median_credit)
print(f"   ✅ CreditScore: NaN → median ({median_credit:.0f})")

# 5e. Balance — 3% missing
# Use median — balance is highly skewed (many zeros), median is safer
median_balance = df['Balance'].median()
df['Balance'] = df['Balance'].fillna(median_balance)
print(f"   ✅ Balance: NaN → median ({median_balance:.0f})")

# 5f. EstimatedSalary — 2.01% missing
# Use median per CustomerSegment — segment-wise is more accurate
for seg in df['CustomerSegment'].unique():
    median_sal = df.loc[df['CustomerSegment'] == seg, 'EstimatedSalary'].median()
    mask = (df['CustomerSegment'] == seg) & (df['EstimatedSalary'].isna())
    df.loc[mask, 'EstimatedSalary'] = median_sal
print(f"   ✅ EstimatedSalary: NaN → median per CustomerSegment")

# 5g. NumLogins — 2.98% missing
# Use median — integer feature, median preserves data type
median_logins = df['NumLogins'].median()
df['NumLogins'] = df['NumLogins'].fillna(median_logins)
print(f"   ✅ NumLogins: NaN → median ({median_logins:.0f})")

# 5h. LastTransactionDate — 4.02% missing
# Use median date — customer likely has a transaction around average date
median_date = pd.to_datetime(
    df['LastTransactionDate'].dropna()
).quantile(0.5, interpolation='midpoint')
df['LastTransactionDate'] = df['LastTransactionDate'].fillna(str(median_date.date()))
print(f"   ✅ LastTransactionDate: NaN → median date ({median_date.date()})")

# Verify no missing values remain
remaining_missing = df.isnull().sum().sum()
print(f"\n   Total missing values remaining: {remaining_missing:,}")

# ── Step 6: Fix Date Formats ──────────────────────────────────────────────────
# EDA found mixed date formats in AccountOpenDate (YYYY-MM-DD and DD/MM/YYYY)
# Also found future dates in LastTransactionDate
# Interview topic: always use dayfirst=False with explicit format parsing
print("\nStep 6: Fixing date columns...")

def parse_mixed_dates(date_series):
    """
    Parse dates that may have mixed formats.
    Tries standard format first, falls back to dayfirst format.
    In prod — use explicit format validation, not inference.
    """
    parsed = pd.to_datetime(date_series, infer_datetime_format=True, 
                            dayfirst=False, errors='coerce')
    # Try dayfirst for failed parses
    failed_mask = parsed.isna() & date_series.notna()
    if failed_mask.sum() > 0:
        parsed[failed_mask] = pd.to_datetime(
            date_series[failed_mask], dayfirst=True, errors='coerce'
        )
    return parsed

date_cols = ['AccountOpenDate', 'LastTransactionDate', 
             'LastLoginDate', 'LastComplaintDate', 'RecordTimestamp']

for col in date_cols:
    df[col] = parse_mixed_dates(df[col])
    print(f"   ✅ {col}: parsed to datetime")

# Fix future dates in LastTransactionDate
# Any date after RecordTimestamp is invalid
reference_date = pd.Timestamp('2024-01-01')
future_mask = df['LastTransactionDate'] > reference_date
future_count = future_mask.sum()
df.loc[future_mask, 'LastTransactionDate'] = reference_date
print(f"   ✅ Fixed {future_count:,} future LastTransactionDate values")

# ── Step 7: Cap Outliers ──────────────────────────────────────────────────────
# Use IQR capping (Winsorization) — replaces outliers with boundary values
# Interview topic: why capping over removal?
# Removal loses data; capping preserves row count while limiting extreme influence
print("\nStep 7: Capping outliers using IQR method...")

def cap_outliers_iqr(df, col, multiplier=1.5):
    """
    Cap outliers at IQR boundaries (Winsorization).
    Values below lower bound → lower bound
    Values above upper bound → upper bound
    Preserves row count unlike outlier removal.
    """
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    
    before_count = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    return before_count, lower, upper

# Cap CreditScore — valid range should be 300-900
# Use business rules instead of IQR for domain-constrained features
cs_outliers = ((df['CreditScore'] < 300) | (df['CreditScore'] > 900)).sum()
df['CreditScore'] = df['CreditScore'].clip(300, 900)
print(f"   ✅ CreditScore: capped to [300, 900] — {cs_outliers:,} values fixed")

# Cap EstimatedSalary using IQR
sal_count, sal_lower, sal_upper = cap_outliers_iqr(df, 'EstimatedSalary')
print(f"   ✅ EstimatedSalary: capped to [{sal_lower:.0f}, {sal_upper:.0f}] — {sal_count:,} values fixed")

# Cap NumLogins using IQR
login_count, login_lower, login_upper = cap_outliers_iqr(df, 'NumLogins')
print(f"   ✅ NumLogins: capped to [{login_lower:.0f}, {login_upper:.0f}] — {login_count:,} values fixed")

# Note: Balance outliers NOT capped — high balance = high value customer
# Interview topic: domain knowledge overrides statistical rules
print(f"   ℹ️  Balance outliers kept — high balance = legitimate high-value customers")

# ── Step 8: Optimize Memory ───────────────────────────────────────────────────
# Reduce memory from 2.7GB by downcasting numeric types
# Interview topic: why is memory optimization critical in prod?
# Less memory = faster processing, lower cloud costs, fits in RAM
print("\nStep 8: Optimizing memory usage...")

# Downcast integers
int_cols = ['Age', 'CreditScore', 'RewardPoints', 'Tenure', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'NumComplaints', 
            'NumTransactionsLast3Months', 'NumLogins', 'MissedPayments',
            'OverdraftCount', 'DataQualityFlag', 'CampaignResponseHistory',
            'RegionCode', 'AadhaarLast4', 'Exited']

for col in int_cols:
    df[col] = pd.to_numeric(df[col], downcast='integer')

# Downcast floats
float_cols = ['Balance', 'SavingsAccountBalance', 'InvestmentAccountBalance',
              'TotalProductValue', 'EstimatedSalary', 'AvgMonthlyBalance',
              'PreviousModelScore']

for col in float_cols:
    df[col] = pd.to_numeric(df[col], downcast='float')

# Convert low cardinality categoricals to category dtype
# Saves significant memory for repeated string values
cat_cols = ['Geography', 'City', 'Gender', 'Occupation', 'CustomerSegment',
            'RiskCategory', 'LoanType', 'Channel', 'PreferredContactMethod',
            'DataSource', 'ModelVersion']

for col in cat_cols:
    df[col] = df[col].astype('category')

memory_after = df.memory_usage(deep=True).sum() / 1024**2
print(f"✅ Memory optimized!")
print(f"   Before: 2,713 MB")
print(f"   After:  {memory_after:.1f} MB")
print(f"   Saved:  {2713 - memory_after:.1f} MB ({(2713-memory_after)/2713*100:.1f}% reduction)")

# ── Step 9: Drop Redundant Columns ───────────────────────────────────────────
# Drop columns identified as redundant during EDA
# Interview topic: always document WHY you drop a column
print("\nStep 9: Dropping redundant columns...")

cols_to_drop = [
    # PII columns — should not be used for modeling
    'Surname', 'Phone', 'Email', 'AadhaarLast4',
    # Highly correlated with Balance (>0.90)
    'AvgMonthlyBalance', 'SavingsAccountBalance', 
    'TotalProductValue',
    # Production metadata — not relevant for model training
    'ModelVersion', 'PreviousModelScore', 'DataQualityFlag',
]

df = df.drop(columns=cols_to_drop)
print(f"✅ Dropped {len(cols_to_drop)} redundant columns")
print(f"   Remaining columns: {df.shape[1]}")

# ── Step 10: Final Validation ─────────────────────────────────────────────────
# Always validate cleaned dataset before saving
# Interview topic: data validation is critical in prod pipelines
print("\nStep 10: Final validation...")

print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Missing values: {df.isnull().sum().sum():,}")
print(f"   Duplicates: {df.duplicated().sum():,}")
print(f"   Churn rate: {df['Exited'].mean():.2%}")
print(f"   Gender values: {df['Gender'].unique().tolist()}")
print(f"   Age range: {df['Age'].min()} - {df['Age'].max()}")
print(f"   CreditScore range: {df['CreditScore'].min()} - {df['CreditScore'].max()}")

# ── Step 11: Save Cleaned Data ────────────────────────────────────────────────
# Save to data/processed — never overwrite raw data
# Interview topic: always keep raw data intact for reproducibility
print("\nStep 11: Saving cleaned dataset...")
output_path = '/app/data/processed/bank_churn_cleaned.csv'
df.to_csv(output_path, index=False)

end = time.time()
print(f"\n{'='*50}")
print(f"✅ DATA CLEANING COMPLETE!")
print(f"{'='*50}")
print(f"📁 Saved to: {output_path}")
print(f"⏱️  Time taken: {end-start:.1f} seconds")
print(f"📊 Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")