import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
sheet_id = "12JWizEwTPYgXOmh_762iy-Q5694zxhlO"
gid = "0"

csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
df = pd.read_csv(csv_url)
df.info()
df.describe(include='all').T
df.isna().sum().sort_values(ascending=False).head(20)
def clean_price(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = re.sub(r'[^\d.]', '', s)
    try:
        return float(s)
    except:
        return np.nan

price_col = 'Price PKR\n4.75 Crore'
df[price_col + '_raw'] = df[price_col]
df[price_col] = df[price_col].apply(clean_price)
df[[price_col + '_raw', price_col]].head(10)
print(df.columns)
def parse_pkr_text(s):
    s = str(s).lower().strip()
    if 'crore' in s:
        num = re.findall(r'[\d\.]+', s)
        num = float(num[0]) if num else 0
        return num * 10_000_000
    if 'lakh' in s:
        num = re.findall(r'[\d\.]+', s)
        return float(num[0]) * 100_000
    return clean_price(s)
area_col = 'Area 128 Sq. Yd.'
def area_to_sqft(a):
    if pd.isna(a):
        return np.nan
    s = str(a).lower().replace(',', '')
    nums = re.findall(r'[\d\.]+', s)
    if not nums:
        return np.nan
    val = float(nums[0])
    if 'marla' in s:
        return val * 272.25
    if 'kanal' in s:
        return val * 20 * 272.25
    if 'sq' in s or 'ft' in s:
        return val
    return val

df[area_col + '_sqft'] = df[area_col].apply(area_to_sqft)
df[[area_col, area_col + '_sqft']].head(10)
print("Total rows:", len(df))
dup_mask = df.duplicated(subset=['City Karachi','Location DHA Defence, Karachi, Sindh','Price PKR\n4.75 Crore','Area 128 Sq. Yd.'], keep='first')
print("Duplicate rows:", dup_mask.sum())
df = df[~dup_mask].copy()

missing = df.isna().sum().sort_values(ascending=False)
missing[missing>0]
df['City_clean'] = (
    df['City Karachi']
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r'[^a-z\s]', '', regex=True)
)

city_map = {
    'isl': 'islamabad',
    'isb': 'islamabad',
    'islamabd': 'islamabad',
    'karachi ': 'karachi',
    'karcahi': 'karachi',
    'lahor': 'lahore',
    'lahor city': 'lahore',
    'rawalpindi': 'rawalpindi',
    'rawal pindi': 'rawalpindi',
    'faislabad': 'faisalabad',
    'faisalabad ': 'faisalabad',
    'multan city': 'multan',
    'quetta ': 'quetta',
    'peshawer': 'peshawar'
}

df['City_std'] = (
    df['City_clean']
    .replace(city_map)
    .str.title()
)

df[['City Karachi', 'City_std']].drop_duplicates().head(20)
def std_property_type(s):
    if pd.isna(s): return np.nan
    s = str(s).lower()
    if any(k in s for k in ['house', 'home', 'villa', 'bungalow']):
        return 'House/Villa'
    if 'flat' in s or 'apartment' in s:
        return 'Apartment'
    if 'plot' in s or 'land' in s:
        return 'Plot/Land'
    return s.title()

df['Property_Type_std'] = df['Type Flat'].apply(std_property_type)

def extract_int(x):
    if pd.isna(x): return np.nan
    m = re.search(r'(\d+)', str(x))
    return int(m.group(1)) if m else np.nan

for col in ['Bedrooms 2', 'Bathrooms 2']:
    if col in df.columns:
        df[col] = df[col].apply(extract_int)
df['price_per_sqft'] = df['Price PKR\n4.75 Crore'] / df[area_col + '_sqft']

df['Price_millions'] = df['Price PKR\n4.75 Crore'] / 1_000_000

df['bed_cat'] = pd.cut(df['Bedrooms 2'], bins=[-1,0,1,2,3,4,100], labels=['Studio','1','2','3','4','4+'])
col = 'price_per_sqft'
df = df[df[col].notna()]

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"Bounds for {col}: {lower:.2f} - {upper:.2f}")
df_no_out = df[(df[col] >= lower) & (df[col] <= upper)].copy()
print("Before:", len(df), "After removing outliers:", len(df_no_out))
df = df_no_out
plt.figure(figsize=(10,5))
plt.hist(df['Price_millions'].dropna(), bins=50)
plt.title('Distribution of Price (millions PKR)')
plt.xlabel('Price (millions)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(y='Property_Type_std', data=df, order=df['Property_Type_std'].value_counts().index)
plt.title('Counts by Property Type')
plt.show()
plt.figure(figsize=(16,8))
sns.boxplot(x='City_std', y='price_per_sqft', data=df)
plt.xticks(rotation=45)
plt.title('Price per sqft by City')
plt.show()

plt.figure(figsize=(10,6))
sns.violinplot(x='Property_Type_std', y='price_per_sqft', data=df)
plt.xticks(rotation=45)
plt.title('Price per sqft by Property Type')
plt.show()
num_cols = ['Price', area_col + '_sqft', 'price_per_sqft', 'Beds', 'Baths']
num_cols = [c for c in num_cols if c in df.columns]
corr = df[num_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()
top_locations = (df.groupby(['City_std','Location DHA Defence, Karachi, Sindh'])
                   .agg(avg_pps=('price_per_sqft','median'),
                        count=('price_per_sqft','size'))
                   .reset_index()
                   .sort_values(by='avg_pps', ascending=False))
top_locations.head(20)
