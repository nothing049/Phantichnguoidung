import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings('ignore')

def read_and_explore_data(file_path):
    df = pd.read_csv(file_path)
    print("Head of the Data:")
    print(df.head(5))
    print("\nTail of the Data:")
    print(df.tail(5))
    print("\nData Shape (rows, columns):", df.shape)
    print("\nColumn Names:", df.columns)
    print("\nData Info:")
    print(df.info())
    return df

def check_and_handle_missing_data(df):
    print("\nUnique Values in Each Column:")
    print(df.apply(lambda col: col.unique()))
    print("\nNumber of Unique Values in Each Column:")
    print(df.nunique())
    print("\nNumber of Missing Values in Each Column:")
    print(df.isnull().sum())
    print("\nNumber of Duplicate Rows:", len(df[df.duplicated()]))

def analyze_by_location(df):
    location_counts = df["Location"].value_counts()
    print("\nLocation Counts:\n", location_counts)

    category_counts = df.groupby("Location")["Category"].value_counts()
    print("\nRegional Category Trends:\n", category_counts)

    location_purchase_stats = df.groupby("Location")["Purchase Amount (USD)"].agg(["mean", "median", "sum"])
    print("\nRegional Purchase Amount Stats:\n", location_purchase_stats)

    shipping_type_counts = df.groupby("Location")["Shipping Type"].value_counts()
    print("\nRegional Shipping Type Trends:\n", shipping_type_counts)

def analyze_by_season(df):
    seasons = df['Season'].unique()
    average_purchase_by_season = df.groupby('Season')['Purchase Amount (USD)'].mean()

    plt.figure(figsize=(6, 4))
    plt.bar(seasons, average_purchase_by_season, color=['skyblue', 'lightcoral', 'lightgreen', 'lightpink'])
    plt.title("Impact of Season on Purchase")
    plt.xlabel("Season")
    plt.ylabel("Average Purchase (USD)")
    plt.show()

def analyze_by_category(df):
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Category', y='Purchase Amount (USD)', data=df, ci=None, palette='viridis')
    plt.title("Impact of Category on Purchase")
    plt.xticks(rotation=45)
    plt.show()

def analyze_by_gender(df):
    gender_purchase = df.groupby('Gender')['Purchase Amount (USD)'].sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pie(gender_purchase, labels=gender_purchase.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'yellow'], wedgeprops=dict(width=0.4))
    ax.set_title("Impact of Gender on Purchase")
    plt.axis('equal')
    center_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(center_circle)
    plt.show()

def analyze_by_size(df):
    plt.figure(figsize=(6, 4))
    sns.swarmplot(x='Size', y='Purchase Amount (USD)', data=df, palette='Set2')
    plt.title("Impact of Size on Purchase")
    plt.xlabel('Size')
    plt.ylabel('Purchase Amount (USD)')
    plt.xticks(rotation=45)
    plt.show()

def analyze_promo_code_usage(df):
    promo_counts = df['Promo Code Used'].value_counts()

    plt.figure(figsize=(6, 4))
    plt.pie(promo_counts, labels=promo_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightcoral'])
    plt.title("Impact of Promo Code Used on Purchase")
    plt.axis('equal')
    plt.show()

def analyze_customer_distribution_by_location(df):
    location_counts = df["Location"].value_counts()
    location_counts.plot(kind="bar", figsize=(12, 4))
    plt.title("Customer Distribution by Location")
    plt.xlabel("Location")
    plt.ylabel("Number of Customers")
    plt.show()

def analyze_product_distribution_by_category(df):
    category_counts = df['Category'].value_counts()
    colors = ['skyblue', 'lightcoral', 'lightseagreen', 'lightsalmon', 'lightpink']

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    bars = plt.bar(category_counts.index, category_counts.values, color=colors)
    plt.xlabel('Product Categories')
    plt.ylabel('Count')
    plt.title('Distribution of Product Categories')
    plt.xticks(rotation=90)
    plt.tight_layout()
    legend_labels = category_counts.index[:len(colors)]
    legend = plt.legend(bars[:len(colors)], legend_labels, title='Categories', loc='upper right')
    plt.setp(legend.get_title(), fontsize=12)
    plt.show()

def analyze_top_locations(df):
    top_locations = df['Location'].value_counts().head(5).index
    colors = ['#98FB98', '#FFE5CC', '#FFCCFF', '#CCE5FF', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig, axes = plt.subplots(5, 1, figsize=(10, 15))

    for i, location in enumerate(top_locations):
        location_data = df[df['Location'] == location]
        category_counts = location_data['Category'].value_counts().head(10)
        ax = axes[i]
        category_counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_title(f"Categories in {location}")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.set_xticklabels(category_counts.index, rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def analyze_category_distribution_by_age(df):
    age_groups = [15, 25, 35, 45, 55, 65]
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(age_groups)))
    category_counts_by_age = {age: [] for age in age_groups}

    for age in age_groups:
        age_group_data = df[(df['Age'] >= age) & (df['Age'] < age + 10)]
        category_counts = age_group_data['Category'].value_counts()
        category_counts_by_age[age] = category_counts

    width = 0.15
    x = np.arange(len(category_counts_by_age[age_groups[0]].index))

    for i, age in enumerate(age_groups):
        category_counts = category_counts_by_age[age]
        ax.bar(x + i * width, category_counts, width=width, label=f'{age}-{age+10}', color=colors[i])

    ax.set_xlabel('Product Categories')
    ax.set_ylabel('Count')
    ax.set_title('Category Distribution by Age Groups')
    ax.set_xticks(x + width * (len(age_groups) - 1) / 2)
    ax.set_xticklabels(category_counts_by_age[age_groups[0]].index, rotation=45)
    ax.legend(title='Age Group')

    plt.tight_layout()
    plt.show()


def preprocess_for_association(df):
    # Loại bỏ các cột không cần thiết
    df = df.drop(['Customer ID', 'Age', 'Item Purchased', 'Color', 'Payment Method', 'Frequency of Purchases'], axis=1)

    # Chuyển đổi biến phân loại thành nhị phân bằng mã hóa one-hot
    categorical_cols = ['Category', 'Location', 'Size', 'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Chuyển đổi cột 'Gender' thành nhị phân
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Chuyển đổi các cột số thành nhị phân dựa trên ngưỡng
    df['Purchase Above 50'] = (df['Purchase Amount (USD)'] > 50).astype(int)
    df['Review Rating Above 3'] = (df['Review Rating'] > 3).astype(int)
    df['Previous Purchases Above 20'] = (df['Previous Purchases'] > 20).astype(int)

    # Loại bỏ các cột số gốc
    df = df.drop(['Purchase Amount (USD)', 'Review Rating', 'Previous Purchases'], axis=1)

    return df

def analyze_association_rules(data):
    # Tiền xử lý dữ liệu cho khai thác quy tắc liên kết
    processed_data = preprocess_for_association(data)

    # Hiển thị tập hợp mặt hàng thường xuyên
    frequent_itemsets = apriori(processed_data, min_support=0.1, use_colnames=True)
    print("Tập Hợp Mặt Hàng Thường Xuyên:")
    print(frequent_itemsets)

    # Tính confidence và antecedent support
    min_confidence = 0.5
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Hiển thị các quy tắc cùng với antecedent support
    print("\nQuy Tắc Liên Kết:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'antecedent support']])


if __name__ == "__main__":
    # Sử dụng các hàm theo thứ tự để thực hiện các phần khác nhau của báo cáo
    file_path = 'C:/Users/tranq/OneDrive/Tài liệu/kpdl/shopping.csv'
    data = read_and_explore_data(file_path)
    check_and_handle_missing_data(data)
    analyze_by_location(data)
    analyze_by_season(data)
    analyze_by_category(data)
    analyze_by_gender(data)
    analyze_by_size(data)
    analyze_promo_code_usage(data)
    analyze_customer_distribution_by_location(data)
    analyze_product_distribution_by_category(data)
    analyze_top_locations(data)
    analyze_category_distribution_by_age(data)
    analyze_association_rules(data)  