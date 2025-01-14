import pandas as pd
from typing import Dict

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the purchase data
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Preprocessed purchase data
    """
    df = pd.read_csv(file_path, parse_dates=['purchaseDate'])
    return df

def get_basic_stats(df: pd.DataFrame) -> Dict:
    """
    Calculate basic statistics about the dataset
    
    Args:
        df (pd.DataFrame): Purchase data
        
    Returns:
        Dict: Dictionary containing basic statistics
    """
    stats = {
        'total_customers': df['customerId'].nunique(),
        'total_products': df['productId'].nunique(),
        'total_transactions': len(df),
        'date_range': (df['purchaseDate'].min(), df['purchaseDate'].max()),
        'total_revenue': df['purchaseAmount'].sum(),
        'avg_transaction_value': df['purchaseAmount'].mean(),
        'avg_customer_value': df.groupby('customerId')['purchaseAmount'].sum().mean(),
        'sd_customer_value': df.groupby('customerId')['purchaseAmount'].sum().std()
    }
    return stats

def analyze_top_products(df: pd.DataFrame, top_n: int = 5, sort_by: str = 'total_revenue') -> pd.DataFrame:
    """
    Analyze top-selling products
    
    Args:
        df (pd.DataFrame): Purchase data
        top_n (int): Number of top items to return
        sort_by (str): Metric to sort by ('total_revenue' or 'number_of_sales')
        
    Returns:
        pd.DataFrame: Top products DataFrame with sales metrics
    """
    top_products = df.groupby('productName').agg({
        'purchaseAmount': ['count', 'sum']
    })
    top_products.columns = ['number_of_sales', 'total_revenue']
    
    top_products = top_products.sort_values(sort_by, ascending=False).head(top_n)
    top_products['total_revenue'] = top_products['total_revenue'].map('${:,.2f}'.format)
    
    return top_products

def analyze_top_categories(df: pd.DataFrame, top_n: int = None, sort_by: str = 'total_revenue') -> pd.DataFrame:
    """
    Analyze top-selling categories
    
    Args:
        df (pd.DataFrame): Purchase data
        top_n (int, optional): Number of top items to return. If None, returns all categories
        sort_by (str): Metric to sort by ('total_revenue' or 'number_of_sales')
        
    Returns:
        pd.DataFrame: Top categories DataFrame with sales metrics
    """
    top_categories = df.groupby('productCategory').agg({
        'purchaseAmount': ['count', 'sum']
    })
    top_categories.columns = ['number_of_sales', 'total_revenue']
    top_categories = top_categories.sort_values(sort_by, ascending=False)
    
    if top_n is not None:
        top_categories = top_categories.head(top_n)
        
    top_categories['total_revenue'] = top_categories['total_revenue'].map('${:,.2f}'.format)
    
    return top_categories


def print_dataset_summary(stats: Dict):
    """
    Print basic summary statistics about the dataset
    
    Args:
        stats (Dict): Dictionary containing basic statistics
    """
    print("\n=== Dataset Summary ===")
    print(f"Number of customers: {stats['total_customers']}")
    print(f"Number of products: {stats['total_products']}")
    print(f"Number of transactions: {stats['total_transactions']}")
    print(f"Date range: {stats['date_range'][0].date()} to {stats['date_range'][1].date()}")

def print_transaction_stats(stats: Dict, df: pd.DataFrame):
    """
    Print statistics about transaction values
    
    Args:
        stats (Dict): Dictionary containing basic statistics
        df (pd.DataFrame): Purchase data
    """
    print("\n=== Transaction Statistics ===")
    print(f"Total revenue: ${stats['total_revenue']:,.2f}")
    print(f"Average transaction value: ${stats['avg_transaction_value']:,.2f}")
    print(f"Standard deviation of transaction value: ${df['purchaseAmount'].std():,.2f}")

def print_customer_stats(stats: Dict):
    """
    Print statistics about customer spending patterns
    
    Args:
        stats (Dict): Dictionary containing basic statistics
    """
    print("\n=== Customer Spending Statistics ===")
    print(f"Average spending per customer: ${stats['avg_customer_value']:,.2f}")
    print(f"Standard deviation of spending per customer: ${stats['sd_customer_value']:,.2f}")

def print_top_products(df: pd.DataFrame):
    """
    Print analysis of top-selling products
    
    Args:
        df (pd.DataFrame): Purchase data
        top_n (int): Number of top items to return
    """
    while True:
        print("\nSort by:")
        print("1. Number of Sales")
        print("2. Total Revenue")
        
        try:
            sort_choice = int(input("\nEnter your choice (1 or 2): "))
            if sort_choice in [1, 2]:
                sort_by = 'number_of_sales' if sort_choice == 1 else 'total_revenue'
                top_n = int(input("Enter the number of top products to display: "))
                top_products = analyze_top_products(df, top_n, sort_by)
                print(f"\n=== Top {top_n} Products (sorted by {sort_by.replace('_', ' ')}) ===")
                print(top_products)
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")

def print_top_categories(df: pd.DataFrame):
    """
    Print analysis of top-selling categories
    
    Args:
        df (pd.DataFrame): Purchase data
    """
    while True:
        print("\nSort by:")
        print("1. Number of Sales")
        print("2. Total Revenue")
        
        try:
            sort_choice = int(input("\nEnter your choice (1 or 2): "))
            if sort_choice in [1, 2]:
                sort_by = 'number_of_sales' if sort_choice == 1 else 'total_revenue'
                
                show_all = input("\nShow all categories? (y/n): ").lower().strip()
                if show_all == 'y':
                    top_categories = analyze_top_categories(df, sort_by=sort_by)
                    print(f"\n=== All Categories (sorted by {sort_by.replace('_', ' ')}) ===")
                else:
                    try:
                        top_n = int(input("Enter the number of top categories to display: "))
                        top_categories = analyze_top_categories(df, top_n, sort_by)
                        print(f"\n=== Top {top_n} Categories (sorted by {sort_by.replace('_', ' ')}) ===")
                    except ValueError:
                        print("Invalid input. Showing all categories.")
                        top_categories = analyze_top_categories(df, sort_by=sort_by)
                        print(f"\n=== All Categories (sorted by {sort_by.replace('_', ' ')}) ===")
                
                print(top_categories)
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")

def display_basic_stats_menu():
    """
    Display the menu options for basic statistics
    """
    print("\nBasic Statistics Menu:")
    print("0. Back to main menu")
    print("1. Dataset Summary statistics")
    print("2. Transaction value statistics")
    print("3. Spending per customer statistics")
    print("4. Top Products")
    print("5. Top Categories")