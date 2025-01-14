import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.customer_segmentation as cs

def plot_revenue_trends(df: pd.DataFrame):
    """
    Plot weekly revenue trends
    
    Args:
        df (pd.DataFrame): Purchase data with purchaseDate and purchaseAmount columns
    """
    # Convert purchaseDate to datetime if it isn't already  
    df['purchaseDate'] = pd.to_datetime(df['purchaseDate'])
    
    # Get week start date (Monday) and group by it
    df['week_start'] = df['purchaseDate'].dt.to_period('W-MON').dt.start_time
    weekly_revenue = df.groupby('week_start')['purchaseAmount'].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    
    # Convert week_start to datetime64[ns] to ensure proper plotting
    weekly_revenue['week_start'] = pd.to_datetime(weekly_revenue['week_start'])
    
    # Plot with datetime x-axis
    plt.plot(weekly_revenue['week_start'], weekly_revenue['purchaseAmount'], 
            marker='o', linestyle='-', linewidth=2)
    
    plt.title('Weekly Revenue Trends')
    plt.xlabel('Week Starting')
    plt.ylabel('Revenue ($)')
    
    # Format y-axis with dollar signs and commas
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis dates using DateFormatter
    from matplotlib.dates import DateFormatter
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('visualizations/weekly_revenue_trends.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_category_performance(df: pd.DataFrame):
    """
    Plot category performance showing revenue bars and transaction count line
    
    Args:
        df (pd.DataFrame): Purchase data with productCategory, purchaseAmount columns
    """
    # Aggregate category metrics
    category_metrics = df.groupby('productCategory').agg({
        'purchaseAmount': ['sum', 'count']
    }).reset_index()
    category_metrics.columns = ['category', 'revenue', 'transactions']
    category_metrics = category_metrics.sort_values('revenue', ascending=False)

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # Plot bars for revenue (changed from skyblue to blue)
    bars = ax1.bar(category_metrics['category'], category_metrics['revenue'], color='blue')
    ax1.set_xlabel('Product Category')
    ax1.set_ylabel('Revenue ($)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot line for transaction count
    line = ax2.plot(category_metrics['category'], category_metrics['transactions'], 
                   color='red', linewidth=2, marker='o')
    ax2.set_ylabel('Number of Transactions', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Rotate x-axis labels by 45 degrees
    ax1.set_xticklabels(category_metrics['category'], rotation=45, ha='right')

    plt.title('Category Performance: Revenue vs Transaction Count')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig('visualizations/category_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_customer_product_diversity(df: pd.DataFrame):
   """
   Creates a scatter plot showing the relationship between a customer's total spend
   and the number of unique products they purchase. Includes a line of best fit.
   
   Args:
       df (pd.DataFrame): Purchase data with customerId, productId, purchaseAmount columns
   """
   # Calculate customer metrics including unique product count
   customer_metrics = df.groupby('customerId').agg({
       'purchaseAmount': 'sum',
       'productId': 'nunique'  # Count distinct products
   }).reset_index()
   customer_metrics.columns = ['customerId', 'total_spend', 'unique_products']

   # Create scatter plot
   plt.figure(figsize=(10, 6))
   
   # Plot scatter points with uniform size
   plt.scatter(customer_metrics['unique_products'], customer_metrics['total_spend'],
               s=100, alpha=0.5)
   
   # Add line of best fit
   z = np.polyfit(customer_metrics['unique_products'], customer_metrics['total_spend'], 1)
   p = np.poly1d(z)
   plt.plot(customer_metrics['unique_products'], p(customer_metrics['unique_products']), 
            "r--", alpha=0.8)

   plt.xlabel('Number of Unique Products Purchased')
   plt.ylabel('Total Spend ($)')
   plt.title('Customer Spending vs Product Diversity')
   
   # Format y-axis with dollar signs
   plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
   
   plt.grid(True, linestyle='--', alpha=0.3)
   plt.tight_layout()
   plt.savefig('visualizations/customer_product_diversity.png', dpi=300, bbox_inches='tight')
   plt.show()

def plot_revenue_pareto(df: pd.DataFrame):
   """
   Creates a Pareto chart showing the cumulative revenue distribution across customers.
   Visualizes the revenue concentration in the customer base - for example, showing 
   that x% of customers generate y% of total revenue. Includes a diagonal reference 
   line representing perfectly equal revenue distribution.
   
   The greater the curve's distance above the diagonal, the higher the revenue 
   concentration among top customers. A steep initial curve indicates revenue is 
   heavily concentrated in a small customer segment.

   Args:
       df (pd.DataFrame): Purchase data with customerId and purchaseAmount columns
   """
   # Calculate customer total spend
   customer_spend = df.groupby('customerId')['purchaseAmount'].sum().sort_values(ascending=False)
   
   # Calculate cumulative percentages
   total_revenue = customer_spend.sum()
   customer_spend_pct = customer_spend.cumsum() / total_revenue * 100
   customer_pct = pd.Series(range(1, len(customer_spend) + 1)) / len(customer_spend) * 100

   plt.figure(figsize=(10, 6))
   plt.plot(customer_pct, customer_spend_pct, 'b-', linewidth=2)
   plt.plot([0, 100], [0, 100], 'r--', linewidth=1)  # diagonal line
   plt.xlabel('% of Customers')
   plt.ylabel('% of Revenue')
   plt.title('Customer Revenue Pareto Analysis')
   plt.grid(True)
   plt.tight_layout()
   plt.savefig('visualizations/revenue_pareto.png', dpi=300, bbox_inches='tight')
   plt.show()

def create_visualizations(df: pd.DataFrame):
    """
    Create and display visualizations of the sales data
    
    Args:
        df (pd.DataFrame): Purchase data
    """
    # Display all visualizations
    print("\nGenerating visualizations...")
    plot_revenue_trends(df)
    plot_category_performance(df)
    plot_customer_product_diversity(df)
    plot_revenue_pareto(df)
    cs.print_customer_segments(df, True)

def display_visualization_menu():
    """
    Display the menu options for visualizations
    """
    print("\nVisualization Menu:")
    print("0. Back to main menu")
    print("1. Show Weekly Revenue Trend")
    print("2. Show Category Performance")
    print("3. Show Customer Spending Patterns")
    print("4. Show Revenue Pareto Analysis")
    print("5. Show Customer Segments")
    print("6. Show All Visualizations")
