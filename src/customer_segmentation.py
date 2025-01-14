import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from anthropic import Anthropic
from typing import Dict

def analyze_customer_segments(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Perform customer segmentation using k-means clustering based on purchase behavior
    
    Args:
        df (pd.DataFrame): Purchase data
        n_clusters (int): Number of clusters to create
        
    Returns:
        pd.DataFrame: Customer segments with their characteristics
    """
    # Create customer features
    customer_features = df.groupby('customerId').agg({
        'purchaseDate': [
            'count',
            lambda x: (x.max() - x.min()).days / (len(x)-1) if len(x) > 1 else 0
        ],
        'purchaseAmount': [
            'sum',  # This will be total spending
            'mean'  # This will be average price per purchase
        ],
        'productCategory': [
            'nunique',
            lambda x: x.mode().iloc[0] if not x.empty else None
        ]
    })
    
    # Name the columns properly
    customer_features.columns = [
        'number_of_purchases',
        'average_days_between_purchase',
        'total_spending',
        'average_price',
        'number_of_categories',
        'top_category'
    ]
    
    # Select features for clustering
    clustering_features = customer_features[[
        'number_of_purchases',
        'total_spending',
        'average_price',
        'number_of_categories',
        'average_days_between_purchase'
    ]]
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(clustering_features)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Add cluster assignments to customer features
    customer_features['cluster'] = clusters
    
    # Calculate cluster characteristics
    cluster_summary = customer_features.groupby('cluster').agg({
        'number_of_purchases': 'mean',
        'total_spending': 'mean',  # Changed from 'sum' to 'mean' to get average per customer
        'average_price': 'mean',
        'number_of_categories': 'mean',
        'average_days_between_purchase': 'mean',
        'top_category': lambda x: x.mode().iloc[0]
    }).round(2)
    
    # Format currency values
    cluster_summary['total_spending'] = cluster_summary['total_spending'].map('${:,.2f}'.format)
    cluster_summary['average_price'] = cluster_summary['average_price'].map('${:,.2f}'.format)
    
    return cluster_summary

def plot_segment_radar(segments: pd.DataFrame, segment_names: Dict[str, str]):
   """
   Create a radar chart comparing customer segments across key metrics.
   
   Args:
       segments (pd.DataFrame): Customer segment data with metrics for each segment
       segment_names (Dict[str, str]): Dictionary mapping cluster numbers to segment names
   """
   # Prepare the data
   # Convert string currency values back to float
   segments['total_spending'] = segments['total_spending'].str.replace('$', '').str.replace(',', '').astype(float)
   segments['average_price'] = segments['average_price'].str.replace('$', '').str.replace(',', '').astype(float)
   
   # Select metrics for radar chart
   metrics = ['number_of_purchases', 'total_spending', 'average_price', 
             'number_of_categories', 'average_days_between_purchase']
   
   # Normalize the data for better visualization
   normalized_data = segments[metrics].copy()
   for metric in metrics:
       normalized_data[metric] = (segments[metric] - segments[metric].min()) / \
                               (segments[metric].max() - segments[metric].min())
   
   # Set up the angles for each metric
   angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
   
   # Close the plot by appending the first value to the end
   angles = np.concatenate((angles, [angles[0]]))
   
   # Create the plot
   fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
   
   # Plot each segment, up to 10 colors
   colors = plt.cm.tab10.colors                
   for idx, (cluster, row) in enumerate(normalized_data.iterrows()):
       values = row.values
       values = np.concatenate((values, [values[0]]))
       ax.plot(angles, values, 'o-', linewidth=2, 
               label=segment_names[str(cluster)], color=colors[idx])
       ax.fill(angles, values, alpha=0.25, color=colors[idx])
   
   # Set the labels
   ax.set_xticks(angles[:-1])
   ax.set_xticklabels(['Purchases', 'Total Spend', 'Avg Price', 
                       'Categories', 'Days Between'])
   
   # Add legend with segment names
   plt.legend(bbox_to_anchor=(1.3, 0.9), title="Customer Segments")
   
   plt.title('Customer Segment Comparison', pad=20)
   plt.tight_layout()
   plt.savefig('visualizations/segment_radar.png', dpi=300, bbox_inches='tight')
   plt.show()

def print_customer_segments(df: pd.DataFrame, show_plot: bool = False):
    """
    Print analysis of customer segments and visualize them

    Args:
    df (pd.DataFrame): Purchase data
    """
    while True:
        n_clusters = int(input("\nEnter the number of customer segments to create (2-10 allowed): "))
        if 2 <= n_clusters <= 10:  # reasonable range
            segments = analyze_customer_segments(df, n_clusters)
            segment_names = name_segments(segments)

            if show_plot:
                # Create radar chart visualization with segment names
                plot_segment_radar(segments, segment_names)

            else:
                # Print characteristics of each segment
                for cluster in range(n_clusters):
                    print(f"\nSegment {cluster+1}: {segment_names[str(cluster)]}")
                    print(f"Number of Purchases: {segments.loc[cluster, 'number_of_purchases']:.1f}")
                    print(f"Average Total Spending: {segments.loc[cluster, 'total_spending']}")
                    print(f"Average Price per Purchase: {segments.loc[cluster, 'average_price']}")
                    print(f"Number of Categories: {segments.loc[cluster, 'number_of_categories']:.1f}")
                    print(f"Days Between Purchases: {segments.loc[cluster, 'average_days_between_purchase']:.1f}")
                    print(f"Top Category: {segments.loc[cluster, 'top_category']}")

        break
    else:
        print("Please enter a number between 2 and 10.")

def name_segments(cluster_summary: pd.DataFrame) -> Dict[str, str]:
    load_dotenv()
    # Initialize Anthropic client using environmental variable for the api key
    anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    message = anthropic.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""
            You are an expert cluster namer. Please provide a name for each of the following clusters 
            based on the characteristics of the customers in each segment. {cluster_summary.to_string()}.
            You should respond with only JSON. The keys will start with 0 and range through n-1, the number of clusters. 
            The values will be the names of the clusters. Both the keys and values will be strings
            There should be nothing else in the response other than pure JSON. 
            Label the clusters with intuitive names (e.g., "High Spenders," "Occasional Buyers")
            """
        }]
    )
    cluster_names = json.loads(message.content[0].text)
    return cluster_names
