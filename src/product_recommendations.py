import pandas as pd
import numpy as np
from typing import List, Set
from sklearn.metrics.pairwise import cosine_similarity

def create_user_product_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a user-product matrix for collaborative filtering
    
    Args:
        df (pd.DataFrame): Purchase data
        
    Returns:
        pd.DataFrame: User-product matrix with purchase amounts
    """
    return pd.pivot_table(
        df,
        values='purchaseAmount',
        index='customerId',
        columns='productId',
        aggfunc='sum',
        fill_value=0
    )

def get_similar_customers(user_product_matrix: pd.DataFrame, customer_id: int, n_similar: int = 20) -> List[int]:
    """
    Find similar customers based on purchase patterns
    
    Args:
        user_product_matrix (pd.DataFrame): User-product matrix
        customer_id (int): Target customer ID
        n_similar (int): Number of similar customers to return
        
    Returns:
        List[int]: List of similar customer IDs
    """
    # Calculate cosine similarity between all users
    user_similarity = cosine_similarity(user_product_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_product_matrix.index,
        columns=user_product_matrix.index
    )
    
    # Get similar users (excluding self)
    similar_users = user_similarity_df.loc[customer_id].sort_values(ascending=False)[1:n_similar+1].index.tolist()
    return similar_users

def get_customer_purchase_history(df: pd.DataFrame, customer_id: int) -> Set[int]:
    """
    Get the set of products already purchased by a customer
    
    Args:
        df (pd.DataFrame): Purchase data
        customer_id (int): Customer ID
        
    Returns:
        Set[int]: Set of product IDs purchased by the customer
    """
    return set(df[df['customerId'] == customer_id]['productId'].unique())


def learn_recommendation_weights(df: pd.DataFrame, n_customers: int = 100) -> float:
    """
    Learn optimal weights for frequency vs price similarity by testing on recent purchases
    
    Args:
        df (pd.DataFrame): Purchase data
        n_customers (int): Number of customers to sample for learning
        
    Returns:
        float: Optimal weight for frequency (1 - this weight will be used for price similarity)
    """
    # Sort by date and split into training/testing
    df = df.sort_values('purchaseDate')
    split_date = df['purchaseDate'].quantile(0.8)  # Use last 20% of data as test set
    
    train_df = df[df['purchaseDate'] < split_date]
    test_df = df[df['purchaseDate'] >= split_date]
    
    # Sample customers who appear in both training and testing
    test_customers = test_df['customerId'].unique()
    train_customers = train_df['customerId'].unique()
    eligible_customers = list(set(test_customers) & set(train_customers))
    sample_customers = pd.Series(eligible_customers).sample(min(n_customers, len(eligible_customers)))
    
    # Try different weights
    weights_to_try = np.arange(0.2, 0.8, 0.2)
    weight_scores = []
    
    for freq_weight in weights_to_try:
        hits = 0
        total = 0
        
        for customer_id in sample_customers:
            # Get actual next purchase
            actual_next_purchase = test_df[test_df['customerId'] == customer_id].iloc[0]
            
            # Generate recommendations using this weight
            recommendations = generate_recommendations_with_weights(
                train_df, 
                customer_id, 
                freq_weight=freq_weight,
                n_recommendations=10
            )
            
            # Check if actual purchase was in recommendations
            if actual_next_purchase['productId'] in recommendations['productId'].values:
                hits += 1
            total += 1
            
        accuracy = hits / total
        weight_scores.append((freq_weight, accuracy))
    
    # Find best weight
    best_weight = max(weight_scores, key=lambda x: x[1])[0]
    return best_weight

def generate_recommendations_with_weights(df: pd.DataFrame, 
                                       customer_id: int, 
                                       freq_weight: float,
                                       n_recommendations: int = 5) -> pd.DataFrame:
    """
    Generate recommendations using specified weights
    
    Args:
        df (pd.DataFrame): Purchase data
        customer_id (int): Customer ID
        freq_weight (float): Weight to give to purchase frequency (0-1)
        n_recommendations (int): Number of recommendations to generate
        
    Returns:
        pd.DataFrame: Recommended products with scores
    """
    # Create user-product matrix
    user_product_matrix = create_user_product_matrix(df)
    
    # Get similar customers
    similar_customers = get_similar_customers(user_product_matrix, customer_id)

    # Get products already purchased
    purchased_products = get_customer_purchase_history(df, customer_id)
    
    # Get candidate products
    similar_customer_purchases = df[
        (df['customerId'].isin(similar_customers)) & 
        (~df['productId'].isin(purchased_products))
    ]
    
    # Score products
    product_scores = similar_customer_purchases.groupby('productId').agg({
        'purchaseAmount': ['count', 'mean']
    })
    product_scores.columns = ['purchase_frequency', 'average_amount']
    
    # Get customer's average purchase amount
    customer_avg_price = df[df['customerId'] == customer_id]['purchaseAmount'].mean()
    
    # Calculate scores
    product_scores['price_similarity'] = 1 / (1 + abs(product_scores['average_amount'] - customer_avg_price) / customer_avg_price)
    product_scores['normalized_frequency'] = product_scores['purchase_frequency'] / product_scores['purchase_frequency'].max()
    
    # Use provided weights
    price_weight = 1 - freq_weight
    product_scores['final_score'] = (
        freq_weight * product_scores['normalized_frequency'] + 
        price_weight * product_scores['price_similarity']
    )
    
    # Get top recommendations
    top_recommendations = product_scores.sort_values('final_score', ascending=False).head(n_recommendations)
    
    # Add product details like we do in recommend_products
    product_details = df[['productId', 'productName', 'productCategory']].drop_duplicates()
    recommendations = (
        top_recommendations
        .reset_index()  # This brings productId back as a column
        .merge(product_details, on='productId')
        [['productId', 'productName', 'productCategory', 'final_score', 'purchase_frequency', 'average_amount']]
    )
    
    return recommendations

def recommend_products(df: pd.DataFrame, customer_id: int, n_recommendations: int = 5, freq_weight: float = 0.84) -> pd.DataFrame:
    """
    Generate product recommendations for a specific customer using collaborative filtering
    
    Args:
        df (pd.DataFrame): Purchase data
        customer_id (int): Customer ID
        n_recommendations (int): Number of recommendations to generate
        
    Returns:
        pd.DataFrame: Recommended products with scores
    """
    # Create user-product matrix
    user_product_matrix = create_user_product_matrix(df)
    
    # Get similar customers (increased to 20)
    similar_customers = get_similar_customers(user_product_matrix, customer_id)

    # Get products already purchased by the target customer
    purchased_products = get_customer_purchase_history(df, customer_id)
    
    # Get products purchased by similar customers
    similar_customer_purchases = df[
        (df['customerId'].isin(similar_customers)) & 
        (~df['productId'].isin(purchased_products))
    ]
    
    # Score products based on purchase frequency and amount
    product_scores = similar_customer_purchases.groupby('productId').agg({
        'purchaseAmount': ['count', 'mean']
    })
    product_scores.columns = ['purchase_frequency', 'average_amount']
    
    # Get customer's average purchase amount
    customer_avg_price = df[df['customerId'] == customer_id]['purchaseAmount'].mean()
    
    # Calculate price similarity (closer to 1 means price is more similar to customer's average)
    product_scores['price_similarity'] = 1 / (1 + abs(product_scores['average_amount'] - customer_avg_price) / customer_avg_price)
    
    # Normalize frequency
    product_scores['normalized_frequency'] = product_scores['purchase_frequency'] / product_scores['purchase_frequency'].max()
    
    price_weight = 1 - freq_weight
    
    product_scores['final_score'] = (
        freq_weight * product_scores['normalized_frequency'] + 
        price_weight * product_scores['price_similarity']
    )
    
    # Get top recommendations
    top_recommendations = product_scores.sort_values('final_score', ascending=False).head(n_recommendations)
    
    # Add product details
    product_details = df[['productId', 'productName', 'productCategory']].drop_duplicates()
    recommendations = (
        top_recommendations
        .reset_index()
        .merge(product_details, on='productId')
        [['productName', 'productCategory', 'final_score', 'purchase_frequency', 'average_amount']]
    )
    
    # Format the output
    recommendations['final_score'] = recommendations['final_score'].round(3)
    recommendations['average_amount'] = recommendations['average_amount'].map('${:,.2f}'.format)
    
    return recommendations

def print_customer_recommendations(df: pd.DataFrame, freq_weight: float = 0.84):
    """
    Print product recommendations for a specific customer along with their segment information
    
    Args:
        df (pd.DataFrame): Purchase data
    """
    # Get list of customer IDs
    customer_ids = sorted(df['customerId'].unique())
    
    # Shows the range of available customer ids from min to max
    print(f"\nAvailable customer IDs: {customer_ids[0]} to {customer_ids[-1]}")
    
    while True:
        try:
            customer_id = int(input("\nEnter customer ID for recommendations: "))
            if customer_id in customer_ids:
                n_recommendations = int(input("Enter number of recommendations to generate (1-10): "))
                if 1 <= n_recommendations <= 10:
                    # Get customer's purchase history
                    history = df[df['customerId'] == customer_id]
                    print(f"\nCustomer {customer_id}'s Profile:")
                    print(f"Total purchases: {len(history)}")
                    print(f"Total spent: ${history['purchaseAmount'].sum():,.2f}")
                    # Prints favorite category based on what has spent the most money on
                    print(f"Favorite category: {history.groupby('productCategory')['purchaseAmount'].sum().idxmax()}")

                    # Display purchase history
                    print("\nPurchase History:")
                    purchase_history = history[['purchaseDate', 'productName', 'productCategory', 'purchaseAmount']].copy()
                    purchase_history['purchaseDate'] = purchase_history['purchaseDate'].dt.strftime('%Y-%m-%d')
                    purchase_history['purchaseAmount'] = purchase_history['purchaseAmount'].map('${:,.2f}'.format)
                    print(purchase_history.sort_values('purchaseDate', ascending=False).to_string(index=False))
                    
                    # Get recommendations
                    recommendations = recommend_products(df, customer_id, n_recommendations, freq_weight)
                    
                    print(f"\nTop {n_recommendations} Recommended Products:")
                    print(recommendations.to_string(index=False))
                    break
                else:
                    print("Please enter a number between 1 and 10.")
            else:
                print("Customer ID not found in dataset.")
        except ValueError:
            print("Please enter valid numbers.")
