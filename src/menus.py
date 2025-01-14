from pathlib import Path
from typing import Dict, Any
import pandas as pd

from src.basic_statistics import (
    load_data,
    get_basic_stats,
    print_dataset_summary,
    print_transaction_stats,
    print_customer_stats,
    print_top_products,
    print_top_categories,
    display_basic_stats_menu
)
from src.customer_segmentation import print_customer_segments
from src.product_recommendations import (
    print_customer_recommendations,
    learn_recommendation_weights
)
from src.data_visualization import (
    create_visualizations,
    plot_revenue_trends,
    plot_category_performance,
    plot_customer_product_diversity,
    plot_revenue_pareto,
    display_visualization_menu
)

def handle_basic_stats_menu(stats: Dict[str, Any], df: pd.DataFrame) -> None:
    """
    Handle the basic statistics submenu and user interactions
    
    Args:
        stats (Dict[str, Any]): Dictionary containing basic statistics about the dataset
        df (pd.DataFrame): The main purchase data DataFrame
    """
    stats_actions = {
        0: None,  # Break the loop
        1: lambda: print_dataset_summary(stats),
        2: lambda: print_transaction_stats(stats, df),
        3: lambda: print_customer_stats(stats),
        4: lambda: print_top_products(df),
        5: lambda: print_top_categories(df)
    }
    
    while True:
        display_basic_stats_menu()
        try:
            stats_choice = int(input("\nEnter your choice: "))
            
            action = stats_actions.get(stats_choice)
            if action is None:
                break
            if action:
                action()
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def handle_visualization_menu(df: pd.DataFrame) -> None:
    """
    Handle the visualization submenu and user interactions
    
    Args:
        df (pd.DataFrame): The main purchase data DataFrame
    """
    viz_functions = {
        1: plot_revenue_trends,
        2: plot_category_performance,
        3: plot_customer_product_diversity,
        4: plot_revenue_pareto,
        5: lambda df: print_customer_segments(df, True)
    }

    while True:
        display_visualization_menu()
        try:
            viz_choice = int(input("\nEnter your choice: "))
            if viz_choice == 0:
                break
            elif viz_choice == 6:
                create_visualizations(df)
            elif viz_choice in viz_functions:
                viz_functions[viz_choice](df)
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def initialize_application() -> tuple[pd.DataFrame, Dict[str, Any], float]:
    """
    Initialize the application by loading data and computing basic statistics
    
    Returns:
        tuple[pd.DataFrame, Dict[str, Any], float]: Tuple containing:
            - DataFrame with purchase data
            - Dictionary of basic statistics
            - Recommendation weight factor
    """
    data_path = Path('purchases.csv')
    df = load_data(data_path)
    print("Data loaded successfully")
    stats = get_basic_stats(df)
    freq_weight = 0.84  # Hardcoded value from training
    return df, stats, freq_weight

def handle_main_menu(df: pd.DataFrame, stats: Dict[str, Any], freq_weight: float) -> None:
    """
    Handle the main application menu and user interactions
    
    Args:
        df (pd.DataFrame): The main purchase data DataFrame
        stats (Dict[str, Any]): Dictionary containing basic statistics about the dataset
        freq_weight (float): Weight factor used for recommendations
    """
    menu_actions = {
        0: lambda: "exit",
        1: lambda: handle_basic_stats_menu(stats, df),
        2: lambda: print_customer_segments(df),
        3: lambda: print_customer_recommendations(df, freq_weight),
        4: lambda: handle_visualization_menu(df)
    }
    
    while True:
        print("\nMain Menu:")
        print("0. Quit")
        print("1. Basic Statistics")
        print("2. Customer Segmentation")
        print("3. Product Recommendations")
        print("4. Visualizations")
        
        try:
            choice = int(input("\nEnter your choice: "))
            
            action = menu_actions.get(choice)
            if action:
                result = action()
                if result == "exit":
                    print("\nThank you for using the application, goodbye!")
                    break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
