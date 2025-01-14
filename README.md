# Purchase Data Analysis Tool

This tool provides comprehensive analysis of purchase data including basic statistics, customer segmentation, product recommendations, and data visualizations.

## Implementation Overview

This project implements all required functionality from the assignment:

- **Data Analysis**: Provides comprehensive analysis of purchase patterns, top-selling products, categories, and customer spending metrics
- **Customer Classification**: Uses K-means clustering to segment customers based on their purchase behavior
- **Product Recommendations**: Implements collaborative filtering for personalized product recommendations
- **AI Integration**: Utilizes scikit-learn for clustering and a recommendation system, Utilizing Anthropic API for dynamically generated cluster names.
- **Reporting**: Generates visualizations and detailed analysis reports

## Data Requirements

The program expects a CSV file (purchases.csv) with the following columns:

- Customer ID
- Product ID
- Product Name
- Product Category
- Purchase Amount
- Purchase Date

## Setup Instructions

1. Create and activate a virtual environment:

   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows
   venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:

   - Copy `.env.example` to a new file named `.env`
   - Replace the placeholder API key with your actual Anthropic API key
   - To get an API key:
     1. Visit https://console.anthropic.com/settings/keys
     2. Create an account if you don't have one
     3. Verify your phone number to claim $5 in free credits
     4. Generate and copy your API key

4. Run the application:
   ```bash
   python main.py
   ```

## Features

### 1. Basic Statistics Analysis

- Dataset summary statistics
- Transaction value analysis
- Customer spending patterns
- Top products and categories analysis

### 2. Customer Segmentation

- K-means clustering of customers
- AI-powered segment naming
- Visual segment analysis
- Radar charts for segment comparison

### 3. Product Recommendations

- Collaborative filtering recommendations
- Personalized suggestions per customer
- Purchase pattern analysis
- Similarity-based recommendations

### 4. Data Visualizations

- Revenue trends
- Category performance
- Customer diversity analysis
- Revenue distribution (Pareto)
- Segment comparisons

## Output

The program generates:

- Visualizations in the `visualizations/` directory
- Interactive console-based reports and analysis
- Customer segment insights
- Personalized product recommendations
