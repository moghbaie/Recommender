import streamlit as st
import pandas as pd
from recommender import ProductRecommender, df_products_with_suppliers, df_suppliers

# Initialize the recommender system
recommender = ProductRecommender(df_products_with_suppliers.copy(), df_suppliers)

# App UI
st.set_page_config(page_title="Product Recommender", layout="wide")
st.title("🔍 Intelligent Product Recommender")

# Input query from the user
query = st.text_input("Enter a product-related query", placeholder="e.g., budget-friendly wireless earbuds")

# Ranking toggle
rank_results = st.checkbox("Rank results by price and return policy")

# Run recommendation when a query is entered
if query:
    with st.spinner("Fetching recommendations..."):
        recommended_df = recommender.recommend_products(query)
  

        if not recommended_df.empty:
            if rank_results:
                ranked_df = recommender.rank_products(recommended_df, query)
                st.success(f"Showing top {len(ranked_df)} ranked recommendations:")
                st.dataframe(ranked_df[['ProductID','Description','Supplier','Price','Supplier Return Policy']].drop_duplicates())
            else:
                st.success(f"Showing top {len(recommended_df)} recommendations:")
                st.dataframe(recommended_df[['ProductID','Description','Supplier','Adjusted Price (USD)','Supplier Return Policy']].drop_duplicates())
        else:
            st.warning("No relevant products found for your query.")
else:
    st.info("Please enter a product query above to get recommendations.")
