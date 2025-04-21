import random
import numpy as np
import pandas as pd
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from typing import List, Tuple
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = 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

# import product list
df_products = pd.read_csv("data/product_list.csv")


supplier_data = {
    "Supplier Name": [
        "ElectroDirect Inc.", "Tech Emporium Ltd.", "Global Gadgets Co.", "Prime Picks Electronic",
        "ValueTech Supply", "Nexus Retail Group", "Innovate Electronics", "Apex Digital Solutions",
        "Circuit Central", "Streamline Tech"
    ],
    "Return Policy": [
        "30 days for full refund, original packaging, unused. Buyer pays return shipping.",
        "14 days for exchange/store credit, unopened. No returns opened items unless defective.",
        "60 days for full refund, open box accepted, 15% restocking fee. Free return shipping.",
        "21 days for full refund or exchange, no restocking fee. Buyer pays return shipping.",
        "7 days for exchange only, original condition. No refunds.",
        "45 days for full refund, open box accepted. Free return shipping.",
        "30 days for exchange/store credit, original packaging. Buyer pays return shipping.",
        "90 days for full refund, any condition. Free return shipping.",
        "10 days for full refund, unopened only. Buyer pays return shipping.",
        "30 days for exchange or full refund, open box accepted, 10% restocking fee. Free return shipping."
    ],
    "Avg. Price Ratio to Base": [1.05, 1.15, 0.95, 1.00, 0.85, 1.08, 1.12, 0.90, 1.20, 0.98],
    "Std. Dev.": [0.08, 0.12, 0.07, 0.05, 0.09, 0.10, 0.11, 0.06, 0.15, 0.04],
    "Supplier_ID": ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10'],  # adding supplier ID
    "Return Policy Rank": [3, 2, 4, 3, 1, 4, 2, 5, 1, 3] #adding return policy rank
}
# Convert supplier_data to DataFrame
df_suppliers = pd.DataFrame(supplier_data)


def assign_supplier_and_adjust_price(product_df, supplier_df):
    """
    Assigns a subset of suppliers to each product, adjusting the price based on the supplier's
    price ratio and standard deviation, and adds supplier information to the DataFrame.

    Args:
        product_df (pd.DataFrame): DataFrame containing product data.
        supplier_df (pd.DataFrame): DataFrame containing supplier data.

    Returns:
        pd.DataFrame: Updated DataFrame with supplier and adjusted price information.
    """
    num_products = len(product_df)
    num_suppliers = len(supplier_df)
    min_suppliers = 1  # Minimum number of suppliers per product
    max_suppliers = 3  # Maximum number of suppliers per product

    # Ensure there are suppliers to assign
    if num_suppliers == 0:
        print("Error: No suppliers available to assign.")
        return product_df  # Return original DataFrame

    # Create lists to store assigned supplier data
    assigned_supplier_names = []
    assigned_return_policies = []
    adjusted_prices = []
    product_ids = []
    supplier_ids = []  # keep track of supplier ids
    return_policy_ranks = []
    product_descriptions = [] #added product description

    # Iterate through each product and assign a subset of suppliers
    for index, row in product_df.iterrows():
        base_price = row["Approx. Price (USD)"]
        if base_price == "Varies":
            base_price = np.nan
        base_price = float(base_price)
        product_description = row["Description"] #added

        # Randomly choose the number of suppliers for this product
        num_assigned_suppliers = random.randint(min_suppliers, max_suppliers)
        # Randomly select supplier indices without replacement
        supplier_indices = random.sample(range(num_suppliers), num_assigned_suppliers)

        for supplier_index in supplier_indices:
            selected_supplier = supplier_df.iloc[supplier_index]
            supplier_name = selected_supplier["Supplier Name"]
            return_policy = selected_supplier["Return Policy"]
            avg_price_ratio = selected_supplier["Avg. Price Ratio to Base"]
            std_dev = selected_supplier["Std. Dev."]
            supplier_id = selected_supplier['Supplier_ID']  # get supplier ID
            return_policy_rank = selected_supplier['Return Policy Rank']

            # Calculate adjusted price
            price_multiplier = np.random.normal(avg_price_ratio, std_dev)
            price_multiplier = max(0.7, min(1.3, price_multiplier))
            adjusted_price = base_price * price_multiplier if pd.notna(base_price) else np.nan

            # Append data to lists
            assigned_supplier_names.append(supplier_name)
            assigned_return_policies.append(return_policy)
            adjusted_prices.append(adjusted_price)
            product_ids.append(row["ProductID"])
            supplier_ids.append(supplier_id)  # store supplier ID
            return_policy_ranks.append(return_policy_rank)
            product_descriptions.append(product_description) # Append description

    # Create a new DataFrame with the expanded data
    expanded_df = pd.DataFrame({
        "ProductID": product_ids,
        "Supplier": assigned_supplier_names,
        "Supplier Return Policy": assigned_return_policies,
        "Adjusted Price (USD)": adjusted_prices,
        'Supplier_ID': supplier_ids,
        'Return Policy Rank': return_policy_ranks,
        'Description': product_descriptions #added description
    })
    return expanded_df



# Assign suppliers and adjust prices
df_products_with_suppliers = assign_supplier_and_adjust_price(df_products.copy(), df_suppliers)

df_products_with_suppliers = df_products_with_suppliers.drop_duplicates()
# You can further export this DataFrame to a CSV file if needed:
df_products_with_suppliers.to_csv("data/products_with_suppliers.csv", index=False)

class ProductRecommender:
    def __init__(self, product_df: pd.DataFrame, supplier_df: pd.DataFrame):
        """
        Initializes the ProductRecommender with product and supplier data.

        Args:
            product_df (pd.DataFrame): DataFrame containing product data.
            supplier_df (pd.DataFrame): DataFrame containing supplier data.
        """
        self.product_df = product_df
        self.supplier_df = supplier_df
        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key ,  # Replace with your actual API key
            model="meta-llama/llama-4-maverick:free",
            temperature=0.2
        )
        self.vectorstore = None  # Will be initialized in create_vectorstore

    def create_vectorstore(self) -> None:
        """
        Creates a FAISS vector store from the product descriptions.
        """
        data = list(zip(self.product_df['Description'].astype(str).tolist(), range(len(self.product_df))))

        # Create metadata for each document (using index as metadata)
        metadatas = [{'index': i} for i in range(len(self.product_df))]

        self.vectorstore = FAISS.from_texts(
            texts=[d[0] for d in data],  # Extract descriptions
            embedding=self.embedding_model,
            metadatas=metadatas  # Associate metadata
        )


    def get_relevant_products(self, query: str) -> List[Tuple[pd.Series, float]]:
        """
        Retrieves relevant products based on a user query using semantic search.

        Args:
            query (str): The user's search query.

        Returns:
            List[Tuple[pd.Series, float]]: A list of tuples, where each tuple contains
            a product (as a pd.Series) and its relevance score.  Returns an empty
            list if no products are found.
        """
        if self.vectorstore is None:
            self.create_vectorstore()  # Initialize vectorstore if it doesn't exist

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 20}),  # Limit to top 5 results
            return_source_documents=True
        )
        result = qa_chain({"query": query})

        # Extract relevant information and handle the case where no documents are returned.
        if 'source_documents' in result and result['source_documents']:
            relevant_products = [
                (pd.Series(doc.page_content), doc.metadata['index'])
                for doc in result['source_documents']
            ]
            return relevant_products
        else:
            return []

    def recommend_products(self, query: str) -> pd.DataFrame:
        """
        Recommends products based on a user query.

        Args:
            query (str): The user's search query.

        Returns:
            pd.DataFrame: A DataFrame containing the recommended products.
            Returns an empty DataFrame if no products match the query.
        """
        relevant_products = self.get_relevant_products(query)
        #print(relevant_products)

        if not relevant_products:
            print("No matching products found.")
            return pd.DataFrame()  # Return an empty DataFrame

        # Get the indices of the relevant products
        relevant_indices = [product[1] for product in relevant_products]

        # Select the relevant rows from the original DataFrame
        # Include 'Adjusted Price (USD)' in the columns to select
        recommended_df = self.product_df.iloc[relevant_indices][
            ['ProductID', 'Description', 'Supplier', 'Supplier Return Policy', 'Adjusted Price (USD)', 'Supplier_ID', 'Return Policy Rank']
        ].copy()  # Create a copy  
        # Reset index if needed
        recommended_df.reset_index(drop=True, inplace=True)

        # Add a Relevance Score column (optional)
        # recommended_df['Relevance Score'] = [product[1] for product in relevant_products]
        return recommended_df

    def rank_products(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Ranks the given product DataFrame based on similarity to the query,
        price (lower is better), and return policy rank (higher is better).

        Args:
            df (pd.DataFrame): DataFrame of recommended products.
            query (str): The user search query.

        Returns:
            pd.DataFrame: Ranked product recommendations.
        """
        if df.empty:
            return df

        # Merge with the original product DataFrame to get the 'Price' column
        # Use a left merge to keep all recommended products and add price info
        df = pd.merge(df, self.product_df[['ProductID']],
                      on='ProductID', how='left')
        
        # Rename the price column for easier access
        df.rename(columns={'Adjusted Price (USD)': 'Price'}, inplace=True)
        # Ensure 'Price' is numeric
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')


        # Compute similarity score using the embedding model
        query_embedding = self.embedding_model.embed_query(query)
        description_embeddings = self.embedding_model.embed_documents(df['Description'].astype(str).tolist())
        df['similarity_score'] = [np.dot(query_embedding, doc_emb) for doc_emb in description_embeddings]

        # Normalize price and return policy
        df['price_score'] = 1 / (df['Price'] + 1e-5)  # Lower price = higher score
        if 'Return Policy Rank' in df.columns:
            df['return_policy_score'] = df['Return Policy Rank']
        else:
            df['return_policy_score'] = 0  # fallback

        # Composite score
        df['total_score'] = (
            0.5 * df['similarity_score'] +
            -0.3 * df['price_score'] +
            0.2 * df['return_policy_score']
        )

        df_sorted = df.sort_values(by='total_score', ascending=False).reset_index(drop=True)
        return df_sorted
    
    
    
def main():
    """
    Main function to run the product recommendation system.
    """
    # Assign suppliers and adjust prices (assuming this is done only once)
    df_products_with_suppliers = assign_supplier_and_adjust_price(df_products.copy(), df_suppliers)

    # Initialize the recommender system
    recommender = ProductRecommender(df_products_with_suppliers, df_suppliers)

    # Interactive loop to get user queries and display recommendations
    while True:
        query = input("Enter your product query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break


        recommendations_df = recommender.recommend_products(query)
        if not recommendations_df.empty:
            ranked_df = recommender.rank_products(recommendations_df, query)
            print("\nRecommended Products:")
            print(ranked_df[['ProductID','Description','Supplier','Price','Supplier Return Policy']].drop_duplicates().head(5).to_string(index=False))  # Use to_string for console



if __name__ == "__main__":
    main()
