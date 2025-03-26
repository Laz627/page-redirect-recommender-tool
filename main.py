import streamlit as st
import pandas as pd
import numpy as np
import openai
import requests
from bs4 import BeautifulSoup
import trafilatura
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import re
import time
from urllib.parse import urlparse
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="301 Redirect Mapping Tool",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None

# Page title and info
st.title("301 Redirect Mapping Tool")
st.write("""
This tool helps you identify the most semantically relevant pages for creating 301 redirects.
Upload your list of existing pages and enter the URLs you want to map redirects for.
""")

# API Key handling
with st.sidebar:
    st.header("API Configuration")
    
    if "openai_api_key" in st.secrets:
        st.success("API key loaded from secrets")
        openai.api_key = st.secrets["openai_api_key"]
    else:
        api_key = st.text_input("Enter your OpenAI API key", type="password")
        if api_key:
            openai.api_key = api_key
        else:
            st.warning("Please enter your OpenAI API key to continue")
            st.stop()

# Function to normalize URLs
def normalize_url(url):
    """Clean and normalize URLs for better comparison"""
    try:
        # Add scheme if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # Parse the URL
        parsed = urlparse(url)
        
        # Rebuild with lowercase domain and no trailing slash
        clean_path = parsed.path.rstrip('/')
        if not clean_path:
            clean_path = '/'
            
        normalized = parsed.scheme.lower() + '://' + parsed.netloc.lower() + clean_path
        
        return normalized
    except Exception as e:
        return url

# Function to extract text from a URL
@st.cache_data(show_spinner=False, ttl=3600)
def extract_text_from_url(url):
    try:
        # Normalize URL
        url = normalize_url(url)
        
        # Set request headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        
        # Get URL content
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        html_content = response.text
        
        # Use trafilatura to extract main content
        extracted_text = trafilatura.extract(html_content, 
                                           include_formatting=False, 
                                           include_links=False, 
                                           include_images=False)
        
        # Fallback to BeautifulSoup if trafilatura fails
        if not extracted_text:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "header", "footer", "nav"]):
                element.extract()
            
            # Get text and clean it
            extracted_text = soup.get_text(separator=' ', strip=True)
            extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
        
        # Truncate if too long (OpenAI has token limits)
        if len(extracted_text) > 8000:
            extracted_text = extracted_text[:8000]
            
        return {
            "url": url,
            "content": extracted_text,
            "status": "success"
        }
    except Exception as e:
        return {
            "url": url,
            "content": "",
            "status": "error",
            "error": str(e)
        }

# Function to get embeddings with batch processing
@st.cache_data(show_spinner=False)
def get_embeddings_batch(texts, model="text-embedding-3-large"):
    """Get embeddings for a batch of texts"""
    try:
        # Filter out empty texts
        valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
        
        if not valid_texts:
            return []
        
        # Get embeddings from OpenAI
        response = openai.Embedding.create(
            input=valid_texts,
            model=model
        )
        
        return [item["embedding"] for item in response["data"]]
    except Exception as e:
        st.error(f"Failed to generate embeddings: {str(e)}")
        return [np.zeros(3072)] * len(texts)  # Return zeros on error

# Function to process URLs in parallel
def process_urls_in_parallel(urls, max_workers=10):
    results = []
    
    with st.spinner(f"Fetching content from {len(urls)} URLs..."):
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process URLs in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(extract_text_from_url, url): url for url in urls}
            
            completed = 0
            success_count = 0
            error_count = 0
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == "success":
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    results.append({
                        "url": url,
                        "content": "",
                        "status": "error",
                        "error": str(e)
                    })
                
                completed += 1
                progress_bar.progress(completed / len(urls))
                status_text.text(f"Processed {completed}/{len(urls)} URLs. Success: {success_count}, Errors: {error_count}")
    
    return results

# Function to process all embeddings in batches
def get_all_embeddings(texts, batch_size=100):
    all_embeddings = []
    
    with st.spinner("Generating embeddings..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Update progress
            batch_num = i // batch_size + 1
            status_text.text(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            # Get embeddings for this batch
            batch_embeddings = get_embeddings_batch(batch)
            
            # Handle case of fewer returned embeddings
            if len(batch_embeddings) < len(batch):
                batch_embeddings.extend([np.zeros(3072)] * (len(batch) - len(batch_embeddings)))
            
            all_embeddings.extend(batch_embeddings)
            
            # Update progress
            progress_bar.progress(min(1.0, (i + len(batch)) / len(texts)))
    
    return all_embeddings

# Function to find similar pages efficiently
def find_similar_pages(target_df, source_df, top_n=3):
    with st.spinner("Finding semantically similar pages..."):
        progress_bar = st.progress(0)
        
        results = []
        
        # Convert to numpy arrays for faster computation
        target_embeddings = np.array(target_df["embedding"].tolist())
        source_embeddings = np.array(source_df["embedding"].tolist())
        
        # For each target URL
        for i, (_, target_row) in enumerate(target_df.iterrows()):
            # Calculate all similarities at once
            target_emb = np.array([target_row["embedding"]])
            similarities = cosine_similarity(target_emb, source_embeddings)[0]
            
            # Get indices of top matches
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            # Collect match details
            matches = []
            for idx in top_indices:
                source_row = source_df.iloc[idx]
                matches.append({
                    "url": source_row["url"],
                    "title": source_row.get("meta_title", ""),
                    "description": source_row.get("meta_description", ""),
                    "similarity": float(similarities[idx])
                })
            
            # Add to results
            results.append({
                "target_url": target_row["url"],
                "matches": matches
            })
            
            # Update progress
            progress_bar.progress((i + 1) / len(target_df))
    
    return results

# Sidebar - File upload section
with st.sidebar:
    st.header("Upload Source Pages")
    st.write("Upload a CSV or Excel file with URLs, Meta Titles, and Meta Descriptions.")
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["csv", "xlsx"],
        help="File should have columns for URL, Meta Title, and Meta Description"
    )
    
    # Sample data format
    with st.expander("View Expected Data Format"):
        sample_data = pd.DataFrame({
            "url": ["https://example.com/page1", "https://example.com/page2"],
            "meta_title": ["Example Page 1", "Example Page 2"],
            "meta_description": ["Description for page 1", "Description for page 2"]
        })
        st.dataframe(sample_data)

# Main content - Target URLs input
st.header("Enter URLs to Find Redirects For")
target_urls_text = st.text_area(
    "Enter one URL per line:",
    height=150,
    help="These are the URLs you want to find redirect targets for"
)

# Additional settings
with st.expander("Advanced Settings"):
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Number of matches per URL", min_value=1, max_value=10, value=3)
        batch_size = st.slider("Embedding batch size", min_value=10, max_value=500, value=100)
    with col2:
        max_workers = st.slider("Max parallel requests", min_value=1, max_value=20, value=10)
        include_content = st.checkbox("Include crawled content in matching", value=True)

# Process button
process_button = st.button("Find Redirect Matches", type="primary")

# Main processing logic
if process_button:
    if not uploaded_file:
        st.error("Please upload a file with source pages.")
    elif not target_urls_text.strip():
        st.error("Please enter at least one target URL.")
    else:
        try:
            # Parse target URLs
            target_urls = [url.strip() for url in target_urls_text.split("\n") if url.strip()]
            
            # Load source data
            if uploaded_file.name.endswith('.csv'):
                source_df = pd.read_csv(uploaded_file)
            else:
                source_df = pd.read_excel(uploaded_file)
            
            # Check required columns
            if "url" not in source_df.columns:
                st.error("The uploaded file must contain a 'url' column.")
                st.stop()
            
            # Show data overview
            st.write(f"Loaded {len(source_df)} source pages and {len(target_urls)} target URLs.")
            
            # Process source data
            with st.spinner("Processing source data..."):
                # Normalize URLs
                source_df["url"] = source_df["url"].apply(normalize_url)
                
                # Combine metadata for better embeddings
                source_df["combined_text"] = source_df["url"].astype(str)
                
                if "meta_title" in source_df.columns:
                    source_df["combined_text"] += " " + source_df["meta_title"].fillna("").astype(str)
                
                if "meta_description" in source_df.columns:
                    source_df["combined_text"] += " " + source_df["meta_description"].fillna("").astype(str)
                
                # Calculate embeddings for source pages
                source_texts = source_df["combined_text"].tolist()
                st.info(f"Generating embeddings for {len(source_texts)} source pages...")
                source_embeddings = get_all_embeddings(source_texts, batch_size=batch_size)
                source_df["embedding"] = source_embeddings
            
            # Process target URLs to get content
            target_results = process_urls_in_parallel(target_urls, max_workers=max_workers)
            
            # Filter successful results
            successful_targets = [r for r in target_results if r["status"] == "success"]
            failed_targets = [r for r in target_results if r["status"] == "error"]
            
            if failed_targets:
                st.warning(f"Failed to process {len(failed_targets)} URLs:")
                for fail in failed_targets[:5]:  # Show first 5 failures
                    st.warning(f"- {fail['url']}: {fail.get('error', 'Unknown error')}")
                if len(failed_targets) > 5:
                    st.warning(f"...and {len(failed_targets) - 5} more failures")
            
            if not successful_targets:
                st.error("Could not process any of the provided URLs. Please check the URLs and try again.")
                st.stop()
            
            # Create DataFrame for target data
            target_df = pd.DataFrame(successful_targets)
            
            # Prepare content for embedding
            if include_content:
                # Include crawled content
                target_df["embedding_text"] = target_df.apply(
                    lambda row: f"{row['url']} {row['content']}", axis=1
                )
            else:
                # Use only URL
                target_df["embedding_text"] = target_df["url"]
            
            # Generate embeddings for target pages
            st.info(f"Generating embeddings for {len(target_df)} target pages...")
            target_embeddings = get_all_embeddings(target_df["embedding_text"].tolist())
            target_df["embedding"] = target_embeddings
            
            # Find similar pages
            results = find_similar_pages(target_df, source_df, top_n=top_n)
            
            # Store results
            st.session_state.results = results
            
            # Display results
            st.header("Redirect Recommendations")
            
            results_data = []  # For CSV export
            
            for result in results:
                st.subheader(f"For: {result['target_url']}")
                
                for i, match in enumerate(result["matches"]):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        expander = st.expander(f"Match {i+1}: {match['url']}")
                        with expander:
                            st.write(f"**Title:** {match['title']}")
                            st.write(f"**Description:** {match['description']}")
                    with col2:
                        st.metric(
                            "Similarity", 
                            f"{match['similarity']:.2%}",
                            delta=None
                        )
                    
                    # Add to results data
                    results_data.append({
                        "source_url": result["target_url"],
                        "redirect_to": match["url"],
                        "redirect_title": match["title"],
                        "similarity_score": match["similarity"],
                        "rank": i+1
                    })
                
                st.markdown("---")
            
            # Create DataFrame for export
            results_df = pd.DataFrame(results_data)
            
            # Provide export options
            col1, col2 = st.columns(2)
            with col1:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results as CSV",
                    data=csv,
                    file_name="redirect_recommendations.csv",
                    mime="text/csv"
                )
            with col2:
                # Excel export
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    results_df.to_excel(writer, index=False, sheet_name='Redirects')
                buffer.seek(0)
                
                st.download_button(
                    "Download Results as Excel",
                    data=buffer,
                    file_name="redirect_recommendations.xlsx",
                    mime="application/vnd.ms-excel"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your input data and try again.")

# Add instructions
with st.expander("How to use this tool"):
    st.write("""
    ### Instructions
    
    1. **Upload a file** with your existing pages (URLs, titles, descriptions) using the sidebar
    2. **Enter the URLs** you want to find redirects for in the text area
    3. **Click "Find Redirect Matches"** to process and get recommendations
    4. Review the results and download the CSV or Excel file
    
    ### How it works
    
    - The tool uses OpenAI's text-embedding-3-large model to find semantically similar pages
    - It crawls the target URLs to extract page content
    - It generates embeddings (vector representations) of your source and target content
    - It finds the most similar pages based on cosine similarity of these embeddings
    
    ### Tips
    
    - For best results, include meta titles and descriptions in your source data
    - The tool works best when your content is clearly categorized
    - Higher similarity scores (>70%) indicate strong semantic matches
    """)

# Add footer
st.markdown("---")
st.markdown(
    "Made with Streamlit and OpenAI's text-embedding-3-large"
)
