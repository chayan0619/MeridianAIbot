# import ast
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from openai import AzureOpenAI


# client = AzureOpenAI(
#   api_key = "f619d2d04b4f44d28708e4c391039d01",
#   api_version = "2024-02-01",
#   azure_endpoint = "https://openainstance001.openai.azure.com/"

# )


# def extract_array_of_embedding_from_file(file_name):
#     df = pd.read_csv(file_name)
#     embedding_list_final = []
#     embedding_list = df.embedding.apply(ast.literal_eval)
#     for temp_element in embedding_list:
#         embedding_list_final.append(temp_element)
#     embedding_array = np.array(embedding_list_final)
#     return embedding_array, df


# def query_array(query, model="text-embedding"):
#     data = client.embeddings.create(input = [query], model=model).data[0].embedding
#     query_array = np.array(data)
#     query_array = query_array.reshape(1, -1)
#     return query_array


# def get_text_cosine_similarity(query_array, db_array, top_k, dataframe):
#     cosine_sim = cosine_similarity(query_array, db_array)
#     cosine_sim = cosine_sim.flatten()
#     top_10_indices = np.argsort(cosine_sim)[-top_k:][::-1]
#     top_10_df = dataframe.iloc[top_10_indices]
#     print(top_10_df)
#     text_list = top_10_df["text"].to_list()
#     # Creating a dictionary with page_no as the key and file_name as the value
#     page_file_dict = top_10_df.set_index('page_no')['file_name'].to_dict()
#     return text_list, page_file_dict


# def extract_content_based_on_query(query,top_k,folder_name):
#     file_name = f"{folder_name}/{folder_name}_embedding.csv"
#     db_array, dataframe = extract_array_of_embedding_from_file(file_name)
#     array_query = query_array(query)
#     resulted_text, citation_dict = get_text_cosine_similarity(array_query, db_array, top_k, dataframe)
#     return resulted_text, citation_dict


import ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from rag_data_processing import CONNECTION_STRING, CONTAINER_NAME
from azure.storage.blob import BlobServiceClient

client = AzureOpenAI(
  api_key = "f619d2d04b4f44d28708e4c391039d01",
  api_version = "2024-02-01",
  azure_endpoint = "https://openainstance001.openai.azure.com/"

)


def extract_array_of_embedding_from_file(file_name):
    df = pd.read_csv(file_name)
    embedding_list_final = []
    embedding_list = df.embedding.apply(ast.literal_eval)
    for temp_element in embedding_list:
        embedding_list_final.append(temp_element)
    embedding_array = np.array(embedding_list_final)
    return embedding_array, df


def query_array(query, model="text-embedding"):
    data = client.embeddings.create(input = [query], model=model).data[0].embedding
    query_array = np.array(data)
    query_array = query_array.reshape(1, -1)
    return query_array


def get_url(connection_string,container_name, folder_name, blob_name):
    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Generate SAS token
    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=f"{folder_name}/{blob_name}",
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=24)  # Set expiry time as needed
    )

    # Construct the full URL to the blob
    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{folder_name}/{blob_name}?{sas_token}"
    return blob_url


def get_text_cosine_similarity(query_array, db_array, top_k, dataframe, folder_name):
    cosine_sim = cosine_similarity(query_array, db_array)
    cosine_sim = cosine_sim.flatten()
    top_10_indices = np.argsort(cosine_sim)[-top_k:][::-1]
    top_10_df = dataframe.iloc[top_10_indices]
    print(top_10_df)
    text_list = top_10_df["text"].to_list()
    # Creating a dictionary with page_no as the key and file_name as the value
    page_file_dict = top_10_df.set_index('page_no')['file_name'].to_dict()
    # List to store the new format
    new_format_list = []

    # Fill the list with dictionaries in the new format
    for page, file in page_file_dict.items():
        file_url = get_url(CONNECTION_STRING,CONTAINER_NAME, folder_name, file)
        new_format_list.append({
            "page_numbers": int(page),
            "file_names": str(file_url)
        })    
    return text_list, new_format_list


def extract_content_based_on_query(query,top_k,folder_name):
    file_name = f"{folder_name}/{folder_name}_embedding.csv"
    db_array, dataframe = extract_array_of_embedding_from_file(file_name)
    array_query = query_array(query)
    resulted_text, citation_dict = get_text_cosine_similarity(array_query, db_array, top_k, dataframe, folder_name)
    return resulted_text, citation_dict