from openai import AzureOpenAI
from datetime import datetime
import gradio as gr
import uvicorn
from model import extract_content_based_on_query
import json
from fastapi import Query
from typing import Dict
from rag_data_processing import extact_content_embedding_from_file, read_and_split_pdf
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import os
from rag_data_processing import CONNECTION_STRING, CONTAINER_NAME
from pydantic import BaseModel
import time 
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
import shutil
from fastapi.middleware.cors import CORSMiddleware




chat_client = AzureOpenAI(
                      azure_endpoint = "https://openainstance001.openai.azure.com/",  # Replace with your Azure OpenAI endpoint
                      api_key = "f619d2d04b4f44d28708e4c391039d01",  # Replace with your API key
                      api_version = "2024-02-15-preview"
                    )



# def get_response_from_query(query, content, history, language):
#     message = [
#         {"role": "system", "content": f"You are an AI assistant that helps to answer the questions from the given content in {language} language."},
#         {"role": "user", "content": f"""Your task is to follow chain of thought method to first extract accurate answer for given user query, chat history and provided input content. Then change the language of response into {language} language. Give the response in the json format only having 'bot answer' and 'scope' as key.\n\nInput Content : {content} \n\nUser Query : {query}\n\nChat History : {history}\n\nImportant Points while generating response:\n1. The answer of the question should be relevant to the input text.\n2. Answer complexity would be based on input content.\n3. If input content is not provided direct the user to provide content.\n4. Answers should not be harmful or spam. If there is such content give the instructions to user accordingly. \n5. If user query is out of scope of given content give the value of 'scope' key False.\n6. Give the response in the json format. \n\nExtracted json response:"""}
#     ]

#     response = chat_client.chat.completions.create(
#       model="gpt4", # model = "deployment_name"
#       messages = message,
#       temperature=0.7,
#       max_tokens=800,
#       top_p=0.95,
#       frequency_penalty=0,
#       presence_penalty=0,
#       stop=None
#     )
#     # Loading the response as a JSON object
#     json_response = json.loads(response.choices[0].message.content)
#     print(json_response)
#     return json_response


# def language_correct_query(query, history):
#     message = [
#         {"role": "system", "content": "You are an AI assistant that helps to identify and extract the language, fixes the typing error and change the any language into english language content by understanding the user query and history."},
#         {"role": "user", "content": f"""Your task is to helps to identify and extract the language of query string, fixes the typing error and change the any language into english language content by rephrasing the user query and history. Give the response always in the json format only. \n\nInput Content : {query} \n\nHistory : {history}\n\nImportant instructions: \n1. Your task is to identify the language of content.(e.g. : english/french/..)\n2. You have to generate the modified content by fixing the typing error and change the language of input content into english language if it is other than english language content.\n3. Rephrase the input content which would contain history and input query to make it meaningful sentence. Do not provide any extra information. \n\nKey Entities for the json response: \n1. Language\n2. Modified Content\n\nExtracted Json Response :"""}
#     ]

#     response = chat_client.chat.completions.create(
#       model="gpt4", # model = "deployment_name"
#       messages = message,
#       temperature=0.7,
#       max_tokens=800,
#       top_p=0.95,
#       frequency_penalty=0,
#       presence_penalty=0,
#       stop=None

#     )
#     # Loading the response as a JSON object
#     json_response = json.loads(response.choices[0].message.content)
#     return json_response
def get_response_from_query(query, content, history, language):
    message = [
        {"role": "system", "content": f"You are an AI assistant that helps to answer the questions from the given content in {language} language."},
        {"role": "user", "content": f"""Your task is to follow chain of thought method to first extract accurate answer for given user query, chat history and provided input content. Then change the language of response into {language} language. Give the response in the json format only having 'bot answer' and 'scope' as key.\n\nInput Content : {content} \n\nUser Query : {query}\n\nChat History : {history}\n\nImportant Points while generating response:\n1. The answer of the question should be relevant to the input text.\n2. Answer complexity would be based on input content.\n3. If input content is not provided direct the user to provide content.\n4. Answers should not be harmful or spam. If there is such content give the instructions to user accordingly. \n5. If user query is out of scope of given content give the value of 'scope' key False.\n6. Give the response in the json format. \n\nExtracted json response:"""}
    ]

    response = chat_client.chat.completions.create(
      model="gpt4", # model = "deployment_name"
      messages = message,
      temperature=0.7,
      max_tokens=800,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None
    )
    # Loading the response as a JSON object
    json_response = json.loads(response.choices[0].message.content)
    print(json_response)
    return json_response


def language_correct_query(query, history):
    message = [
        {"role": "system", "content": "You are an AI assistant that helps to identify and extract the language, fixes the typing error and change the any language into english language content by understanding the user query."},
        {"role": "user", "content": f"""Your task is to helps to identify and extract the language of query string, fixes the typing error and change the any language into english language content. Give the response always in the json format only. \n\nInput Content : {query} \n\nHistory : {history}\n\nImportant instructions: \n1. Your task is to identify the language of content.(e.g. : english/french/..)\n2. You have to generate the modified content by fixing the typing error and change the language of input content into english language if it is other than english language content.\n\nKey Entities for the json response: \n1. Language\n2. Modified Content\n\nExtracted Json Response :"""}
    ]

    response = chat_client.chat.completions.create(
      model="gpt4", # model = "deployment_name"
      messages = message,
      temperature=0.7,
      max_tokens=800,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None

    )
    # Loading the response as a JSON object
    json_response = json.loads(response.choices[0].message.content)
    return json_response



# Define the query request model
class QueryRequest(BaseModel):
    query_string: str
    folder_name : str

class DownloadRequest(BaseModel):
    folder_name: str

def background_task(folder_path: str):
    # Simulate a long-running task
    _ = extact_content_embedding_from_file(folder_path)
    print(f"Background task completed ")

# Define the response model
class QueryResponse(BaseModel):
    bot_answer: str
    citation_dict: list


def download_blobs_from_folder(container_name, folder_name, connection_string, local_download_path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    folder_path = os.path.join(local_download_path, folder_name)
    
    # Create local download path if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    blob_list = container_client.list_blobs(name_starts_with=folder_name)
    csv_blobs = [blob for blob in blob_list if blob.name.endswith('.csv')]
    
    if not csv_blobs:
        print("No .csv files found in the folder.")
        return False

    for blob in csv_blobs:
        blob_client = container_client.get_blob_client(blob.name)
        local_file_path = os.path.join(folder_path, os.path.relpath(blob.name, folder_name))
        
        # Create directories if they don't exist
        local_dir = os.path.dirname(local_file_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        
        with open(local_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        print(f"Downloaded {blob.name} to {local_file_path}")
    
    return True

def respond_to_question(query_string, folder_name):
    # if csv not present
    current_working_directory = os.getcwd()
    db_path = os.path.join(current_working_directory, folder_name)
    if not os.path.exists(db_path):
        result = download_blobs_from_folder(CONTAINER_NAME, folder_name, CONNECTION_STRING, current_working_directory)
        if result == False:
            return {"bot_answer": "Data Base not craeted yet", "citation_dict": {}}
    history = " "
    # This function should already exist with the required logic
    language_response = language_correct_query(query_string, history)
    # Placeholder response logic
    print(type(language_response))
    print(language_response)
    query_string = language_response["Modified Content"] 
    print("modified query string", query_string)
    content_list, citation_dict = extract_content_based_on_query(query_string, 10,folder_name)
    content = " ".join(content_list)
    print(content)
    print(citation_dict)
    answer = get_response_from_query(query_string, content, history, language_response["Language"].strip().lower())
    print(answer)
    if answer["scope"] == False:
        citation_dict = []

    output_response = {"bot_answer": answer["bot answer"], "citation_dict": citation_dict}  

    return output_response


app = FastAPI()


origins = [
    "https://ai-dashboard-nine-theta.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.post("/list-folders")
def list_folders():
    try:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blobs = container_client.walk_blobs()
        
        # Extract folder names (prefixes)
        folders = set()
        for blob in blobs:
            folder_path = os.path.dirname(blob.name)
            if folder_path:  # Only add if it's not an empty string
                folders.add(folder_path)
        
        return {"folders": list(folders)}
    except ResourceNotFoundError:
        raise HTTPException(status_code=404, detail="Container not found")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        response = respond_to_question(request.query_string, request.folder_name)
        return QueryResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/create-database")
def trigger_task(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    folder_name: str = Form(...)
):
    try:
        # Create folder if it doesn't exist
        folder_path = os.path.join(os.getcwd(), folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Save files to the folder
        for file in files:
            file_path = os.path.join(folder_path, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        files = os.listdir(folder_name)
        pdf_files = [f for f in files] 
        total_chunks = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            print(f"Reading {pdf_file}...")
            chunks = read_and_split_pdf(pdf_path, pdf_file)
            total_chunks += chunks  # Accumulate total chunks  
        if len(total_chunks) <= 0: 
            minutes_to_wait = 0    
        minutes_to_wait = (len(total_chunks) * 2)/60  

        # Add the background task
        background_tasks.add_task(background_task, folder_name)

        return {"message": f"You have to wait for {minutes_to_wait} minutes!"}
    except Exception as Argument:
    
        # creating/opening a file
        f = open("log.txt", "a")
    
        # writing in the file
        f.write(str(Argument))
        
        # closing the file
        f.close() 


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8000)


# Mock function to simulate the bot's response and citation dictionary.

# # Function to handle the query and return the bot's response and citation dict
# def handle_query(query_string, folder_name, user_id):
#     try:
#         response = respond_to_question(query_string, folder_name, user_id)
#         return response["bot_answer"]
#     except Exception as e:
#         return str(e)

# # Create Gradio interface
# iface = gr.Interface(
#     fn=handle_query,
#     inputs=[
#         gr.Textbox(label="User Query"),
#         gr.Textbox(label="Database Name"),
#         gr.Textbox(label="User Id")
#     ],
#     outputs=[
#         gr.Textbox(label="Bot Response")
#     ],
#     title="Meridian Chatbot",
#     description="Enter the Query, Database Name, and your User Id to get a response from the bot."
# )

# # Launch the interface
# iface.launch()


















