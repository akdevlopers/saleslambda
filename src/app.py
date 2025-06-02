# Flask version of the Streamlit chatbot as an API

from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from utils.memory_utils import load_persistent_memory, save_persistent_memory
import google.generativeai as genai
import os
from dotenv import load_dotenv
from flask_cors import CORS  # <-- Import
from pydantic import BaseModel, TypeAdapter, ValidationError
import json
import re
from typing import List, Optional, Union
from flask_cors import cross_origin
from langchain.schema import Document
from werkzeug.utils import secure_filename
import pandas as pd


from utils.imagesearch_utils import compute_similarity, load_or_create_embeddings,create_embeddings,get_products

app = Flask(__name__)
CORS(app)
# Constants
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment variables")

os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)
MEMORY_DIR = "faiss_memories"
UPLOAD_FOLDER = './static/uploads'
FAISS_FOLDER = './static/GeminiEmbeddings'
ALLOWED_EXTENSIONS = {'xls', 'xlsx'}
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_FOLDER, exist_ok=True)
# ---------------- Pydantic Models ----------------
from dataclasses import dataclass, asdict

@dataclass
class Product(BaseModel):
    name: str
    variant_title: str
    price:Union[str, int, float] 
    id: Union[str, int]
    images: Optional[List[str]]
    product_link: str
    product_suggestion: bool
    reply: str



class ProductResponse(BaseModel):
    product_suggestion: bool
    products: List[Product]

class FriendlyReply(BaseModel):
    product_suggestion: bool
    reply: str



class FriendlyReply(BaseModel):
    product_suggestion: bool
    reply: str

# TypeAdapters for validation
ProductResponseAdapter = TypeAdapter(ProductResponse)
FriendlyReplyAdapter = TypeAdapter(FriendlyReply)

# ---------------- Helper Functions ----------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def load_excel_with_structured_documents(file_path):
    if file_path.endswith(".xls"):
        df_dict = pd.read_excel(file_path, engine='xlrd', sheet_name=None)
    elif file_path.endswith(".xlsx"):
        df_dict = pd.read_excel(file_path, engine='openpyxl', sheet_name=None)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = []
    for sheet_name, df in df_dict.items():
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            content = f"Sheet Name: {sheet_name}\n" + "\n".join([f"{k}: {v}" for k, v in row_dict.items()])
            doc = Document(page_content=content, metadata={"sheet_name": sheet_name, "row_index": index})
            documents.append(doc)
    return documents


def create_embedding(file_path, faiss_file_id, file_type_func, model_name="models/text-embedding-004"):
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
    faiss_file_directory = os.path.join(FAISS_FOLDER, faiss_file_id)

    if os.path.exists(faiss_file_directory):
        return FAISS.load_local(faiss_file_directory, embeddings, allow_dangerous_deserialization=True)

    documents = file_type_func(file_path)
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(faiss_file_directory)
    return db


def clean_json_string(raw_output: str) -> str:
    """Remove markdown backticks and extra formatting."""
    return re.sub(r"```(json)?", "", raw_output).strip("` \n")

def handle_user_query(raw_output: str) -> Union[ProductResponse, FriendlyReply, str]:
    """
    Takes the LLM raw output (bot_reply), cleans it, parses JSON, and validates using Pydantic.
    Returns a validated ProductResponse or FriendlyReply or an error string.
    """
    try:
        cleaned_output = clean_json_string(raw_output)
        print(cleaned_output)
        parsed_json = json.loads(cleaned_output)

        # If it's a list → assume product suggestions
        if isinstance(parsed_json, list):
            for product in parsed_json:
                # Ensure required keys are present with fallback values
                product.setdefault("id", "")
                product.setdefault("inventory_item_id", "")
                product.setdefault("vendor_id", "")
                product.setdefault("product_link", "")
                product.setdefault("images", [])

            return ProductResponseAdapter.validate_python({
                "product_suggestion": True,
                "products": parsed_json
            })

        # If it's a dict → check if it's a friendly reply or single product
        elif isinstance(parsed_json, dict):
            if parsed_json.get("product_suggestion") is False:
                return FriendlyReplyAdapter.validate_python(parsed_json)
            elif parsed_json.get("product_suggestion") is True:
                # Treat as one-item product list (rare case)
                return ProductResponseAdapter.validate_python({
                    "product_suggestion": True,
                    "products": [parsed_json]
                })

        return "❌ Unexpected format in LLM response."

    except (json.JSONDecodeError, ValidationError) as e:
        return f"❌ Error parsing response: {str(e)}"


assistant_template = """
You are a helpful and friendly customer support chatbot. Your name is Kaira.

You are given product documents. Each document has:
- name
- variant_title
- product_link
- price (INR)
- id
- product_link
- images (Image Link)

Your task:
1. If the user is asking for or referring to products, respond with a JSON array of objects using this structure:
   - name
   - variant_title
   - price
   - id
   - product_link 
   - images
   - product_suggestion: true
   - reply: A friendly message relevant to the product and user query.

📌 Use the following format exactly for product-related responses:

[
  {{
    "name": "Product Name",
    "variant_title": "Variant Title",
    "price": "Rs XXX",
    "id": "Product ID",
    "product_link": "https://xxxx",(if available)
    "images": ["Image URL"],
    "product_suggestion": true,
    "reply": "Your friendly reply based on the product and user intent."
  }}
]


2. If the user is just chatting (not asking about products), respond with a JSON object:
{{
  "product_suggestion": false,
  "reply": "Your friendly chatbot message."
}}

Behavior Guidelines:
- Suggest relevant products clearly.
- Ask follow-up questions to understand user preferences.
- Aim to provide useful product suggestions within a few messages(1-3). Avoid lengthy conversations.
- Mention available categories if the user is unsure.
- If products are unavailable, politely suggest alternatives.
- Be concise and friendly in all replies.
- Display a maximum of 4 products at a time. If more are available, show them only upon user request

Examples:

User: What’s your name?  
Reply: {{
  "product_suggestion": false,
  "reply": "I’m your customer support assistant. I'm here to help with your shopping experience!"
}}

User: Do you have something cool for gifting?  
Reply: [
  {{
    "name": "Product Name",
    "variant_title": "Gift Edition",
    "price": "Rs 999",
    "id": "prod_123",
    "images": ["https://example.com/product.jpg"],
    "product_suggestion": true,
    "reply": "Yes! Here's a great gift option for you. Let me know if you’d like more ideas!"
  }}
]

Context: {context}

User Question: {question}

Chat History: {chat_history}
"""



# assistant_template = """
# You are a helpful and friendly customer support chatbot.
# You are given some product documents. Each document contains metadata fields:
# - variant_title
# - price (INR)
# - name
# - id
# - images (Image Link)

# Your task is to:
# 1. If the user is looking for products, suggest relevant products and respond with a JSON list of objects using the following fields:
# don't add ```json in your response.
#    - name
#    - variant_title
#    - price
#    - id
#    - images
#    - product_suggestion: true
#    - reply  ("Your reply message here.")

# **If product_suggestion key is true reply must have above format and keys **

# 2. If the user is just chatting or not asking for products, respond with a JSON object like:
#    {{
#      "product_suggestion": false,
#      "reply": "Your friendly message here."
#    }}

# Friendly Chat Examples:
# User: What is your name?
# reply:  I'm a customer support assistant. I'm here to help you with your shoping experince

# User: I'm looking for something nice to wear to a wedding.
# Bot: That sounds exciting! Could you tell me more about your style preferences or the kind of outfit you're imagining? I can suggest some great options!    

# As a customer support chatbot:
# - Suggest suitable product variants and alternative options.
# - Ask the user about their preferences and interests.
# - Analyze the current context and utilize previous chat history from memory to personalize responses.

# User: Do you have DSLR cameras?
# reply: "I checked, and it looks like DSLR cameras aren't available right now. 📷 But we do have electronics, mobile phones, and sports gadgets. Would you like to explore one of those instead?

# User: What kind of things you have?
# Bot: "We offer a wide range of products including ("tell the available categories") and more. Would you like to explore any specific category?
# **Be concise**
# Context:
# {context}

# Question: {question}

# Chat history:
# {chat_history}
# """
qa_sessions = {}

def Gemini_QA_System(faiss_file_id,session_id, model_name="models/text-embedding-004"):
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
    faiss_file_directory = f"./static/GeminiEmbeddings/{faiss_file_id}"
    loadfile = FAISS.load_local(faiss_file_directory, embeddings, allow_dangerous_deserialization=True)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    memory = load_persistent_memory(session_id, llm)

    compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.3)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=loadfile.as_retriever(search_kwargs={"k": 100})
    )

    prompt_template = PromptTemplate(
    input_variables=["question", "context", "chat_history"], 
    template=assistant_template
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain, 
        document_variable_name="context"  
    )


    question_generator_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
        Given the following chat history and follow-up question, rephrase the follow-up question to be a standalone question.

        Chat History:
        {chat_history}

        Follow-up Question:
        {question}

        Standalone question:"""
    )
    question_generator = LLMChain(llm=llm, prompt=question_generator_prompt)

    qa = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        memory=memory,
        verbose=True
    )
    qa.memory = memory
    qa._session_id = session_id
    return qa

@app.route("/")
def home():
    folder_names = [name for name in os.listdir(FAISS_FOLDER)
                    if os.path.isdir(os.path.join(FAISS_FOLDER, name))]
    return render_template("chat2.html",folders=folder_names)


@app.route("/chat-inputs")
def upload():
    return render_template("upload_excel.html")

@app.route("/upload-inputs")
def chatinput():
    return render_template("upload-inputs.html")







@app.route("/chat2", methods=["POST"])
def chat2():
    data = request.get_json()
    faiss_file_id = data.get("faiss_file_id", "mayilo")
    user_query = data.get("message")

    if faiss_file_id not in qa_sessions:
        qa_sessions[faiss_file_id] = Gemini_QA_System(faiss_file_id)

    qa_system = qa_sessions[faiss_file_id]

    res = qa_system.invoke({"question": user_query})
    bot_reply = res["answer"]

    if hasattr(qa_system, 'memory') and hasattr(qa_system, '_session_id'):
        save_persistent_memory(qa_system._session_id, qa_system.memory)

    return jsonify({"response": bot_reply})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    faiss_file_id = data.get("faiss_file_id", "mayilo")
    user_query = data.get("message")
    session_id = data.get("session_id")
    print("session id: ",session_id)

    if faiss_file_id not in qa_sessions:
        qa_sessions[faiss_file_id] = Gemini_QA_System(faiss_file_id,session_id)

    qa_system = qa_sessions[faiss_file_id]
    print("hiiii")
    # Get response from QA system
    res = qa_system.invoke({"question": user_query})
    bot_reply = res["answer"]
    print("bot_reply",bot_reply)
    # Save memory
    if hasattr(qa_system, 'memory') and hasattr(qa_system, '_session_id'):
        save_persistent_memory(qa_system._session_id, qa_system.memory)

    # Use the bot_reply directly for parsing
    parsed_response = handle_user_query(bot_reply)
    
    if isinstance(parsed_response, ProductResponse):
        product_dicts = [
            {
                'name': p.name,
                'variant_title': p.variant_title,
                'price': p.price,
                'id': p.id,
                'images': p.images,
                'product_link': p.product_link,
                'product_suggestion': p.product_suggestion,
                'reply': p.reply,
            }
            for p in parsed_response.products
        ]
        reply = {
            'product_suggestion': parsed_response.product_suggestion,
            'products': product_dicts,
            'query': user_query,
            'status':True
        }

    elif isinstance(parsed_response, FriendlyReply):
        reply = {
            'product_suggestion': parsed_response.product_suggestion,
            'reply': parsed_response.reply,
            'query': user_query,
             'status':True
        }

    else:
        # Fallback for unexpected formats or errors
        reply = {
            'product_suggestion': False,
            'reply': parsed_response if isinstance(parsed_response, str) else "An error occurred.",
            'query': user_query,
             'status':False
        }

    return jsonify(reply)




@app.route('/upload_excel', methods=['POST'])
@cross_origin()

def upload_excel():
    missing_fields = []
    if 'excel' not in request.files:
        missing_fields.append('excel file')
    if 'faiss_file_id' not in request.form:
        missing_fields.append('faiss_file_id')

    if missing_fields:
        return jsonify({
            'error': f"Missing required field(s): {', '.join(missing_fields)}."
        }), 400


    file = request.files['excel']
    faiss_file_id = request.form['faiss_file_id']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            create_embedding(file_path, faiss_file_id, load_excel_with_structured_documents)
            return jsonify({ 'status':True, 'message': 'Embedding created successfully!', 'faiss_file_id': faiss_file_id})
        except Exception as e:
            return jsonify({ 'status':False, 'error': str(e)}), 500
    else:
        return jsonify({'status':False, 'error': 'Invalid file type. Only .xls and .xlsx are allowed.'}), 400
    



@app.route("/image_search", methods=["GET", "POST"])
def image_search():
    if request.method == "POST":
        file = request.files["query_image"]
        vendor_id=19
        if file:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)
            
            try:
                vendor_id=str(vendor_id)
                dataset_embeddings, image_paths = load_or_create_embeddings(vendor_id)

                results = compute_similarity(img_path, dataset_embeddings, image_paths)
                print("results")
                print(results)
                filtered_restult=[(path,score )for path,score in results if score>0.52]

                return render_template("image_search.html", query_path=img_path, results=filtered_restult)
            except Exception as e:
                return f"Error: {e}"

    return render_template("image_search.html", results=None)


@app.route("/ai_image_search", methods=["POST"])
def ai_image_search():
    if "query_image" not in request.files:
        return jsonify({ 'status':False,"message": "Missing 'query_image' file",'data':[]})

    if "vendor_id" not in request.form:
        return jsonify({ 'status':False,"message": "Missing 'vendor_id' in form data",'data':[]})
    
    file = request.files["query_image"]
    vendor_id = request.form["vendor_id"]
    if file:
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(img_path)
    else:
        return jsonify({ 'status':False,"message": "No file selected",'data':[]})

        
    try:
        vendor_id=str(vendor_id)

        dataset_embeddings, image_paths = load_or_create_embeddings(vendor_id)

        results = compute_similarity(img_path, dataset_embeddings, image_paths)
        print("results")
        print(results)
        filtered_restult=[(path,score )for path,score in results if score>0.52]
        images=[path for path,score in results if score>0.52]
        ProductIDs = [int(match.group(1)) for image in images if (match := re.search(r'-(\d+)', image))]
        if  len(ProductIDs)>0:
            products=get_products(ProductIDs)
        else:
            products=[]
            

        return jsonify({ 'status':True, 'message': 'Search results retrieved successfully', 'data': products})


        return render_template("image_search.html", query_path=img_path, results=filtered_restult)
    except Exception as e:
        return f"Error: {e}"

@app.route("/update_image_search_data", methods=["POST"])
def upupdate_image_search_datadate():
        vendor_id = request.form["vendor_id"]
        vendor_id=str(vendor_id)

        dataset_embeddings, image_paths = create_embeddings(vendor_id)
        return jsonify({ 'status':True, 'message': 'Image dataset updated', 'data': []})





if __name__ == "__main__":
    app.run(debug=True)
