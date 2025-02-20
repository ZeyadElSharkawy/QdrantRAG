import PyPDF2
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid
import google.generativeai as genai
import nltk

#Splits doc into sentances, base of sentence-based chunking strategy
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


GENAI_API_KEY = "AIzaSyC9fBUyi5ffTiY0FLSNFf9AJsUweXyt-ek"
genai.configure(api_key=GENAI_API_KEY)


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() or ""
    return text


def chunk_text_sentences(text, chunk_size=256, chunk_overlap=32):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:  # +1 for space
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks



pdf_file_path = "D:\\A-One\\Qdrant\\1221300933_2741_amended final report.pdf"
document_text = extract_text_from_pdf(pdf_file_path)


chunks = chunk_text_sentences(document_text)



try:
    client = QdrantClient(host="localhost", port=6333)
    print("Successfully connected to Qdrant server.")
except Exception as e:
    print(f"Could not connect to Qdrant server.  Make sure Qdrant is running. Error: {e}")
    exit()

collection_name = "my_document_collection"
vector_size = 384

# Create Collection (Check if it exists)
try:
    client.get_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
except:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created successfully.")


model = SentenceTransformer('all-MiniLM-L6-v2')

#Embed and Add Chunks to Collection
points = []
for chunk in chunks:
    embedding = model.encode(chunk)
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={"text": chunk},
    )
    points.append(point)

client.upsert(
    collection_name=collection_name,
    points=points,
    wait=True
)

print(f"{len(chunks)} chunks added to Qdrant.")


query_text = "What are the advantages of one-class SVM?"
query_embedding = model.encode(query_text)

search_results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding.tolist(),
    limit=3,
    with_payload=True
)


if search_results:
    print("Retrieved Chunks:")
    context = "\n".join([result.payload['text'] for result in search_results])
    print(context)
else:
    print("No results found.")
    context = ""


if context:
    prompt = f"""
    Context: {context}

    You are an expert in malware detection and machine learning.  Based on the context provided, answer the question as if you're the expert.  Be specific and provide details. If the context does not provide enough information to answer the question, respond with "The context does not provide a clear explanation of how one-class SVM is used in malware detection."

    Question: {query_text}
    Answer:
    """
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    response = gemini_model.generate_content(prompt)

    print("\nGemini's Answer:")
    print(response.text)
else:
    print("No context to send for generation.")