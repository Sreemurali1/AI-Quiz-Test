import os
import logging
import json
from dotenv import load_dotenv
from pdf import get_pdf_text, get_text_chunks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from pymongo.mongo_client import MongoClient
import streamlit as st
import json
from bson import ObjectId

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for API Keys
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("Google API Key is not set.")
    raise ValueError("Google API key is not set.")
if not os.getenv('COHERE_API_KEY'):
    logger.error("Cohere API Key is not set.")
    raise ValueError("Cohere API Key is not set.")
if not os.getenv('QDRANT_URL'):
    logger.error("QDRANT URL is not set.")
    raise ValueError("QDRANT URL is not set.")
if not os.getenv('MONGO_URI'):
    logger.error("MONGO_URI is not set.")
    raise ValueError("MONGO_URI is not set.")
if not os.getenv('QDRANT_API_KEY'):
    logger.error("QDRANT API KEY is not set.")
    raise ValueError("QDRANT API KEY is not set.")

# API Key and URL
cohere_api_key = os.getenv("COHERE_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
mongodb_uri = os.getenv("MONGO_URI")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Configure Google Generative AI API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def conversational_chain():
    prompt_template = """
    You are an AI assistant tasked with creating a quiz based on a provided document. Follow these guidelines:

    1. **Document Content**: Base questions on the document.
    2. **User Query**: Focus on the specified lesson or topic.
    3. **Question Types**: Create multiple-choice questions (MCQs) with one correct answer and three distractors.
    4. **Question Format**:
       - **Question_no**:
       - **Question**: [Question text]
       - **Options**:
         -  "A: Option A"
         -  "B: "Option B"
         -  "C: "Option C"
         -  "D: "Option D"
       - **Answer**: [Correct option]
       - **Explanation**: 100-word explanation of the answer.
    6. **Put a same json for all documents.
    7. **Difficulty Levels**: Include easy, medium, and hard questions.
    8. **Clarity and Precision**: Ensure questions are clear and relevant to the document.
    9. **Coverage**: Cover different sections relevant to the user query.
    10. **Number of Questions**: generate MCQs based on user input: {no_of_questions}
   
     Generate a quiz based on the following document and user query, and format the output as JSON:

    

    **context:**
    {context}
    
    **question:**
    {question}
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "no_of_questions"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Initialize Qdrant Client
client = QdrantClient(url=qdrant_url,api_key=qdrant_api_key)

# Initialize the Cohere Embedding
cohere_embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-light-v3.0")

class DocumentChunk:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

def vector_store(text_chunks, collection_name):
    try:
        # Attempt to create a new collection
        try:
            # Create a new collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": 384, "distance": "Cosine"}  # Adjust vector size and distance metric
            )
            logger.info(f"Created new collection: {collection_name}")
        except Exception as e:
            # Handle case where collection might already exist
            logger.warning(f"Collection creation failed, checking if collection exists: {e}")
            if "already exists" in str(e):
                # Delete existing collection
                client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
                
                # Try creating the collection again
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={"size": 384, "distance": "Cosine"}  # Adjust vector size and distance metric
                )
                logger.info(f"Created new collection after deletion: {collection_name}")
            else:
                # Re-raise the exception if it's not related to collection existence
                raise

        # Ensure text chunks are properly wrapped in DocumentChunk if needed
        document_chunks = [DocumentChunk(chunk) if not isinstance(chunk, DocumentChunk) else chunk for chunk in text_chunks]

        # Verify that cohere_embeddings is properly initialized
        if not cohere_embeddings:
            raise ValueError("cohere_embeddings must be properly initialized")

        # Initialize QdrantVectorStore with the collection
        vector_store = QdrantVectorStore(
            client,
            embedding=cohere_embeddings,
            collection_name=collection_name
        )

        vector_store.add_documents(document_chunks)
        return vector_store

    except Exception as e:
        logger.error(f"Error in vector_store: {e}")
        raise

# Global variables
collection_name = None

# Function to Generate Quiz
def generate_quiz(document, user_query, no_of_questions):
    try:
        # Read the PDF file
        pdf_text = get_pdf_text(document)
        text_chunks = get_text_chunks(pdf_text)
        
        # Generate vectors
        vectors = vector_store(text_chunks, collection_name)

        # Retrieve
        db = QdrantVectorStore(client=client, embedding=cohere_embeddings, collection_name=collection_name)

        # Perform similarity search
        docs = db.similarity_search_with_score(query=user_query, k=5)

        # Wrap the docs in DocumentChunk if necessary
        wrapped_docs = [DocumentChunk(doc.page_content, doc.metadata) if isinstance(doc, tuple) else doc for doc, score in docs]

        # Invoke the QA chain
        chain = conversational_chain()
        response = chain.invoke({"input_documents": wrapped_docs, "question": user_query, "no_of_questions": no_of_questions}, return_only_outputs=True)

        # Inspect the response
        logger.info(f"Response from chain: {response}")

        # Handle the response
        if isinstance(response, dict):
            response_json = response
        else:
            try:
                # If the response is a string containing JSON, parse it
                response_json = json.loads(response)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON response.")
                raise ValueError("Failed to parse JSON response.")

        # Return the quiz JSON response
        return response_json

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise ValueError("The provided PDF document could not be found.")

    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise ValueError("There was an issue with the input values or response format.")

    except KeyError as e:
        logger.error(f"Key error: {e}")
        raise ValueError("A required key was missing in the response or document.")

    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        raise ConnectionError("There was an error connecting to the database or service.")

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise ValueError("Failed to decode JSON response from the QA chain.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise RuntimeError("An unexpected error occurred while generating the quiz.")

# Store the Generate Quiz in MongoDB
def store_quiz_in_mongodb(quiz, user_query):
    try:
        # Global MongoDB Client Initialization
        mongo_uri = mongodb_uri
        mongo_client = MongoClient(mongo_uri)

        db = mongo_client["quiz_database"]
        collection = db["quizzes"]
        quiz_document = {
            "user_query": user_query,
            "quiz": quiz
        }
        result = collection.insert_one(quiz_document)
        return result.inserted_id
    except Exception as e:
        logger.error(f"Error storing quiz in MongoDB: {e}")
        raise

def retrieve_quiz_from_mongodb(quiz_id):
    try:
        mongo_uri = mongodb_uri
        mongo_client = MongoClient(mongo_uri)
        db = mongo_client["quiz_database"]
        collection = db["quizzes"]

        # Ensure quiz_id is an ObjectId
        if isinstance(quiz_id, str):
            quiz_id = ObjectId(quiz_id)

        logger.info(f"Retrieving quiz with ID: {quiz_id}")
        quiz_document = collection.find_one({"_id": quiz_id})

        logger.info(f"Type of quiz_document: {type(quiz_document)}")
        logger.info(f"Contents of quiz_document: {quiz_document}")

        if quiz_document and isinstance(quiz_document, dict):
            if "quiz" in quiz_document and "output_text" in quiz_document["quiz"]:
                output_text = quiz_document["quiz"]["output_text"]

                # Log the output_text to debug its contents
                logger.info(f"Output text before parsing: {output_text}")

                try:
                    # Remove the surrounding code block markers if they exist
                    if output_text.startswith("```json") and output_text.endswith("```"):
                        json_str = output_text[7:-3]
                    else:
                        json_str = output_text

                    quiz_data = json.loads(json_str)
                    logger.info(f"Parsed quiz data: {quiz_data}")
                    
                    # Check for the key in quiz_data
                    if isinstance(quiz_data, dict):
                        if "questions" in quiz_data:
                            return quiz_data["questions"]
                        elif "quiz" in quiz_data:
                            return quiz_data["quiz"]
                        else:
                            logger.error("Parsed quiz data does not contain 'questions' or 'quiz' key.")
                            raise ValueError("Parsed quiz data does not contain 'questions' or 'quiz' key.")
                    else:
                        logger.error("Parsed quiz data is not a dictionary.")
                        raise ValueError("Parsed quiz data is not a dictionary.")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from output_text: {e}")
                    raise ValueError("Failed to parse JSON from output_text.")
            else:
                logger.error("Document does not contain 'quiz' field or 'output_text' key.")
                raise ValueError("Document does not contain 'quiz' field or 'output_text' key.")
        else:
            logger.error(f"Quiz with ID {quiz_id} not found or invalid format.")
            raise ValueError("Quiz not found or is in an invalid format.")
    except Exception as e:
        logger.error(f"Error retrieving quiz from MongoDB: {e}")
        raise



# Streamlit UI
# Page setting
st.set_page_config(layout="wide", page_title="AI-Quiz")

st.title("AI-Generated Quiz Test")

# Tabbed interface
tab1, tab2, tab3 = st.tabs(["Instructions", "Quiz Generator", "Take Quiz"])

with tab1:
    st.header("Instructions")
    st.write("""
    1. **Upload a PDF**: Select and upload your document.
    2. **Enter a Topic**: Type the lesson or topic for the quiz.
    3. **Generate Quiz**: Click 'Generate Quiz' to create questions.
    4. **Take the Quiz**: Answer the questions displayed in Take Quiz tab.
    5. **Submit Answers**: Click 'Submit' to see your score and feedback.
    6. **Review Feedback**: Check correct answers and explanations.
    """)

with tab2:
    st.header("Quiz Generator")
    
    # PDF Upload
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    # User Query
    user_query = st.text_input("Enter the lesson or topic for the quiz")
    
    no_of_questions = st.slider("Number of questions to generate:", 0, 50, 10)
    
    # Generate Quiz button
    if st.button("Generate Quiz"):
        if uploaded_file is not None and user_query:
            with st.spinner('Generating quiz...'):
                try:
                    # Set the collection name globally
                    collection_name = uploaded_file.name.split('.')[0]
                    
                    # Read the uploaded PDF file
                    document = uploaded_file.read()
                    
                    # Generate the quiz and retrieve Quiz ID
                    quiz = generate_quiz(document, user_query, no_of_questions)
                    quiz_id = store_quiz_in_mongodb(quiz, user_query)
                    
                    # Display the Quiz ID and store quiz data in session state
                    st.write(f"Your Quiz Id is Generated successfully! You can go Take-Quiz tab to take a test")
                    
                    # Store quiz data in session state
                    st.session_state.quiz_data = quiz
                    st.session_state.quiz_id = quiz_id
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a PDF and enter a user query.")

# Initialize session state variables
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'quiz_loaded' not in st.session_state:
    st.session_state.quiz_loaded = False
if 'quiz_id' not in st.session_state:
    st.session_state.quiz_id = ""

with tab3:
    st.header("Take Quiz")
    
    # Retrieve Quiz ID from session state or user input
    quiz_id = st.text_input("Enter the Quiz ID", value=st.session_state.get('quiz_id', ''))

    if st.button("Load Quiz"):
        if quiz_id:
            try:
                quiz_data = retrieve_quiz_from_mongodb(quiz_id)
                
                if isinstance(quiz_data, list):
                    st.session_state.quiz_data = quiz_data
                    st.session_state.quiz_id = quiz_id
                    st.session_state.quiz_loaded = True
                    st.session_state.user_answers = {}  # Reset user answers on new quiz load
                    st.session_state.submitted = False  # Reset submission state
                else:
                    st.error("Retrieved data is not in the expected format.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a valid Quiz ID.")

    if st.session_state.quiz_loaded:
        quiz_data = st.session_state.quiz_data
        
        for i, q in enumerate(quiz_data):
            st.write(f"**Q{i + 1}: {q['question']}**")
            selected_option = st.session_state.user_answers.get(f"q{i + 1}", "")
            st.session_state.user_answers[f"q{i + 1}"] = st.radio(
                f"Options for Q{i + 1}",
                options=q['options'],
                key=f"answer{i + 1}",
                index=q['options'].index(selected_option) if selected_option in q['options'] else 0
            )
        
        # Initialize the session state for button click tracking
        if 'submitted' not in st.session_state:
            st.session_state.submitted = False

        # Function to handle the submit action
        def handle_submit():
            st.session_state.submitted = True
            st.session_state.score = 0
            st.session_state.feedback = []
            for i, q in enumerate(quiz_data):
                correct_option = q['answer']
                selected_option = st.session_state.user_answers.get(f"q{i + 1}")
                if selected_option == correct_option:
                    st.session_state.score += 1
                st.session_state.feedback.append({
                    "question_no": i + 1,
                    "question": q['question'],
                    "selected_option": selected_option,
                    "correct_option": correct_option,
                    "explanation": q['explanation']
                })

        # Display the submit button
        if not st.session_state.submitted:
            if st.button("Submit", on_click=handle_submit):
                pass  # The button click will trigger the handle_submit function

        # Display score and feedback if the button has been clicked
        if st.session_state.submitted:
            # Calculate and display the score percentage
            score = st.session_state.score
            quiz_data = st.session_state.quiz_data
            score_percentage = (score * 100) / len(quiz_data)
            score_message = f"Your score: {score_percentage:.2f}%"

            # Get the total number of correct answers
            correct_answers = st.session_state.score
            total_questions = len(st.session_state.quiz_data)
            correct_message = f"Correct: {correct_answers}/{total_questions}"

            # Create a two-column layout for the cards
            col1, col2 = st.columns(2)

            # Using st.container to create card-like layouts
            with col1:
                st.markdown(
                    """
                    <div style="background-color: #4CAF50; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h2 style="color: #FFFFFF;">{}</h2>
                    </div>
                    """.format(score_message),
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <div style="background-color: #2196F3; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); text-align: center;">
                        <h2 style="color: #FFFFFF;">{}</h2>
                    </div>
                    """.format(correct_message),
                    unsafe_allow_html=True
                )

            

            st.header("Feedback:")
            for f in st.session_state.feedback:
                # Determine the colors based on whether the answer is correct or not
                question_color = 'white'
                selected_option_color = 'green' if f['selected_option'] == f['correct_option'] else 'red'
                explanation_color = 'white'
                emoji = '✅' if f['selected_option'] == f['correct_option'] else '❌'
                
                st.markdown(
                    f"""
                    <div style="margin-bottom: 20px;">
                        <strong style="color: {question_color};">Q{f['question_no']}: {f['question']}</strong><br>
                        <span style="color: {selected_option_color};">{emoji} Selected Option: {f['selected_option']}</span><br>
                        <span style="color: {question_color};">Correct Option: {f['correct_option']}</span><br>
                        <span style="color: {explanation_color};">Explanation: {f['explanation']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

