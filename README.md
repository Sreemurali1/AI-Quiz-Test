# AI-Quiz-Test
The AI-Generated Quiz System automates quiz creation and management from user-uploaded documents. It features "Test Your Knowledge," enabling users to take quizzes post-generation. The system offers instant feedback, displays the final score, and provides detailed explanations, enhancing the learning experience efficiently.The system integrates advanced AI models with cloud services to ensure accurate and contextually relevant quiz generation, making it a valuable tool for educators, trainers, and students alike.

## Key Features

### Document-Based Quiz Generation
- **PDF Upload**: Users can upload PDF documents, from which the system extracts content to generate quiz questions.
- **Focused Content**: Quizzes are based on specific lessons or topics defined by the user, ensuring relevance and accuracy.

### Multiple-Choice Questions (MCQs)
- **Question Structure**: The system generates MCQs, each with one correct answer and three distractors.
- **Explanations**: Each question includes a detailed explanation to enhance understanding and learning outcomes.

### Customizable Quiz Parameters
- **Question Count**: Users can specify the number of questions they wish to generate.
- **Varied Difficulty**: The system creates questions ranging from easy to hard, offering a comprehensive assessment.

### Advanced AI Integration
- **AI Models**: Uses Google Generative AI for content generation and Cohere for embeddings, improving the relevance and precision of quiz content.
- **Vector Storage**: Qdrant is used for storing embeddings and performing similarity searches, ensuring questions align with user queries.

### Interactive Quiz Interface
- **Web Interface**: Quizzes can be taken directly within the Streamlit-based interface.
- **Immediate Feedback**: The system provides instant feedback, highlighting correct answers in green and incorrect ones in red, along with detailed explanations.

### Database Management
- **Persistent Storage**: Quizzes are stored in MongoDB, allowing users to retrieve and retake quizzes anytime.
- **Efficient Retrieval**: The system is designed to ensure efficient storage and quick retrieval of quiz data.

### Error Handling and Logging
- **Robust Error Handling**: Comprehensive error handling mechanisms inform users of any issues during quiz generation or retrieval.
- **Logging**: Implemented logging helps monitor system performance and facilitates troubleshooting.

## Technical Stack

- **Python**: Core programming language for backend development.
- **Langchain & Qdrant**: Utilized for document processing, embeddings, and vector storage.
- **Google Generative AI & Cohere**: Power the AI-driven content generation.
- **MongoDB**: Database for storing and managing quiz data.
- **Streamlit**: Framework for building an interactive web interface.
- **Docker**: Used for containerizing the application, ensuring consistency across different environments.

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sreemurali1/AI-Quiz-Test.git
   ```
   
2. **Navigate to the Project Directory**:
   ```bash
   cd AI-Quiz-Test
   ```
   
3. **Install the Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Docker (Optional)**:
   - Build and run the Docker container:
     ```bash
     docker build -t AI-Quiz-Test .
     docker run -p 8501:8501 AI-Quiz-Test
     ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload PDF**: Upload a document to generate quiz questions.
2. **Customize Parameters**: Set the number of questions and specify the lesson or topic.
3. **Take the Quiz**: Answer the questions in the interactive interface.
4. **Review Feedback**: Get immediate feedback with explanations for each question.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
