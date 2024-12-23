# DocumentGPT - Advanced Document Analysis with RAG

## Try it Live!

🚀 **[Try DocumentGPT on HuggingFace Spaces](https://huggingface.co/spaces/CamiloVega/Easy_RAG)**

Upload your documents and start asking questions right away - no installation required!

## Overview

DocumentGPT is a cutting-edge document analysis system that leverages the power of Retrieval Augmented Generation (RAG) to provide intelligent responses based on your documents. Built with advanced AI technologies, it allows users to upload multiple document types and get accurate, context-aware responses to their questions.

## Key Features

- Multi-Format Support: Process PDF, DOCX, CSV, and TXT files seamlessly
- Advanced RAG Implementation: Using state-of-the-art LLM technology with Llama-2
- GPU-Accelerated: Optimized performance with GPU acceleration
- Real-Time Processing: Dynamic document processing and instant responses
- Source Attribution: Every response includes references to source documents
- Interactive Interface: User-friendly Gradio interface for easy interaction

## Project Structure

```
documentgpt/
│
├── src/                  # Source code
│   ├── document_loader.py
│   ├── rag_system.py
│   └── text_processor.py
├── config/               # Configuration files
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Technical Stack

- Large Language Model: Llama-2-7b-chat-hf
- Embeddings: multilingual-e5-large
- Vector Store: FAISS
- Framework: Gradio
- Processing: Langchain
- Acceleration: HuggingFace Accelerate

## Getting Started

### Prerequisites

- Python 3.8 or higher
- GPU support (recommended)
- HuggingFace account with access to Llama-2

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/documentgpt.git
cd documentgpt
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create a .env file and add your HuggingFace token
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

### Running the Application

```bash
python app.py
```

## Usage

You can either use the [live demo](https://huggingface.co/spaces/CamiloVega/Easy_RAG) on HuggingFace Spaces or run your own instance:

1. Launch the application
2. Upload your documents (PDF, DOCX, CSV, or TXT)
3. Wait for the processing to complete
4. Start asking questions about your documents
5. View responses with source attributions

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
flake8 .
isort .
```

## Author

**Camilo Vega**
- AI Professor and Solutions Consultant
- LinkedIn: [Camilo Vega](https://www.linkedin.com/in/camilo-vega-169084b1/)
- GitHub: [CamiloVega](https://github.com/camilovegag)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to:
- HuggingFace for providing GPU acceleration support
- Meta AI for the Llama-2 model
- The Langchain community for their excellent tools

## Contact

For questions and support, please reach out through:
- LinkedIn: [Camilo Vega](https://www.linkedin.com/in/camilo-vega-169084b1/)

---
Made by Camilo Vega