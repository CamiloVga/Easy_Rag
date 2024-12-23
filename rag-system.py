"""
RAG system module for document processing and question answering.
"""
import os
import logging
from typing import List, Dict
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class RAGSystem:
    """Enhanced RAG system with dynamic document loading."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize RAG system.
        
        Args:
            model_name (str): Name of the HuggingFace model to use
        """
        self.model_name = model_name
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.tokenizer = None
        self.model = None
        self.is_initialized = False
        self.processed_files = set()
    
    def initialize_model(self):
        """Initialize the language model and tokenizer."""
        try:
            logger.info("Initializing language model...")
            
            # Get HuggingFace token
            hf_token = os.environ.get('HUGGINGFACE_TOKEN')
            if not hf_token:
                raise ValueError("No Hugging Face token found. Please set HUGGINGFACE_TOKEN in your environment variables")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=hf_token,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            
            # Create generation pipeline
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.15,
                device_map="auto"
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.is_initialized = True
            
            logger.info("Model initialization completed")
            
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}")
            raise

    def process_documents(self, documents: List) -> None:
        """
        Process documents and update the vector store.
        
        Args:
            documents (List): List of documents to process
        """
        try:
            from src.text_processor import TextProcessor
            
            if not documents:
                logger.warning("No documents to process")
                return
                
            processor = TextProcessor()
            chunks = processor.process_documents(documents)
            
            if not self.vector_store:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vector_store.add_documents(chunks)
            
            # Initialize QA chain
            prompt_template = """
            Context: {context}
            
            Based on the provided context, please answer the following question clearly and concisely.
            If the information is not in the context, please say so explicitly.
            
            Question: {question}
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 6}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            logger.info(f"Successfully processed {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def generate_response(self, question: str) -> Dict:
        """
        Generate response for a given question.
        
        Args:
            question (str): Question to answer
            
        Returns:
            Dict: Response containing answer and sources
        """
        if not self.is_initialized or self.qa_chain is None:
            return {
                'answer': "Please upload some documents first before asking questions.",
                'sources': []
            }
        
        try:
            result = self.qa_chain({"query": question})
            
            response = {
                'answer': result['result'],
                'sources': []
            }
            
            for doc in result['source_documents']:
                source = {
                    'title': doc.metadata.get('title', 'Unknown'),
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata
                }
                response['sources'].append(source)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise