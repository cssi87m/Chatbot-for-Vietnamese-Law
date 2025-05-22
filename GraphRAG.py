"""
Graph RAG Pipeline using LangChain

This implementation demonstrates a complete Graph RAG (Retrieval-Augmented Generation) pipeline
that combines graph databases with vector embeddings for more contextual retrieval.

The pipeline includes:
1. Document loading and processing
2. Text embedding using BGE-M3 embeddings
3. Graph database integration with Neo4j
4. Query processing with graph traversal
5. LLM response generation using retrieved context

Requirements:
- langchain
- langchain-ollama
- langchain-community
- neo4j
- python-dotenv
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import re
import json
import spacy
from tqdm.auto import tqdm

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_ollama import ChatOllama
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.schema.runnable import RunnablePassthrough

from utils import ENTITY_EXTRACTION_PROMPT, GRAPH_RAG_CHAIN_PROMPT,EMBEDDING, BATCH_ENTITY_EXTRACTION_PROMPT
# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

print(f"Connecting to Neo4j at {NEO4J_URI} with user {NEO4J_USERNAME}, password {NEO4J_PASSWORD}")
# BGE-M3 embeddings


LANGUAGE_MODEL = ChatOllama(
    model = "llama3.2:1b",
    num_predict = -1,
)
class GraphRAGPipeline:
    """
    A Graph RAG (Retrieval-Augmented Generation) pipeline integrating Neo4j graph database
    with LangChain for document retrieval and LLM generation.
    """

    def __init__(self):
        """Initialize the Graph RAG pipeline components."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize embeddings
        self.embeddings = EMBEDDING
        
        # Initialize LLM
        self.llm = LANGUAGE_MODEL
        
        # Initialize Neo4j connections
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        
        try: 
            # Initialize vector store in Neo4j
            self.vector_store = Neo4jVector.from_existing_index(
                embedding=self.embeddings,
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                index_name="document_embeddings",
                node_label="Document"
            )
        
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            # If the vector store does not exist, create it
            self.vector_store = self.create_vector_database()
        
        # Initialize retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def create_vector_database(self, directory_path: str = 'data/corpus') -> Neo4jVector:
        """
        Create a new vector database in Neo4j.
        
        Returns:
            Neo4jVector instance
        """
        docs = self.load_documents(directory_path)
        docs = self.process_documents(docs)
        vector_store = Neo4jVector.from_documents(
            documents=docs,  # List of LangChain Document objects
            embedding=self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name="document_embeddings",
            node_label="Document"
        )

        print(f"Created vector database with {len(docs)} documents")
        return vector_store

    def load_documents(self, directory_path: str) -> List[Document]:
        """
        Load documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of Document objects
        """
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for processing.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunked_documents = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunked_documents)} chunks")
        return chunked_documents
    
    def extract_entities_spacy(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract entities from documents using spaCy (much faster than LLM).
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of extracted entities with their relationships
        """
        # Load spaCy model if not already loaded
        if not hasattr(self, 'nlp'):
            # Use small model for speed, medium for better accuracy
            self.nlp = spacy.load("en_core_web_lg")  # or "en_core_web_md" for better but slower results
        
        # Define entity types of interest
        entity_types = {
            "PERSON": "person",
            "ORG": "organization",
            "DATE": "date",
            "TIME": "time",
            "EVENT": "event",
            "LAW": "law",
            "NORP": "group",
        }
        
        entities = []
        
        # Process documents in batches for better performance
        batch_size = 5  # Adjust based on memory constraints
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:min(i+batch_size, len(documents))]
            texts = [doc.page_content for doc in batch]
            
            # Process batch through spaCy
            doc_objects = list(self.nlp.pipe(texts))
            
            for j, doc_obj in enumerate(doc_objects):
                doc = batch[j]
                doc_id = doc.metadata.get("source", f"doc_{i+j}")
                doc_entities = []
                
                # Extract named entities
                for ent in doc_obj.ents:
                    print(f"Entity: {ent.text}, Label: {ent.label_}")
                    if ent.label_ in entity_types:
                        entity = {
                            "entity": ent.text,
                            "type": entity_types[ent.label_],
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "document_id": doc_id,
                            "confidence": 0.8  # Estimated confidence score
                        }
                        doc_entities.append(entity)
                
                # Basic relationship extraction (entities in same sentence)
                sentences = list(doc_obj.sents)
                for sent in sentences:
                    sent_entities = []
                    for entity in doc_entities:
                        if entity["start"] >= sent.start_char and entity["end"] <= sent.end_char:
                            sent_entities.append(entity)
                    
                    # Create simple co-occurrence relationships
                    for idx1 in range(len(sent_entities)):
                        for idx2 in range(idx1+1, len(sent_entities)):
                            e1 = sent_entities[idx1]
                            e2 = sent_entities[idx2]
                            
                            # Infer relationship type based on entity types
                            rel_type = "related_to"
                            if e1["type"] == "person" and e2["type"] == "organization":
                                # Check for employment keywords
                                context = sent.text.lower()
                                if any(word in context for word in ["work", "employ", "lead", "head", "ceo", "founder"]):
                                    rel_type = "works_at"
                            
                            relationship = {
                                "source": e1["entity"],
                                "target": e2["entity"],
                                "type": rel_type,
                                "document_id": doc_id
                            }
                            doc_entities.append(relationship)
                
                entities.extend(doc_entities)
                
                # Log entities for debugging
                try:
                    with open("logs/entities.txt", "a") as log_file:
                        log_file.write(f"Entities: {doc_entities}\n")
                except Exception as e:
                    print(f"Error logging entities: {e}")
        
        return entities
        
    def extract_entities_batch_llm(self, documents: List[Document], batch_size=10) -> List[Dict[str, Any]]:
        """
        Process documents in batches with the LLM to improve efficiency.
        This still uses the LLM but reduces the number of API calls.
        
        Args:
            documents: List of Document objects
            batch_size: Number of documents to process in each LLM call
            
        Returns:
            List of extracted entities with their relationships
        """
        cnt_error = 0
        entity_extraction_prompt = ChatPromptTemplate.from_template(
            BATCH_ENTITY_EXTRACTION_PROMPT
        )
        
        # Create entity extraction chain
        entity_extraction_chain = entity_extraction_prompt | self.llm
        
        entities = []
        
        # Process documents in batches
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:min(i+batch_size, len(documents))]
            
            # # Format batch into a string with document IDs
            # docs_text = ""
            # for j, doc in enumerate(batch):
            #     doc_id = doc.metadata.get("source", f"doc_{i+j}")
            #     docs_text += f"DOCUMENT {j+1} (ID: {doc_id}):\n{doc.page_content}\n\n"
            
            # Process batch
            try:
                result = entity_extraction_chain.invoke({"documents": batch})
                
                # Clean output
                pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
                match = re.search(pattern, result.content)
                
                if match:
                    cleaned_content = match.group(1)
                else:
                    cleaned_content = result.content
                cleaned_content = cleaned_content.replace("[...]", "[]")
                extracted = json.loads(cleaned_content)
                
                # Process entities
                results = extracted.get("results", [])
                if results is not []:
                    for result in results: 
                        entities_batch = result.get("entities", [])
                        for entity in entities_batch:
                            entity["document_id"] = batch[0].metadata.get("source", f"doc_{i}")
                            entities.append(entity)
                    
                            # Log successful extraction
                            with open("logs/entities.txt", "a") as log_file:
                                log_file.write(f"Batch {i//batch_size + 1} Entities: {entity}\n")
                    
            except Exception as e:
                cnt_error += 1
                print(f"Error processing batch {i//batch_size + 1}: {e}, number of errors: {cnt_error}")
                with open("logs/error.txt", "a", encoding = "utf-8") as log_file:
                    log_file.write(f"Count: {cnt_error}, Content error: {cleaned_content}\n **** \n")
        
        return entities

    
    def build_knowledge_graph(self, entities: List[Dict[str, Any]]):
        """
        Build a knowledge graph in Neo4j from extracted entities.
        
        Args:
            entities: List of entity dictionaries with relationships
        """
        # Create constraints for faster lookups
        self.graph.query("""
        CREATE CONSTRAINT entity_name IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.name IS UNIQUE
        """)
        
        # Build the graph
        for entity in entities:
            # Create entity node
            entity_properties = {
                "name": entity["id"],
                "type": entity["type"]
            }
            
            # Add attributes
            for key, value in entity.get("attributes", {}).items():
                entity_properties[key] = value
            
            # Create Cypher for the entity
            create_entity_query = f"""
            MERGE (e:Entity {{name: $name}})
            SET e += $properties
            RETURN e
            """
            
            self.graph.query(
                create_entity_query,
                {"name": entity["id"], "properties": entity_properties}
            )
            
            # Create document node if not exists
            if "document_id" in entity:
                self.graph.query(
                    """
                    MERGE (d:Document {id: $doc_id})
                    """,
                    {"doc_id": entity["document_id"]}
                )
                
                # Connect entity to document
                self.graph.query(
                    """
                    MATCH (e:Entity {name: $entity_name})
                    MATCH (d:Document {id: $doc_id})
                    MERGE (e)-[:MENTIONED_IN]->(d)
                    """,
                    {"entity_name": entity["id"], "doc_id": entity["document_id"]}
                )
            
            # Create relationships
            for rel in entity.get("relationships", []):
                self.graph.query(
                    """
                    MATCH (e1:Entity {name: $from_entity})
                    MERGE (e2:Entity {name: $to_entity})
                    MERGE (e1)-[:RELATED {type: $rel_type}]->(e2)
                    """,
                    {
                        "from_entity": entity["id"],
                        "to_entity": rel["target"],
                        "rel_type": rel["type"]
                    }
                )
    
    def index_documents(self, documents: List[Document]):
        """
        Index documents in Neo4j vector store.
        
        Args:
            documents: List of Document objects to index
        """
        # Create vector store if it doesn't exist
        try:
            self.vector_store = Neo4jVector.from_documents(
                documents=documents,
                embedding=self.embeddings,
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                index_name="document_embeddings",
                node_label="Document",
                pre_delete_collection=False  # Set to True to rebuild index
            )
            print(f"Indexed {len(documents)} documents in Neo4j vector store")
        except Exception as e:
            print(f"Error indexing documents: {e}")
    
    def enhance_retriever_with_graph(self):
        """
        Enhance the vector retriever with graph-based context.
        Creates a hybrid retriever that combines vector similarity with graph traversal.
        """
        # Define a graph retrieval function
        def graph_retrieval_compressor(documents: List[Document], query: str) -> List[Document]:
            """Use graph relationships to enhance retrieved documents."""
            if not documents:
                return documents
                
            # Extract entity mentions from the query using simple keyword matching
            entity_mentions = []
            # In a real implementation, use NER here
            for doc in documents:
                source_id = doc.metadata.get("source", "")
                
                # Find related entities through graph traversal
                related_docs_query = """
                MATCH (d:Document {id: $doc_id})<-[:MENTIONED_IN]-(e:Entity)
                MATCH (e)-[:RELATED]->(related:Entity)-[:MENTIONED_IN]->(relatedDoc:Document)
                WHERE relatedDoc.id <> $doc_id
                RETURN relatedDoc.id AS related_doc_id, 
                       relatedDoc.content AS content,
                       collect(related.name) AS related_entities
                LIMIT 3
                """
                
                results = self.graph.query(related_docs_query, {"doc_id": source_id})
                
                # Add relationship context to document metadata
                if results:
                    for result in results:
                        doc.metadata["graph_context"] = {
                            "related_entities": result.get("related_entities", []),
                            "related_doc_id": result.get("related_doc_id")
                        }
            
            return documents
        
        # Create a compressor pipeline
        compressor = DocumentCompressorPipeline(
            transformers=[graph_retrieval_compressor]
        )
        
        # Create a contextual compression retriever
        self.enhanced_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever
        )
    
    def create_rag_chain(self):
        """
        Create the RAG chain combining retrieval with generation.
        """
        # Define prompt template
        prompt = ChatPromptTemplate.from_template(GRAPH_RAG_CHAIN_PROMPT)
        
        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Function to extract graph context for the prompt
        def get_graph_context(query: str) -> str:
            # Query the graph for relevant entity relationships
            entity_query = """
            WITH $query AS q
            CALL db.index.fulltext.queryNodes("entity_text", q, {limit: 5}) YIELD node AS entity
            MATCH (entity)-[r:RELATED]->(related:Entity)
            RETURN entity.name AS entity, 
                   collect(related.name) AS related_entities,
                   collect(type(r)) AS relationship_types
            LIMIT 3
            """
            
            try:
                results = self.graph.query(entity_query, {"query": query})
                if results:
                    context_parts = []
                    for result in results:
                        entity = result.get("entity", "")
                        related = result.get("related_entities", [])
                        rel_types = result.get("relationship_types", [])
                        
                        if entity and related:
                            relations = [f"{entity} {rel_types[i]} {related[i]}" 
                                        for i in range(min(len(related), len(rel_types)))]
                            context_parts.append(", ".join(relations))
                    
                    return " | ".join(context_parts)
                return "No graph context found"
            except Exception as e:
                print(f"Error getting graph context: {e}")
                return "Error retrieving graph context"
        
        # Create retrieval chain
        self.rag_chain = create_retrieval_chain(
            self.enhanced_retriever, 
            document_chain
        )
        
        # Augment with graph context
        self.augmented_rag_chain = {
            "query": RunnablePassthrough(),
            "context": self.enhanced_retriever,
            "graph_context": get_graph_context
        } | document_chain
    
    def query(self, query: str) -> str:
        """
        Query the Graph RAG pipeline.
        
        Args:
            query: User query string
            
        Returns:
            Generated response from the LLM
        """
        try:
            result = self.augmented_rag_chain.invoke(query)
            return result
        except Exception as e:
            print(f"Error during query: {e}")
            return f"Error processing query: {str(e)}"
    
    def setup_complete_pipeline(self, docs_directory: str):
        """
        Set up the complete Graph RAG pipeline from documents to queryable chain.
        
        Args:
            docs_directory: Directory containing documents to process
        """
        # 1. Load documents
        documents = self.load_documents(docs_directory)
        
        # 2. Process and chunk documents
        chunked_docs = self.process_documents(documents)
        
        # 3. Extract entities
        entities = self.extract_entities(chunked_docs)
        
        # 4. Build knowledge graph
        self.build_knowledge_graph(entities)
        
        # 5. Index documents in vector store
        self.index_documents(chunked_docs)
        
        # 6. Enhance retriever with graph context
        self.enhance_retriever_with_graph()
        
        # 7. Create RAG chain
        self.create_rag_chain()
        print("Graph RAG pipeline setup complete!")


def main(): 
    # Initialize pipeline
    graph_rag = GraphRAGPipeline()
    # Set up the complete pipeline with a directory of documents
    documents = graph_rag.load_documents("data/corpus")
    documents = graph_rag.process_documents(documents)
    entities = graph_rag.extract_entities_batch_llm(documents, batch_size=50)
    
    # # Query the pipeline
    # query = "How do transformers work in relation to LLMs?"
    # response = graph_rag.query(query)
    # print(f"\nQuery: {query}")
    # print(f"\nResponse: {response}")

    # # Interactive query loop
    # while True:
    #     user_query = input("\nEnter your question (or 'exit' to quit): ")
    #     if user_query.lower() == 'exit':
    #         break
    #     response = graph_rag.query(user_query)
    #     print(f"\nResponse: {response}")
if __name__ == "__main__":
    main()