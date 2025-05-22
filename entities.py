import spacy
import re
import json
from typing import List, Dict, Any
from tqdm import tqdm
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langchain.prompts import ChatPromptTemplate

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
        self.nlp = spacy.load("en_core_web_sm")  # or "en_core_web_md" for better but slower results
    
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
                    log_file.write(f"Entities: {json.dumps(doc_entities)}\n")
            except Exception as e:
                print(f"Error logging entities: {e}")
    
    return entities


def extract_entities_rule_based(self, documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract entities using simple rules and regex patterns.
    Very fast but less accurate than model-based approaches.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of extracted entities with their relationships
    """
    entities = []
    
    # Define regex patterns for common entity types
    patterns = {
        "person": [
            r'(?:[A-Z][a-z]+\s+[A-Z][a-z]+)', # Simple name pattern
            r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+'  # Titles
        ],
        "organization": [
            r'(?:[A-Z][a-zA-Z]*\s+(?:Inc\.|Corp\.|LLC|Company|Ltd\.?))',
            r'(?:[A-Z][a-zA-Z\&\s]+(?:Inc\.|Corp\.|LLC|Company|Ltd\.?))'
        ],
        "location": [
            r'(?:[A-Z][a-z]+,\s+[A-Z]{2})',  # City, State
            r'(?:[A-Z][a-z]+\s+[A-Z][a-z]+\s+Park|Forest|Mountain|River|Lake)'
        ],
        "date": [
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ],
        "email": [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
        "url": [r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'],
        "phone": [r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'],
        "money": [r'\$\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?\s*(?:dollars|USD|EUR|GBP)']
    }
    
    # Process each document
    for i, doc in enumerate(tqdm(documents)):
        doc_id = doc.metadata.get("source", f"doc_{i}")
        doc_entities = []
        
        # Apply each pattern to find entities
        for entity_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                for match in re.finditer(pattern, doc.page_content):
                    entity = {
                        "entity": match.group(),
                        "type": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "document_id": doc_id,
                        "confidence": 0.7  # Estimated confidence score
                    }
                    doc_entities.append(entity)
        
        # Simple relationship extraction (entities within N characters)
        for i in range(len(doc_entities)):
            for j in range(i+1, len(doc_entities)):
                e1 = doc_entities[i]
                e2 = doc_entities[j]
                
                # If entities are within 100 characters of each other, consider them related
                if abs(e1["end"] - e2["start"]) < 100 or abs(e2["end"] - e1["start"]) < 100:
                    # Get context between entities
                    start = min(e1["start"], e2["start"])
                    end = max(e1["end"], e2["end"])
                    context = doc.page_content[start:end].lower()
                    
                    # Determine relationship type
                    rel_type = "related_to"
                    if e1["type"] == "person" and e2["type"] == "organization":
                        if any(word in context for word in ["work", "employ", "lead", "head", "ceo", "founder"]):
                            rel_type = "works_at"
                    elif e1["type"] == "person" and e2["type"] == "person":
                        if any(word in context for word in ["colleague", "coworker", "friend", "partner"]):
                            rel_type = "associated_with"
                    
                    relationship = {
                        "source": e1["entity"],
                        "target": e2["entity"],
                        "type": rel_type,
                        "document_id": doc_id
                    }
                    doc_entities.append(relationship)
        
        entities.extend(doc_entities)
        
        # Log entities
        try:
            with open("logs/entities.txt", "a") as log_file:
                log_file.write(f"Entities: {json.dumps(doc_entities)}\n")
        except Exception as e:
            print(f"Error logging entities: {e}")
    
    return entities


def extract_entities_tfidf(self, documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract entities using TF-IDF to identify important terms and phrases.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of extracted entities with their relationships
    """
    entities = []
    
    # Create corpus from documents
    corpus = [doc.page_content for doc in documents]
    
    # Extract n-grams (1-3 words)
    vectorizer = TfidfVectorizer(
        max_df=0.7,         # Ignore terms that appear in >70% of docs
        min_df=1,           # Include terms that appear in at least 1 doc
        max_features=1000,  # Limit vocabulary size
        stop_words='english', 
        ngram_range=(1, 3)  # Use unigrams, bigrams, and trigrams
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    
    # Process each document
    for i, doc in enumerate(tqdm(documents)):
        doc_id = doc.metadata.get("source", f"doc_{i}")
        doc_entities = []
        
        # Get top keywords for this document (adjust the number for more or fewer entities)
        top_n = 20
        doc_vector = tfidf_matrix[i].toarray()[0]
        top_indices = doc_vector.argsort()[-top_n:][::-1]
        
        # Extract entities based on TF-IDF score
        for idx in top_indices:
            term = feature_names[idx]
            score = doc_vector[idx]
            
            # Skip very common words and very short terms
            if score < 0.1 or len(term) < 3:
                continue
                
            # Determine entity type (crude heuristic)
            entity_type = "keyword"
            
            # Check if term appears to be a named entity
            if term[0].isupper():
                words = term.split()
                if len(words) > 1 and all(w[0].isupper() for w in words):
                    entity_type = "organization"  # Multi-word capitalized phrase
                else:
                    entity_type = "concept"  # Single capitalized word
            
            # Find locations in document
            for match in re.finditer(re.escape(term), doc.page_content, re.IGNORECASE):
                entity = {
                    "entity": term,
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "score": float(score),  # TF-IDF score
                    "document_id": doc_id,
                    "confidence": min(0.9, float(score) * 2)  # Convert TF-IDF to confidence
                }
                doc_entities.append(entity)
        
        # Simple co-occurrence based relationships
        text_chunks = []
        chunk_size = 200
        for i in range(0, len(doc.page_content), chunk_size):
            text_chunks.append(doc.page_content[i:i+chunk_size])
        
        # Find entities that co-occur in chunks
        for chunk_idx, chunk in enumerate(text_chunks):
            chunk_entities = []
            
            # Find entities in this chunk
            for entity in doc_entities:
                start, end = entity["start"], entity["end"]
                chunk_start = chunk_idx * chunk_size
                chunk_end = (chunk_idx + 1) * chunk_size
                
                if start >= chunk_start and end <= chunk_end:
                    chunk_entities.append(entity)
            
            # Create relationships between entities in same chunk
            for i in range(len(chunk_entities)):
                for j in range(i+1, len(chunk_entities)):
                    e1 = chunk_entities[i]
                    e2 = chunk_entities[j]
                    
                    # Only create relationships between different entities
                    if e1["entity"] != e2["entity"]:
                        relationship = {
                            "source": e1["entity"],
                            "target": e2["entity"],
                            "type": "co_occurs_with",
                            "document_id": doc_id,
                            "chunk": chunk_idx
                        }
                        doc_entities.append(relationship)
        
        entities.extend(doc_entities)
        
        # Log entities
        try:
            with open("logs/entities.txt", "a") as log_file:
                log_file.write(f"Entities: {json.dumps(doc_entities)}\n")
        except Exception as e:
            print(f"Error logging entities: {e}")
    
    return entities


def extract_entities_hybrid(self, documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Combine rule-based extraction with TF-IDF for better balance of speed and accuracy.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of extracted entities with their relationships
    """
    # First use rule-based extraction to get basic entities
    rule_entities = extract_entities_rule_based(self, documents)
    
    # Then use TF-IDF to identify important keywords that might be missed
    # Creating a simplified TF-IDF extractor
    entities = []
    
    # Create corpus from documents
    corpus = [doc.page_content for doc in documents]
    
    # Extract n-grams (1-2 words)
    vectorizer = TfidfVectorizer(
        max_df=0.7,         # Ignore terms that appear in >70% of docs
        min_df=1,           # Include terms that appear in at least 1 doc
        max_features=500,   # Limit vocabulary size for speed
        stop_words='english', 
        ngram_range=(1, 2)  # Use unigrams and bigrams only
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    
    # Process each document
    for i, doc in enumerate(tqdm(documents)):
        doc_id = doc.metadata.get("source", f"doc_{i}")
        doc_entities = []
        
        # Get top keywords for this document
        top_n = 10
        doc_vector = tfidf_matrix[i].toarray()[0]
        top_indices = doc_vector.argsort()[-top_n:][::-1]
        
        # Extract entities based on TF-IDF score
        for idx in top_indices:
            term = feature_names[idx]
            score = doc_vector[idx]
            
            # Skip very common words and very short terms
            if score < 0.1 or len(term) < 3:
                continue
                
            # Check if this term is already captured by rule-based extraction
            is_duplicate = False
            for entity in rule_entities:
                if 'entity' in entity and entity.get('document_id') == doc_id:
                    if entity['entity'].lower() == term.lower():
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                # Determine entity type (crude heuristic)
                entity_type = "keyword"
                if term[0].isupper():
                    entity_type = "concept"
                
                # Find locations in document
                for match in re.finditer(re.escape(term), doc.page_content, re.IGNORECASE):
                    entity = {
                        "entity": term,
                        "type": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "score": float(score),
                        "document_id": doc_id,
                        "extraction_method": "tfidf"
                    }
                    doc_entities.append(entity)
        
        # Get rule-based entities for this document
        for entity in rule_entities:
            if entity.get('document_id') == doc_id:
                # Mark the extraction method
                entity_copy = entity.copy()
                entity_copy["extraction_method"] = "rule_based"
                doc_entities.append(entity_copy)
        
        # Create relationships between entities 
        # (simple approach: entities within 100 chars of each other)
        for i in range(len(doc_entities)):
            for j in range(i+1, len(doc_entities)):
                e1 = doc_entities[i]
                e2 = doc_entities[j]
                
                # Skip if entities are the same or if any entity is a relationship
                if ('entity' not in e1 or 'entity' not in e2 or
                    e1["entity"] == e2["entity"] or
                    'source' in e1 or 'source' in e2):
                    continue
                
                # Check if entities are close to each other
                if ('start' in e1 and 'end' in e1 and 
                    'start' in e2 and 'end' in e2):
                    if abs(e1["end"] - e2["start"]) < 100 or abs(e2["end"] - e1["start"]) < 100:
                        relationship = {
                            "source": e1["entity"],
                            "target": e2["entity"],
                            "type": "related_to",
                            "document_id": doc_id,
                            "extraction_method": "hybrid"
                        }
                        doc_entities.append(relationship)
        
        entities.extend(doc_entities)
        
        # Log entities
        try:
            with open("logs/entities.txt", "a") as log_file:
                log_file.write(f"Entities: {json.dumps(doc_entities)}\n")
        except Exception as e:
            print(f"Error logging entities: {e}")
    
    return entities


def extract_entities_batch_llm(self, documents: List[Document], batch_size=5) -> List[Dict[str, Any]]:
    """
    Process documents in batches with the LLM to improve efficiency.
    This still uses the LLM but reduces the number of API calls.
    
    Args:
        documents: List of Document objects
        batch_size: Number of documents to process in each LLM call
        
    Returns:
        List of extracted entities with their relationships
    """
    entity_extraction_prompt = ChatPromptTemplate.from_template(
        """Extract entities and their relationships from the following documents.
        For each document, identify key entities such as people, organizations, locations, dates,
        and concepts. Also identify relationships between entities if present.
        
        Return the results in JSON format with 'entities' as a list of objects.
        Each entity should have 'entity', 'type', and 'document_id' fields.
        
        DOCUMENTS:
        {documents}
        
        JSON RESPONSE:
        """
    )
    
    # Create entity extraction chain
    entity_extraction_chain = entity_extraction_prompt | self.llm
    
    entities = []
    
    # Process documents in batches
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:min(i+batch_size, len(documents))]
        
        # Format batch into a string with document IDs
        docs_text = ""
        for j, doc in enumerate(batch):
            doc_id = doc.metadata.get("source", f"doc_{i+j}")
            docs_text += f"DOCUMENT {j+1} (ID: {doc_id}):\n{doc.page_content}\n\n"
        
        # Process batch
        try:
            result = entity_extraction_chain.invoke({"documents": docs_text})
            
            # Clean output
            pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            match = re.search(pattern, result.content)
            
            if match:
                cleaned_content = match.group(1)
            else:
                cleaned_content = result.content
                
            extracted = json.loads(cleaned_content)
            
            # Process entities
            for entity in extracted.get("entities", []):
                entities.append(entity)
                
            # Log successful extraction
            with open("logs/entities.txt", "a") as log_file:
                log_file.write(f"Batch {i//batch_size + 1} Entities: {json.dumps(extracted.get('entities', []))}\n")
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            with open("logs/error.txt", "a") as log_file:
                log_file.write(f"Error processing batch {i//batch_size + 1}: {str(e)}\n")
    
    return entities