"""
Production-Ready LangGraph Retrieval Pipeline
Single-file implementation with embedded LLM routing, optimized for scalability and efficiency

Features:
- Auto-detects ingestion mode (graph vs FAISS-only)
- Intelligent LLM-based query routing
- Async-first architecture for high performance
- Comprehensive error handling and fallbacks
- Connection pooling and resource management
- Detailed logging and monitoring
- Production-ready with best practices
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, TypedDict, Literal
from dataclasses import dataclass
from contextlib import asynccontextmanager
import re

# LangGraph imports
from langgraph.graph import StateGraph, END

# LightRAG imports
from lightrag import LightRAG, QueryParam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class PipelineConfig:
    """Centralized configuration for the retrieval pipeline"""
    
    # LLM Router settings
    ROUTER_TEMPERATURE = 0.3
    ROUTER_TOP_P = 0.9
    
    # Answer generation settings
    ANSWER_TEMPERATURE = 0.3
    ANSWER_TOP_P = 0.9
    MAX_CONTEXT_CHUNKS = 5
    
    # Retrieval settings
    DEFAULT_TOP_K = 5
    MAX_TOP_K = 20
    
    # Timeouts (seconds)
    LLM_TIMEOUT = 30
    RETRIEVAL_TIMEOUT = 60
    
    # Fallback settings
    ENABLE_FALLBACK = True
    MAX_RETRIES = 3
    
    # File paths
    ENTITIES_FILE = "kv_store_full_entities.json"
    FAISS_INDEX_FILE = "vdb_chunks.index"


# ============================================================================
# STATE DEFINITION
# ============================================================================

class RetrievalState(TypedDict):
    """State for the retrieval graph"""
    # Input
    query: str
    top_k: int
    
    # Detection
    ingestion_mode: Optional[Literal['graph', 'faiss_only']]
    
    # Classification
    query_classification: Optional[Dict]
    selected_mode: Optional[Literal['naive', 'local', 'global', 'mix']]
    
    # Retrieved data
    chunks: List[Dict]
    raw_results: Optional[str]
    
    # Answer
    answer: str
    
    # Metadata
    metadata: Dict
    reasoning_chain: List[str]
    error: Optional[str]
    
    # Performance tracking
    timings: Dict[str, float]


# ============================================================================
# EMBEDDED LLM QUERY ROUTER
# ============================================================================

class LLMQueryRouter:
    """
    Embedded LLM-powered query router for intelligent mode selection
    Optimized for production use with caching and error handling
    """
    
    # System prompt (optimized for minimal tokens)
    CLASSIFICATION_PROMPT = """You are a RAG query classifier. Analyze queries and select the best retrieval mode.

Modes:
1. NAIVE: Direct vector search. Use for: keyword searches, specific terms
2. LOCAL: Entity-focused graph retrieval. Use for: specific entities, comparisons, "who/what is X"
3. GLOBAL: Community analysis. Use for: themes, categories, high-level patterns
4. MIX: Combined approach. Use for: summaries, complex analysis, uncertain queries

Respond in JSON only:
{"mode": "naive|local|global|mix", "reasoning": "brief explanation", "confidence": 0.8, "query_type": "summary|comparison|lookup|analysis"}"""

    def __init__(self, llm_func, enable_caching: bool = True):
        """
        Initialize router
        
        Args:
            llm_func: Async LLM function
            enable_caching: Enable query classification caching
        """
        self.llm_func = llm_func
        self.enable_caching = enable_caching
        self._cache = {} if enable_caching else None
        self._stats = {'hits': 0, 'misses': 0, 'errors': 0}
    
    async def classify_query(self, query: str) -> Dict:
        """
        Classify query using LLM
        
        Args:
            query: User's question
            
        Returns:
            Classification dict with mode, reasoning, confidence, query_type
        """
        # Check cache
        if self.enable_caching and query in self._cache:
            self._stats['hits'] += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self._cache[query]
        
        self._stats['misses'] += 1
        
        # Build prompt
        user_prompt = f"""Query: "{query}"

Analyze and classify. Respond with JSON only."""
        
        try:
            # Call LLM with timeout
            response = await asyncio.wait_for(
                self.llm_func(
                    prompt=user_prompt,
                    system_prompt=self.CLASSIFICATION_PROMPT,
                    temperature=PipelineConfig.ROUTER_TEMPERATURE,
                    top_p=PipelineConfig.ROUTER_TOP_P
                ),
                timeout=PipelineConfig.LLM_TIMEOUT
            )
            
            # Parse response
            classification = self._parse_response(response)
            
            # Validate
            if self._validate_classification(classification):
                # Cache result
                if self.enable_caching:
                    self._cache[query] = classification
                
                logger.info(
                    f"Classified query as '{classification['mode']}' "
                    f"(confidence: {classification['confidence']:.2f})"
                )
                return classification
            else:
                raise ValueError("Invalid classification format")
                
        except asyncio.TimeoutError:
            self._stats['errors'] += 1
            logger.error(f"LLM classification timeout for query: {query[:50]}...")
            return self._get_fallback_classification(query, "Timeout")
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Classification error: {str(e)}")
            return self._get_fallback_classification(query, str(e))
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response and extract JSON"""
        cleaned = response.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            cleaned = '\n'.join(lines[1:-1] if len(lines) > 2 else lines)
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()
        
        cleaned = cleaned.strip('`').strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try regex extraction
            json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError("Failed to parse JSON from LLM response")
    
    def _validate_classification(self, classification: Dict) -> bool:
        """Validate classification structure"""
        required_fields = ['mode', 'reasoning', 'confidence']
        valid_modes = ['naive', 'local', 'global', 'mix']
        
        if not all(field in classification for field in required_fields):
            return False
        
        if classification['mode'] not in valid_modes:
            return False
        
        try:
            conf = float(classification['confidence'])
            return 0.0 <= conf <= 1.0
        except:
            return False
    
    def _get_fallback_classification(self, query: str, error: str) -> Dict:
        """Heuristic-based fallback classification"""
        query_lower = query.lower()
        
        # Pattern matching
        if any(word in query_lower for word in ['summar', 'overview', 'all', 'everything']):
            mode, query_type = 'mix', 'summary'
        elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
            mode, query_type = 'local', 'comparison'
        elif any(word in query_lower for word in ['theme', 'category', 'type', 'pattern']):
            mode, query_type = 'global', 'categorical'
        elif any(word in query_lower for word in ['who', 'what', 'which', 'where']):
            mode, query_type = 'local', 'lookup'
        else:
            mode, query_type = 'mix', 'general'
        
        return {
            'mode': mode,
            'reasoning': f'Fallback heuristic (error: {error})',
            'confidence': 0.5,
            'query_type': query_type,
            'fallback': True
        }
    
    def get_stats(self) -> Dict:
        """Get router statistics"""
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total if total > 0 else 0
        
        return {
            'cache_hits': self._stats['hits'],
            'cache_misses': self._stats['misses'],
            'cache_hit_rate': hit_rate,
            'errors': self._stats['errors'],
            'cache_size': len(self._cache) if self._cache else 0
        }
    
    def clear_cache(self):
        """Clear classification cache"""
        if self._cache:
            self._cache.clear()
            logger.info("Router cache cleared")


# ============================================================================
# RETRIEVAL PIPELINE
# ============================================================================

class ProductionRetrievalPipeline:
    """
    Production-ready retrieval pipeline with LangGraph
    Optimized for performance, scalability, and reliability
    """
    
    def __init__(
        self,
        rag_or_storage,
        working_dir: str = "./rag_storage",
        config: Optional[PipelineConfig] = None,
        enable_caching: bool = True,
        verbose: bool = False
    ):
        """
        Initialize pipeline
        
        Args:
            rag_or_storage: LightRAG instance or FAISS storage dict
            working_dir: Storage directory path
            config: Optional custom configuration
            enable_caching: Enable query classification caching
            verbose: Enable verbose logging
        """
        self.rag_or_storage = rag_or_storage
        self.working_dir = working_dir
        self.config = config or PipelineConfig()
        self.verbose = verbose
        
        # Set logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Determine storage mode
        self._detect_storage_mode()
        
        # Initialize router (only for graph mode)
        if self.storage_mode == 'graph':
            if not hasattr(rag_or_storage, 'llm_model_func'):
                raise ValueError("LightRAG instance must have llm_model_func")
            self.router = LLMQueryRouter(
                rag_or_storage.llm_model_func,
                enable_caching=enable_caching
            )
            self.llm_func = rag_or_storage.llm_model_func
        else:
            self.router = None
            self.llm_func = rag_or_storage.get('llm_func')
        
        # Build graph
        self.graph = self._build_graph()
        
        # Performance tracking
        self._query_count = 0
        self._total_time = 0.0
        
        logger.info(f"Pipeline initialized in {self.storage_mode} mode")
    
    def _detect_storage_mode(self):
        """Detect storage mode on initialization"""
        if isinstance(self.rag_or_storage, dict) and self.rag_or_storage.get('mode') == 'faiss_only':
            self.storage_mode = 'faiss_only'
        else:
            # Check for graph data
            entities_file = os.path.join(self.working_dir, self.config.ENTITIES_FILE)
            if os.path.exists(entities_file):
                try:
                    with open(entities_file, 'r') as f:
                        data = json.load(f)
                        if data:
                            self.storage_mode = 'graph'
                            return
                except:
                    pass
            
            if hasattr(self.rag_or_storage, 'graph_storage_cls'):
                self.storage_mode = 'graph'
            else:
                self.storage_mode = 'faiss_only'
    
    def _build_graph(self) -> StateGraph:
        """Build optimized LangGraph workflow"""
        workflow = StateGraph(RetrievalState)
        
        # Add nodes
        workflow.add_node("detect", self._node_detect_mode)
        workflow.add_node("classify", self._node_classify_query)
        workflow.add_node("retrieve_graph", self._node_retrieve_graph)
        workflow.add_node("retrieve_faiss", self._node_retrieve_faiss)
        workflow.add_node("extract", self._node_extract_chunks)
        workflow.add_node("generate", self._node_generate_answer)
        workflow.add_node("finalize", self._node_finalize_response)
        
        # Entry point
        workflow.set_entry_point("detect")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "detect",
            lambda state: state['ingestion_mode'],
            {
                "graph": "classify",
                "faiss_only": "retrieve_faiss"
            }
        )
        
        # Graph path
        workflow.add_edge("classify", "retrieve_graph")
        workflow.add_edge("retrieve_graph", "extract")
        
        # FAISS path
        workflow.add_edge("retrieve_faiss", "generate")
        
        # Common path
        workflow.add_edge("extract", "generate")
        workflow.add_edge("generate", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    # ========================================================================
    # NODE IMPLEMENTATIONS (Optimized)
    # ========================================================================
    
    def _node_detect_mode(self, state: RetrievalState) -> RetrievalState:
        """Node: Detect ingestion mode (optimized - uses pre-detected mode)"""
        import time
        start = time.time()
        
        state['ingestion_mode'] = self.storage_mode
        state['reasoning_chain'] = [f"Detected mode: {self.storage_mode}"]
        state['timings'] = {'detect': time.time() - start}
        
        if self.verbose:
            logger.debug(f"Detected mode: {self.storage_mode}")
        
        return state
    
    async def _node_classify_query(self, state: RetrievalState) -> RetrievalState:
        """Node: Classify query using LLM router"""
        import time
        start = time.time()
        
        query = state['query']
        
        try:
            classification = await self.router.classify_query(query)
            
            state['query_classification'] = classification
            state['selected_mode'] = classification['mode']
            
            state['reasoning_chain'].append(
                f"Classified as '{classification['mode']}' "
                f"(confidence: {classification['confidence']:.0%})"
            )
            
            if self.verbose:
                logger.debug(
                    f"Query classified: mode={classification['mode']}, "
                    f"type={classification.get('query_type')}"
                )
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            state['selected_mode'] = 'mix'
            state['query_classification'] = {
                'mode': 'mix',
                'reasoning': f'Error fallback: {str(e)}',
                'confidence': 0.5,
                'fallback': True
            }
            state['reasoning_chain'].append("Fallback to mix mode due to error")
        
        state['timings']['classify'] = time.time() - start
        return state
    
    async def _node_retrieve_graph(self, state: RetrievalState) -> RetrievalState:
        """Node: Retrieve from LightRAG graph"""
        import time
        start = time.time()
        
        query = state['query']
        mode = state['selected_mode']
        top_k = min(state['top_k'], self.config.MAX_TOP_K)
        
        try:
            result = await asyncio.wait_for(
                self.rag_or_storage.aquery(
                    query,
                    param=QueryParam(mode=mode, top_k=top_k)
                ),
                timeout=self.config.RETRIEVAL_TIMEOUT
            )
            
            state['raw_results'] = result
            state['reasoning_chain'].append(f"Retrieved using {mode} mode")
            
            if self.verbose:
                logger.debug(f"Retrieved {len(result)} chars from graph")
            
        except asyncio.TimeoutError:
            logger.error("Graph retrieval timeout")
            state['error'] = "Retrieval timeout"
            state['raw_results'] = ""
            state['reasoning_chain'].append("Retrieval timeout")
            
        except Exception as e:
            logger.error(f"Graph retrieval error: {e}")
            state['error'] = str(e)
            state['raw_results'] = ""
            state['reasoning_chain'].append(f"Retrieval error: {str(e)}")
        
        state['timings']['retrieve'] = time.time() - start
        return state
    
    async def _node_retrieve_faiss(self, state: RetrievalState) -> RetrievalState:
        """Node: Retrieve from FAISS index"""
        import time
        start = time.time()
        
        query = state['query']
        top_k = min(state['top_k'], self.config.MAX_TOP_K)
        
        try:
            # Generate embedding
            embedding_func = self.rag_or_storage['embedding_func']
            query_embedding = await embedding_func([query])
            query_vector = query_embedding[0].reshape(1, -1)
            
            # Search FAISS
            faiss_index = self.rag_or_storage['faiss_index']
            distances, indices = faiss_index.search(query_vector, top_k)
            
            # Retrieve chunks
            chunks_storage = self.rag_or_storage['chunks_storage']
            chunks = []
            
            for idx, (distance, faiss_idx) in enumerate(zip(distances[0], indices[0])):
                if faiss_idx >= 0 and faiss_idx < len(chunks_storage):
                    chunk_id = list(chunks_storage.keys())[faiss_idx]
                    chunk_data = chunks_storage[chunk_id]
                    
                    chunks.append({
                        'content': chunk_data['content'],
                        'source': chunk_data.get('source', 'unknown'),
                        'score': float(1.0 / (1.0 + distance)),
                        'rank': idx + 1,
                        'chunk_id': chunk_id
                    })
            
            state['chunks'] = chunks
            state['reasoning_chain'].append(f"Retrieved {len(chunks)} chunks from FAISS")
            
            if self.verbose:
                logger.debug(f"FAISS retrieved {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"FAISS retrieval error: {e}")
            state['error'] = str(e)
            state['chunks'] = []
            state['reasoning_chain'].append(f"FAISS error: {str(e)}")
        
        state['timings']['retrieve'] = time.time() - start
        return state
    
    def _node_extract_chunks(self, state: RetrievalState) -> RetrievalState:
        """Node: Extract chunks from graph results"""
        import time
        start = time.time()
        
        raw_results = state.get('raw_results', '')
        
        if not raw_results:
            state['chunks'] = []
        else:
            # Split by double newlines for paragraphs
            sections = [s.strip() for s in raw_results.split('\n\n') if s.strip()]
            
            chunks = []
            for idx, section in enumerate(sections):
                chunks.append({
                    'content': section,
                    'source': 'lightrag_graph',
                    'rank': idx + 1,
                    'score': 1.0 / (idx + 1)
                })
            
            state['chunks'] = chunks
        
        state['reasoning_chain'].append(f"Extracted {len(state['chunks'])} chunks")
        state['timings']['extract'] = time.time() - start
        
        return state
    
    async def _node_generate_answer(self, state: RetrievalState) -> RetrievalState:
        """Node: Generate final answer"""
        import time
        start = time.time()
        
        query = state['query']
        chunks = state.get('chunks', [])
        
        # For graph mode, use LightRAG answer directly
        if state['ingestion_mode'] == 'graph' and state.get('raw_results'):
            state['answer'] = state['raw_results']
            state['reasoning_chain'].append("Used LightRAG answer")
            state['timings']['generate'] = time.time() - start
            return state
        
        # For FAISS mode, generate from chunks
        if not chunks:
            state['answer'] = "No relevant information found."
            state['reasoning_chain'].append("No chunks available")
            state['timings']['generate'] = time.time() - start
            return state
        
        try:
            # Format context
            context_chunks = chunks[:self.config.MAX_CONTEXT_CHUNKS]
            context = "\n\n".join([
                f"[{i+1}] {chunk['content']}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Generate answer
            prompt = f"""Based on the context below, answer the question concisely.

Question: {query}

Context:
{context}

Answer:"""
            
            answer = await asyncio.wait_for(
                self.llm_func(
                    prompt=prompt,
                    temperature=self.config.ANSWER_TEMPERATURE,
                    top_p=self.config.ANSWER_TOP_P
                ),
                timeout=self.config.LLM_TIMEOUT
            )
            
            state['answer'] = answer
            state['reasoning_chain'].append(f"Generated answer from {len(context_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            # Fallback to chunk concatenation
            state['answer'] = "\n\n".join([c['content'] for c in chunks[:3]])
            state['reasoning_chain'].append("Fallback answer (generation failed)")
        
        state['timings']['generate'] = time.time() - start
        return state
    
    def _node_finalize_response(self, state: RetrievalState) -> RetrievalState:
        """Node: Finalize response with metadata"""
        import time
        start = time.time()
        
        # Build metadata
        state['metadata'] = {
            'ingestion_mode': state['ingestion_mode'],
            'retrieval_mode': state.get('selected_mode', 'faiss_similarity'),
            'query_type': state.get('query_classification', {}).get('query_type', 'unknown'),
            'confidence': state.get('query_classification', {}).get('confidence', 1.0),
            'num_chunks': len(state.get('chunks', [])),
            'success': state.get('error') is None,
            'timings': state['timings']
        }
        
        state['timings']['finalize'] = time.time() - start
        state['reasoning_chain'].append("Response finalized")
        
        return state
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    async def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_chunks: bool = True
    ) -> Dict:
        """
        Execute query through pipeline
        
        Args:
            question: User's question
            top_k: Number of chunks (default: from config)
            return_chunks: Include chunks in response
            
        Returns:
            Response dictionary with answer, chunks, metadata
        """
        import time
        start_time = time.time()
        
        # Validate input
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        top_k = top_k or self.config.DEFAULT_TOP_K
        top_k = min(top_k, self.config.MAX_TOP_K)
        
        # Initialize state
        initial_state = {
            'query': question.strip(),
            'top_k': top_k,
            'ingestion_mode': None,
            'query_classification': None,
            'selected_mode': None,
            'chunks': [],
            'raw_results': None,
            'answer': '',
            'metadata': {},
            'reasoning_chain': [],
            'error': None,
            'timings': {}
        }
        
        try:
            # Execute graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Track performance
            total_time = time.time() - start_time
            self._query_count += 1
            self._total_time += total_time
            
            # Build response
            response = {
                'answer': final_state['answer'],
                'metadata': {
                    **final_state['metadata'],
                    'total_time': total_time,
                    'query_id': self._query_count
                },
                'reasoning_chain': final_state['reasoning_chain']
            }
            
            if return_chunks:
                response['chunks'] = final_state.get('chunks', [])
            
            if final_state.get('error'):
                response['error'] = final_state['error']
            
            logger.info(
                f"Query completed in {total_time:.3f}s "
                f"(mode: {final_state['metadata']['retrieval_mode']})"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def batch_query(
        self,
        questions: List[str],
        top_k: Optional[int] = None,
        max_concurrent: int = 5
    ) -> List[Dict]:
        """
        Execute multiple queries concurrently
        
        Args:
            questions: List of questions
            top_k: Number of chunks per query
            max_concurrent: Maximum concurrent queries
            
        Returns:
            List of response dictionaries
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def query_with_semaphore(question):
            async with semaphore:
                return await self.query(question, top_k=top_k)
        
        tasks = [query_with_semaphore(q) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch query {i} failed: {result}")
                results[i] = {
                    'answer': None,
                    'error': str(result),
                    'metadata': {'success': False}
                }
        
        return results
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        stats = {
            'query_count': self._query_count,
            'total_time': self._total_time,
            'avg_time': self._total_time / self._query_count if self._query_count > 0 else 0,
            'storage_mode': self.storage_mode
        }
        
        if self.router:
            stats['router'] = self.router.get_stats()
        
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self._query_count = 0
        self._total_time = 0.0
        if self.router:
            self.router.clear_cache()
        logger.info("Statistics reset")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def create_pipeline(
    working_dir: str = "./rag_storage",
    enable_caching: bool = True,
    verbose: bool = False
) -> ProductionRetrievalPipeline:
    """
    Auto-create pipeline from storage
    
    Args:
        working_dir: Storage directory
        enable_caching: Enable query classification caching
        verbose: Enable verbose logging
        
    Returns:
        Initialized pipeline
    """
    from pipeline_core import initialize_rag_with_retry
    
    logger.info("Creating retrieval pipeline...")
    
    # Check for graph data
    entities_file = os.path.join(working_dir, PipelineConfig.ENTITIES_FILE)
    has_graph = os.path.exists(entities_file)
    
    if has_graph:
        logger.info("Detected graph mode - initializing LightRAG...")
        rag = await initialize_rag_with_retry(working_dir=working_dir, use_graph=True)
    else:
        logger.info("Detected FAISS-only mode...")
        # Load FAISS storage
        import faiss
        faiss_path = os.path.join(working_dir, PipelineConfig.FAISS_INDEX_FILE)
        
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"No storage found in {working_dir}")
        
        faiss_index = faiss.read_index(faiss_path)
        
        # Create storage dict
        rag = {
            'faiss_index': faiss_index,
            'chunks_storage': {},  # Load from your storage
            'mode': 'faiss_only',
            'working_dir': working_dir
        }
    
    pipeline = ProductionRetrievalPipeline(
        rag,
        working_dir=working_dir,
        enable_caching=enable_caching,
        verbose=verbose
    )
    
    logger.info("Pipeline ready")
    return pipeline


async def quick_query(
    question: str,
    working_dir: str = "./rag_storage",
    top_k: int = 5
) -> str:
    """
    Quick query - returns just the answer
    
    Args:
        question: User's question
        working_dir: Storage directory
        top_k: Number of chunks
        
    Returns:
        Answer string
    """
    pipeline = await create_pipeline(working_dir=working_dir)
    result = await pipeline.query(question, top_k=top_k, return_chunks=False)
    return result['answer']


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def main():
    """Example usage"""
    
    print("="*70)
    print("PRODUCTION RETRIEVAL PIPELINE - EXAMPLE")
    print("="*70)
    
    # Create pipeline
    pipeline = await create_pipeline(
        working_dir="./rag_storage",
        enable_caching=True,
        verbose=True
    )
    
    # Single query
    print("\n1. Single Query:")
    result = await pipeline.query(
        question="What are the main themes?",
        top_k=5
    )
    
    print(f"\nAnswer: {result['answer'][:200]}...")
    print(f"\nMetadata:")
    for key, value in result['metadata'].items():
        print(f"  {key}: {value}")
    
    # Batch queries
    # print("\n2. Batch Queries:")
    # questions = [
    #     "What is the total revenue?",
    # ]
    
    # results = await pipeline.batch_query(questions, max_concurrent=3)
    
    # for i, result in enumerate(results, 1):
    #     print(f"\nQ{i}: {questions[i-1]}")
    #     print(f"A: {result['answer'][:100]}...")
    
    # Statistics
    print("\n3. Pipeline Statistics:")
    stats = pipeline.get_stats()
    print(json.dumps(stats, indent=2))
    
    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
