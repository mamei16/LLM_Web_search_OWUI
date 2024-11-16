"""
LLM Web Search
version: 0.2.0

Copyright (C) 2024 mamei16

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from typing import Dict, Tuple, cast, Any, List, Literal, Optional, Union, Callable, Iterable, Sequence, Iterator
from dataclasses import dataclass
import urllib
from urllib.parse import urlparse
import re
import warnings
import copy
from abc import abstractmethod
from collections import defaultdict
from itertools import chain
import asyncio
import concurrent.futures

from pydantic import BaseModel, Field
import aiohttp
import numpy as np
from requests.exceptions import JSONDecodeError
from duckduckgo_search import AsyncDDGS
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_array
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM



async def emit_status(event_emitter, description: str, done: bool):
    if event_emitter:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                },
            }
        )

async def emit_message(event_emitter, content: str):
    if event_emitter:
        await event_emitter(
            {
                "type": "message",
                "data": {
                    "content": content
                },
            }
        )


class Tools:
    class Valves(BaseModel):
        embedding_model_save_path: str = Field(
            default="", description="Path to the folder in which embedding models will be saved"
        )
        num_results: int = Field(
            default=10, description="Number of search engine results to process per query", ge=1
        )
        max_results: int = Field(
            default=8, description="Max. number of search results to return per query", ge=1
        )
        cpu_only: bool = Field(
            default=False, description="Run the tool on CPU only"
        )
        simple_search: bool = Field(
            default=False,
            description="Use just the website snippets returned by the search engine, instead of processing entire webpages",
        )
        chunk_size: int = Field(
            default=500, description="Max. chunk size. The maximal size of the individual chunks that each webpage will"
                                     " be split into, in characters", ge=5, le=100000,
        )
        include_citations: bool = Field(
            default=True, description="Include a citation for each retrieved search result"
         )
        ensemble_weighting: float = Field(
            default=0.5, description="Ensemble Weighting. "
                                     "Smaller values = More keyword oriented, Larger values = More focus on semantic similarity",
            ge=0.0, le=1.0
        )
        keyword_retriever: str = Field(
            default="splade", description="Keyword retriever. Must be either 'bm25' or 'splade'.",
            pattern=r'^(bm25|splade)$'
        )
        splade_batch_size: int = Field(
            default=8, description="SPLADE batch size. Smaller values = Slower retrieval (but lower VRAM usage), "
                                   "Larger values = Faster retrieval (but higher VRAM usage).",
            ge=2, le=1024
        )
        chunker: str = Field(
            default="semantic", description="Chunking method. Must be either 'character-based' or 'semantic'.",
            pattern=r'^(character-based|semantic)$'
        )
        chunker_breakpoint_threshold_amount: int = Field(
            default=30, description="Semantic chunking: sentence split threshold (%)."
                                    "Defines how different two consecutive sentences have"
                                    " to be for them to be split into separate chunks",
            ge=1, le=100
        )
        similarity_score_threshold: float = Field(
            default=0.5, description="Similarity Score Threshold. "
                                     "Discard chunks that are not similar enough to the "
                                     "search query and hence fall below the threshold.",
            ge=0.0, le=1.0
        )
        client_timeout: int = Field(
            default=10, description="Client timeout (in seconds)."
                                    "When reached, pending or unfinished webpage "
                                    "downloads will be cancelled to start the retrieval process immediately",
            ge=0, le=1000
        )
        searxng_url: str = Field(
            default="None", description='SearXNG URL. If not equal to "None", searXNG will be used instead of DuckDuckGo',
        )

    def __init__(self):
        self.valves = self.Valves()
        self.document_retriever = DocumentRetriever()

    @staticmethod
    def reuse_existing_web_search_results(__user__: dict, __event_emitter__=None):
        """
        Choose this tool if existing search results from a previous web search can be used to answer the user's query.
        """
        return ""

    @staticmethod
    def no_tool_necessary(__user__: dict, __event_emitter__=None):
        """
        Choose this tool if you can answer the user without using any tool.
        """
        return ""

    async def search_web(
            self, query: str, __user__: dict, __event_emitter__=None
    ) -> str:
        """
        The search tool will search the web and return the results. You must formulate your own search query based on the user's message.
        """
        self.document_retriever.update_settings(self.valves)

        if self.valves.embedding_model_save_path == "":
            await emit_status(__event_emitter__,
                             "Error: Please configure the embedding model save path", True)
            error_message = ("Error: Please configure the embedding model save path. "
                             "To solve this issue, go to Workspace-->Tools and click on the gear symbol next to the LLM_Web_search tool. "
                             'Then, fill out the field titled "Embedding Model Save Path" with the absolute path to the directory '
                             "n which the embedding models should be stored.")
            await emit_message(__event_emitter__, f"\[ % {error_message}\n \] ")
            return error_message

        try:
            if self.document_retriever.splade_doc_model is None or self.document_retriever.splade_query_model is None or self.document_retriever.embedding_model is None:
                await self.document_retriever.aload_models(__event_emitter__)

            if self.valves.searxng_url != "None":
                result_docs = await self.document_retriever.aretrieve_from_searxng(query, self.valves.simple_search,
                                                                                   __event_emitter__)
            else:
                result_docs = await self.document_retriever.aretrieve_from_duckduckgo(query, self.valves.simple_search,
                                                                                      __event_emitter__)
            source_url_set = list({d.metadata["source"] for d in result_docs})
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "action": "web_search",
                            "description": f"Web search retrieved {len(result_docs)} results from {len(source_url_set)} sources",
                            "done": True,
                            "query": query,
                            "urls": source_url_set
                        },
                    }
                )

            if self.valves.include_citations and __event_emitter__:
                for result_doc in result_docs:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [result_doc.page_content],
                                "metadata": [result_doc.metadata],
                                "source": {"name": urlparse(result_doc.metadata["source"]).netloc},
                            },
                        }
                    )

            pretty_docs_string = docs_to_pretty_str(result_docs)
            formatted_docs_string = pretty_docs_string.replace("\n", "\\n")

            await emit_message(__event_emitter__, f"\[ % {formatted_docs_string}\n \] ")
            return pretty_docs_string
        except Exception as exc:
            exception_message = str(exc)
            await emit_status(__event_emitter__,
                             f'The search tool encountered an error: {exception_message}',
                             True)
            return f"The search tool encountered an error: {exception_message}"


def load_splade_model(repo_id: str, cache_dir: str, device: str):
    kwargs = {"cache_dir": cache_dir, "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
              "attn_implementation": "eager"}
    try:
        return AutoTokenizer.from_pretrained(
            repo_id, cache_dir=cache_dir
        ), AutoModelForMaskedLM.from_pretrained(
            repo_id,
            local_files_only=True, **kwargs
        )
    except OSError:
        return AutoTokenizer.from_pretrained(
            repo_id, cache_dir=cache_dir
        ), AutoModelForMaskedLM.from_pretrained(
            repo_id, **kwargs
        )


def load_embedding_model(repo_id: str, cache_dir: str, device: str):
    return SentenceTransformer(repo_id, cache_folder=cache_dir,
                               device=device,
                               model_kwargs={"torch_dtype": torch.float32 if device == "cpu" else torch.float16})


@dataclass
class Document:
    page_content: str
    metadata: Dict


class DocumentRetriever:
    spaces_regex: re.Pattern
    device: str
    model_cache_dir: str
    num_results: int
    max_results: int
    similarity_threshold: float
    keyword_retriever: str
    chunking_method: str
    chunk_size: int
    chunker_breakpoint_threshold_amount: int
    ensemble_weighting: float
    client_timeout: int
    searxng_url: str
    splade_batch_size: int

    def __init__(self):
        self.embedding_model = None
        self.splade_doc_tokenizer = None
        self.splade_doc_model = None
        self.splade_query_tokenizer = None
        self.splade_query_model = None
        self.spaces_regex = re.compile(r" {3,}")

    def update_settings(self, settings: Tools.Valves):
        self.device = "cpu" if settings.cpu_only else "cuda"
        self.model_cache_dir = settings.embedding_model_save_path
        self.num_results = settings.num_results
        self.max_results = settings.max_results
        self.similarity_threshold = settings.similarity_score_threshold
        self.keyword_retriever = settings.keyword_retriever
        self.chunking_method = settings.chunker
        self.chunk_size = settings.chunk_size
        self.chunker_breakpoint_threshold_amount = settings.chunker_breakpoint_threshold_amount
        self.ensemble_weighting = settings.ensemble_weighting
        self.client_timeout = settings.client_timeout
        self.searxng_url = settings.searxng_url
        self.splade_batch_size = settings.splade_batch_size

    async def aload_models(self, __event_emitter__):
        await emit_status(__event_emitter__, "Loading embedding model 1/3...", False)

        self.embedding_model = await asyncio.to_thread(load_embedding_model, "all-MiniLM-L6-v2",
                                                       self.model_cache_dir, self.device)
        self.embedding_model.to(self.device)

        await emit_status(__event_emitter__, "Loading embedding model 2/3...", False)
        self.splade_doc_tokenizer, self.splade_doc_model = await asyncio.to_thread(load_splade_model,
                                                                                   "naver/efficient-splade-VI-BT-large-doc",
                                                                                   self.model_cache_dir,
                                                                                   self.device)
        self.splade_doc_model.to(self.device)

        await emit_status(__event_emitter__, "Loading embedding model 3/3...", False)
        self.splade_query_tokenizer, self.splade_query_model = await asyncio.to_thread(load_splade_model,
                                                                                       "naver/efficient-splade-VI-BT-large-query",
                                                                                       self.model_cache_dir, self.device
        )
        self.splade_query_model.to(self.device)


    async def aretrieve_from_duckduckgo(self, query: str, simple_search: bool, event_emitter):
        documents = []
        query = query.strip("\"'")
        max_results = self.max_results
        await emit_status(event_emitter, f'Searching DuckDuckGo for "{query}"...', False)

        with AsyncDDGS() as ddgs:
            answer_list = await ddgs.aanswers(query)
            if answer_list:
                if max_results > 1:
                    max_results -= 1  # We already have 1 result now
                answer_dict = answer_list[0]
                instant_answer_doc = Document(page_content=answer_dict["text"],
                                              metadata={"source": answer_dict["url"]})
                documents.append(instant_answer_doc)

            result_documents = []
            result_urls = []
            for result in await ddgs.atext(query, region='wt-wt', safesearch='moderate', timelimit=None,
                                           max_results=self.num_results):
                result_document = Document(page_content=f"Title: {result['title']}\n{result['body']}",
                                           metadata={"source": result["href"]})
                result_documents.append(result_document)
                result_urls.append(result["href"])

        if simple_search:
            retrieved_docs = await self.aretrieve_from_snippets(query, result_documents, event_emitter)
        else:
            retrieved_docs = await self.aretrieve_from_webpages(query, result_urls, event_emitter)

        documents.extend(retrieved_docs)

        if not documents:  # Fall back to old simple search rather than returning nothing
            print("LLM_Web_search | Could not find any page content "
                  "similar enough to be extracted, using basic search fallback...")
            return result_documents[:max_results]
        return documents[:max_results]

    async def aretrieve_from_searxng(self, query: str, simple_search: bool, event_emitter):
        await emit_status(event_emitter, f'Searching SearXNG for "{query}"...', False)

        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
                   "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                   "Accept-Language": "en-US,en;q=0.5"}
        result_documents = []
        result_urls = []
        request_str = f"/search?q={urllib.parse.quote(query)}&format=json&pageno="
        pageno = 1
        url = self.searxng_url if self.searxng_url.startswith('http') else ('http://' + self.searxng_url)
        async with aiohttp.ClientSession(headers=headers) as session:
            while len(result_urls) < self.num_results:
                response = await session.get(url + request_str + str(pageno))

                if not result_urls:  # no results to lose by raising an exception here
                    response.raise_for_status()
                try:
                    response_dict = await response.json()
                except JSONDecodeError:
                    raise ValueError(
                        "JSONDecodeError: Please ensure that the SearXNG instance can return data in JSON format")

                result_dicts = response_dict["results"]
                if not result_dicts:
                    break
                for result in result_dicts:
                    if "content" in result:  # Since some websites don't provide any description
                        result_document = Document(page_content=f"Title: {result['title']}\n{result['content']}",
                                                   metadata={"source": result["url"]})
                        result_documents.append(result_document)
                    result_urls.append(result["url"])

                answers = response_dict["answers"]
                for answer in answers:
                    answer_document = Document(page_content=f"Title: {query}\n{answer}",
                                               metadata={"source": "SearXNG instant answer"})
                    result_documents.append(answer_document)
                pageno += 1

        if simple_search:
            retrieved_docs = await self.aretrieve_from_snippets(query, result_documents, event_emitter)
        else:
            retrieved_docs = await self.aretrieve_from_webpages(query, result_urls, event_emitter)

        return retrieved_docs[:self.max_results]

    def preprocess_text(self, text: str) -> str:
        text = text.replace("\n", " \n")
        text = self.spaces_regex.sub(" ", text)
        text = text.strip()
        return text

    async def aretrieve_from_snippets(self, query: str, documents: list[Document], event_emitter) -> list[Document]:
        await emit_status(event_emitter, "Retrieving relevant results...", False)

        dense_retriever = DenseRetriever(self.embedding_model, num_results=self.num_results,
                                         similarity_threshold=self.similarity_threshold)
        dense_retriever.add_documents(documents)
        return dense_retriever.get_relevant_documents(query)

    async def aretrieve_from_webpages(self, query: str, url_list: list[str], event_emitter) -> list[Document]:
        if self.chunking_method == "semantic":
            text_splitter = BoundedSemanticChunker(self.embedding_model, breakpoint_threshold_type="percentile",
                                                   breakpoint_threshold_amount=self.chunker_breakpoint_threshold_amount,
                                                   max_chunk_size=self.chunk_size)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=10,
                                                           separators=["\n\n", "\n", ".", ", ", " ", ""])

        await emit_status(event_emitter, "Downloading and chunking webpages...", False)
        split_docs = await async_fetch_chunk_websites(url_list, text_splitter, self.client_timeout)

        await emit_status(event_emitter, "Retrieving relevant results...", False)
        if self.ensemble_weighting > 0:
            dense_retriever = DenseRetriever(self.embedding_model, num_results=self.num_results,
                                             similarity_threshold=self.similarity_threshold)
            dense_retriever.add_documents(split_docs)
            dense_result_docs = dense_retriever.get_relevant_documents(query)
        else:
            dense_result_docs = []

        if self.ensemble_weighting < 1:
            #  The sparse keyword retriever is good at finding relevant documents based on keywords,
            #  while the dense retriever is good at finding relevant documents based on semantic similarity.
            if self.keyword_retriever == "bm25":
                keyword_retriever = BM25Retriever.from_documents(split_docs,
                                                                 preprocess_func=self.preprocess_text)
                keyword_retriever.k = self.num_results
            elif self.keyword_retriever == "splade":
                keyword_retriever = SpladeRetriever(
                    splade_doc_tokenizer=self.splade_doc_tokenizer,
                    splade_doc_model=self.splade_doc_model,
                    splade_query_tokenizer=self.splade_query_tokenizer,
                    splade_query_model=self.splade_query_model,
                    device=self.device,
                    batch_size=self.splade_batch_size,
                    k=self.num_results
                )
                await asyncio.to_thread(keyword_retriever.add_documents, split_docs)
            else:
                raise ValueError("self.keyword_retriever must be one of ('bm25', 'splade')")
            sparse_results_docs = await asyncio.to_thread(keyword_retriever.get_relevant_documents, query)
        else:
            sparse_results_docs = []

        return weighted_reciprocal_rank([dense_result_docs, sparse_results_docs],
                                        weights=[self.ensemble_weighting, 1 - self.ensemble_weighting])[:self.num_results]


def cosine_similarity(X, Y) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    # Ignore divide by zero errors run time warnings as those are handled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


def dict_list_to_pretty_str(data: list[dict]) -> str:
    ret_str = ""
    if isinstance(data, dict):
        data = [data]
    if isinstance(data, list):
        for i, d in enumerate(data):
            ret_str += f"Result {i + 1}\n"
            ret_str += f"Title: {d['title']}\n"
            ret_str += f"{d['body']}\n"
            ret_str += f"Source URL: {d['href']}\n"
        return ret_str
    else:
        raise ValueError("Input must be dict or list[dict]")


class TextSplitter:
    """Interface for splitting text into chunks.
    Source: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/base.py#L30
    """

    def __init__(
            self,
            chunk_size: int = 4000,
            chunk_overlap: int = 200,
            length_function: Callable[[str], int] = len,
            keep_separator: Union[bool, Literal["start", "end"]] = False,
            add_start_index: bool = False,
            strip_whitespace: bool = True,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator and where to place it
                            in each corresponding chunk (True='start')
            add_start_index: If `True`, includes chunk's start index in metadata
            strip_whitespace: If `True`, strips whitespace from the start and end of
                              every document
        """
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def create_documents(
            self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                    total + _len + (separator_len if len(current_doc) > 0 else 0)
                    > self._chunk_size
            ):
                if total > self._chunk_size:
                    warnings.warn(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                            total + _len + (separator_len if len(current_doc) > 0 else 0)
                            > self._chunk_size
                            and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def transform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))


class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.

    Adapted from Langchain:
    https://github.com/langchain-ai/langchain/blob/0606aabfa39acb2ec575ea8bbfa4c8e662a6134f/libs/text-splitters/langchain_text_splitters/character.py#L58
    """

    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200, length_function: Callable[[str], int] = len,
                 add_start_index: bool = False, strip_whitespace: bool = True, separators: Optional[List[str]] = None,
                 keep_separator: Union[bool, Literal["start", "end"]] = True, is_separator_regex: bool = False,
                 **kwargs: Any) -> None:
        """Create a new TextSplitter."""
        super().__init__(chunk_size, chunk_overlap, length_function, keep_separator, add_start_index, strip_whitespace)
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )

        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)


def _split_text_with_regex(
        text: str, separator: str, keep_separator: Union[bool, Literal["start", "end"]]
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = (
                ([_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)])
                if keep_separator == "end"
                else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
            )
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = (
                (splits + [_splits[-1]])
                if keep_separator == "end"
                else ([_splits[0]] + splits)
            )
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


def calculate_cosine_distances(sentence_embeddings) -> np.array:
    """Calculate cosine distances between sentences.

    Args:
        sentence_embeddings: List of sentence embeddings to calculate distances for.

    Returns:
        Distance between each pair of adjacent sentences
    """
    # Sliding window array over each pair of adjacent sentence embeddings
    sliding_windows = np.lib.stride_tricks.sliding_window_view(sentence_embeddings, 2, axis=0)

    dot_prod = np.prod(sliding_windows, axis=-1).sum(axis=1)

    magnitude_prod = np.prod(np.linalg.norm(sliding_windows, axis=1), axis=1)

    cos_sim = dot_prod / magnitude_prod
    return 1 - cos_sim


BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile"]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
}


class BoundedSemanticChunker(TextSplitter):
    """First splits the text using semantic chunking according to the specified
    'breakpoint_threshold_amount', but then uses a RecursiveCharacterTextSplitter
    to split all chunks that are larger than 'max_chunk_size'.

    Adapted from langchain_experimental.text_splitter.SemanticChunker"""

    def __init__(self, embedding_model: SentenceTransformer, buffer_size: int = 1, add_start_index: bool = False,
                 breakpoint_threshold_type: BreakpointThresholdType = "percentile",
                 breakpoint_threshold_amount: Optional[float] = None, number_of_chunks: Optional[int] = None,
                 max_chunk_size: int = 500, min_chunk_size: int = 4):
        super().__init__(add_start_index=add_start_index)
        self._add_start_index = add_start_index
        self.embedding_model = embedding_model
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.number_of_chunks = number_of_chunks
        if breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[
                breakpoint_threshold_type
            ]
        else:
            self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        # Splitting the text on '.', '?', and '!'
        self.sentence_split_regex = re.compile(r"(?<=[.?!])\s+")

        assert self.breakpoint_threshold_type == "percentile", "only breakpoint_threshold_type 'percentile' is currently supported"
        assert self.buffer_size == 1, "combining sentences is not supported yet"

    def _calculate_sentence_distances(
            self, sentences: List[dict]
    ) -> Tuple[List[float], List[dict]]:
        """Split text into multiple components."""
        sentences = list(map(lambda x: x.replace("\n", " "), sentences))
        embeddings = self.embedding_model.encode(sentences)
        return calculate_cosine_distances(embeddings.tolist())

    def _calculate_breakpoint_threshold(self, distances: np.array, alt_breakpoint_threshold_amount=None) -> float:
        if alt_breakpoint_threshold_amount is None:
            breakpoint_threshold_amount = self.breakpoint_threshold_amount
        else:
            breakpoint_threshold_amount = alt_breakpoint_threshold_amount
        if self.breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, breakpoint_threshold_amount),
            )
        elif self.breakpoint_threshold_type == "standard_deviation":
            return cast(
                float,
                np.mean(distances)
                + breakpoint_threshold_amount * np.std(distances),
            )
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1

            return np.mean(distances) + breakpoint_threshold_amount * iqr
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )

    def _threshold_from_clusters(self, distances: List[float]) -> float:
        """
        Calculate the threshold based on the number of chunks.
        Inverse of percentile method.
        """
        if self.number_of_chunks is None:
            raise ValueError(
                "This should never be called if `number_of_chunks` is None."
            )
        x1, y1 = len(distances), 0.0
        x2, y2 = 1.0, 100.0

        x = max(min(self.number_of_chunks, x1), x2)

        # Linear interpolation formula
        y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)
        y = min(max(y, 0), 100)

        return cast(float, np.percentile(distances, y))

    def split_text(
            self,
            text: str,
    ) -> List[str]:
        sentences = self.sentence_split_regex.split(text)

        # having len(sentences) == 1 would cause the following
        # np.percentile to fail.
        if len(sentences) == 1:
            return sentences

        bad_sentences = []

        distances = self._calculate_sentence_distances(sentences)

        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
        else:
            breakpoint_distance_threshold = self._calculate_breakpoint_threshold(
                distances
            )

        indices_above_thresh = [
            i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
        ]

        chunks = []
        start_index = 0

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index: end_index + 1]
            combined_text = " ".join(group)
            if self.min_chunk_size <= len(combined_text) <= self.max_chunk_size:
                chunks.append(combined_text)
            else:
                sent_lengths = np.array([len(sd) for sd in group])
                good_indices = np.flatnonzero(np.cumsum(sent_lengths) <= self.max_chunk_size)
                smaller_group = [group[i] for i in good_indices]
                if smaller_group:
                    combined_text = " ".join(smaller_group)
                    if len(combined_text) >= self.min_chunk_size:
                        chunks.append(combined_text)
                        group = group[good_indices[-1]:]
                bad_sentences.extend(group)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            group = sentences[start_index:]
            combined_text = " ".join(group)
            if self.min_chunk_size <= len(combined_text) <= self.max_chunk_size:
                chunks.append(combined_text)
            else:
                sent_lengths = np.array([len(sd) for sd in group])
                good_indices = np.flatnonzero(np.cumsum(sent_lengths) <= self.max_chunk_size)
                smaller_group = [group[i] for i in good_indices]
                if smaller_group:
                    combined_text = " ".join(smaller_group)
                    if len(combined_text) >= self.min_chunk_size:
                        chunks.append(combined_text)
                        group = group[good_indices[-1]:]
                bad_sentences.extend(group)

        # If pure semantic chunking wasn't able to split all text,
        # split the remaining problematic text using a recursive character splitter instead
        if len(bad_sentences) > 0:
            recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=self.max_chunk_size, chunk_overlap=10,
                                                                separators=["\n\n", "\n", ".", ", ", " ", ""])
            for bad_sentence in bad_sentences:
                if len(bad_sentence) >= self.min_chunk_size:
                    chunks.extend(recursive_splitter.split_text(bad_sentence))
        return chunks


class DenseRetriever:

    def __init__(self, embedding_model: SentenceTransformer, num_results: int = 5, similarity_threshold: float = 0.5):
        self.embedding_model = embedding_model
        self.num_results = num_results
        self.similarity_threshold = similarity_threshold
        self.knn = NearestNeighbors(n_neighbors=num_results)
        self.documents = None
        self.document_embeddings = None

    def add_documents(self, documents: List[Document]):
        self.documents = documents
        self.document_embeddings = self.embedding_model.encode([doc.page_content for doc in documents])
        self.knn.fit(self.document_embeddings)

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self.embedding_model.encode(query)

        _, neighbor_indices = self.knn.kneighbors(query_embedding.reshape(1, -1))
        neighbor_indices = neighbor_indices.squeeze(0)

        relevant_doc_embeddings = self.document_embeddings[neighbor_indices]
        # Filter out redundant documents
        included_idxs = filter_similar_embeddings(relevant_doc_embeddings, cosine_similarity,
                                                  threshold=0.95)
        relevant_doc_embeddings = relevant_doc_embeddings[included_idxs]

        # Filter out documents that aren't similar enough
        similarity = cosine_similarity([query_embedding], relevant_doc_embeddings)[0]
        similar_enough = np.where(similarity > self.similarity_threshold)[0]
        included_idxs = [included_idxs[i] for i in similar_enough]

        filtered_result_indices = neighbor_indices[included_idxs]
        return [self.documents[i] for i in filtered_result_indices]


def filter_similar_embeddings(
        embedded_documents: List[List[float]], similarity_fn: Callable, threshold: float
) -> List[int]:
    """Filter redundant documents based on the similarity of their embeddings."""
    similarity = np.tril(similarity_fn(embedded_documents, embedded_documents), k=-1)
    redundant = np.where(similarity > threshold)
    redundant_stacked = np.column_stack(redundant)
    redundant_sorted = np.argsort(similarity[redundant])[::-1]
    included_idxs = set(range(len(embedded_documents)))
    for first_idx, second_idx in redundant_stacked[redundant_sorted]:
        if first_idx in included_idxs and second_idx in included_idxs:
            # Default to dropping the second document of any highly similar pair.
            included_idxs.remove(second_idx)
    return list(sorted(included_idxs))


class SimilarLengthsBatchifyer:
    """
    Generator class to split samples into batches. Groups sample sequences
    of equal/similar length together to minimize the need for padding within a batch.
    """

    def __init__(self, batch_size, inputs, max_padding_len=10):
        # Remember number of samples
        self.num_samples = len(inputs)

        self.unique_lengths = set()
        self.length_to_sample_indices = {}

        for i in range(0, len(inputs)):
            len_input = len(inputs[i])

            self.unique_lengths.add(len_input)

            # For each length, keep track of the indices of the samples that have this length
            # E.g.: self.length_to_sample_indices = { 3: [3,5,11], 4: [1,2], ...}
            if len_input in self.length_to_sample_indices:
                self.length_to_sample_indices[len_input].append(i)
            else:
                self.length_to_sample_indices[len_input] = [i]

        # Use a dynamic batch size to speed up inference at a constant VRAM usage
        self.unique_lengths = sorted(list(self.unique_lengths))
        max_chars_per_batch = self.unique_lengths[-1] * batch_size
        self.length_to_batch_size = {length: int(max_chars_per_batch / (length * batch_size)) * batch_size for length in
                                     self.unique_lengths}

        # Merge samples of similar lengths in those cases where the amount of samples
        # of a particular length is < dynamic batch size
        accum_len_diff = 0
        for i in range(1, len(self.unique_lengths)):
            if accum_len_diff >= max_padding_len:
                accum_len_diff = 0
                continue
            curr_len = self.unique_lengths[i]
            prev_len = self.unique_lengths[i - 1]
            len_diff = curr_len - prev_len
            if (len_diff <= max_padding_len and
                    (len(self.length_to_sample_indices[curr_len]) < self.length_to_batch_size[curr_len]
                     or len(self.length_to_sample_indices[prev_len]) < self.length_to_batch_size[prev_len])):
                self.length_to_sample_indices[curr_len].extend(self.length_to_sample_indices[prev_len])
                self.length_to_sample_indices[prev_len] = []
                accum_len_diff += len_diff
            else:
                accum_len_diff = 0

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        # Iterate over all possible sentence lengths
        for length in self.unique_lengths:

            # Get indices of all samples for the current length
            # for example, all indices of samples with a length of 7
            sequence_indices = self.length_to_sample_indices[length]
            if len(sequence_indices) == 0:
                continue

            dyn_batch_size = self.length_to_batch_size[length]

            # Compute the number of batches
            num_batches = np.ceil(len(sequence_indices) / dyn_batch_size)

            # Loop over all possible batches
            for batch_indices in np.array_split(sequence_indices, num_batches):
                yield batch_indices


def neg_dot_dist(x, y):
    dist = np.dot(x, y).data
    if dist.size == 0:  # no overlapping non-zero entries between x and y
        return np.inf
    return -dist.sum()


class SpladeRetriever:
    def __init__(self, splade_doc_tokenizer, splade_doc_model, splade_query_tokenizer, splade_query_model,
                 device, batch_size, k):
        self.splade_doc_tokenizer = splade_doc_tokenizer
        self.splade_doc_model = splade_doc_model
        self.splade_query_tokenizer = splade_query_tokenizer
        self.splade_query_model = splade_query_model
        self.device = device
        self.batch_size = batch_size
        self.k = k
        self.vocab_size = splade_doc_model.config.vocab_size
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []
        self.sparse_doc_vecs: List[csr_array] = []

    def compute_document_vectors(self, texts: List[str], batch_size: int) -> Tuple[List[List[int]], List[List[float]]]:
        indices = []
        values = []
        batchifyer = SimilarLengthsBatchifyer(batch_size, texts)
        texts = np.array(texts)
        batch_indices = []
        for index_batch in batchifyer:
            batch_indices.append(index_batch)
            with torch.no_grad():
                tokens = self.splade_doc_tokenizer(texts[index_batch].tolist(), truncation=True, padding=True,
                                                   return_tensors="pt").to(self.device)
                output = self.splade_doc_model(**tokens)
            logits, attention_mask = output.logits, tokens.attention_mask
            relu_log = torch.log(1 + torch.relu(logits))
            weighted_log = relu_log * attention_mask.unsqueeze(-1)
            tvecs, _ = torch.max(weighted_log, dim=1)

            # extract all non-zero values and their indices from the sparse vectors
            for batch in tvecs.cpu():
                indices.append(batch.nonzero(as_tuple=True)[0].numpy())
                values.append(batch[indices[-1]].numpy())

        # Restore order after SimilarLengthsBatchifyer disrupted it:
        # Ensure that the order of 'indices' and 'values' matches the order of the 'texts' parameter
        batch_indices = np.concatenate(batch_indices)
        sorted_indices = np.argsort(batch_indices)
        indices = [indices[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        return indices, values

    def compute_query_vector(self, text: str):
        """
        Computes a vector from logits and attention mask using ReLU, log, and max operations.
        """
        with torch.no_grad():
            tokens = self.splade_query_tokenizer(text, return_tensors="pt").to(self.device)
            output = self.splade_query_model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        query_vec = max_val.squeeze().cpu()

        query_indices = query_vec.nonzero().numpy().flatten()
        query_values = query_vec.detach().numpy()[query_indices]

        return query_indices, query_values

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents (List[Document]: Documents to add to the vectorstore.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas)

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None):

        # Remove duplicate and empty texts
        text_to_metadata = {texts[i]: metadatas[i] for i in range(len(texts)) if len(texts[i]) > 0}
        texts = list(text_to_metadata.keys())
        metadatas = list(text_to_metadata.values())
        self.texts = texts
        self.metadatas = metadatas

        indices, values = self.compute_document_vectors(texts, self.batch_size)
        self.sparse_doc_vecs = [csr_array((val, (ind,)),
                                          shape=(self.vocab_size,)) for val, ind in zip(values, indices)]

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_indices, query_values = self.compute_query_vector(query)

        sparse_query_vec = csr_array((query_values, (query_indices,)), shape=(self.vocab_size,))
        dists = [neg_dot_dist(sparse_query_vec, doc_vec) for doc_vec in self.sparse_doc_vecs]
        sorted_indices = np.argsort(dists)

        return [Document(self.texts[i], self.metadatas[i]) for i in sorted_indices[:self.k]]


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()


class BM25Retriever:
    """ Adapted from Langchain:
    https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/bm25.py"""
    vectorizer: Any
    """ BM25 vectorizer."""
    docs: List[Document]
    """ List of documents."""
    k: int = 4
    """ Number of documents to return."""
    preprocess_func: Callable[[str], List[str]] = default_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    def __init__(self, vectorizer: Any, docs: List[Document], k: int = 4,
                 preprocess_func: Callable[[str], List[str]] = default_preprocessing_func):
        self.vectorizer = vectorizer
        self.docs = docs
        self.k = k
        self.preprocess_func = preprocess_func

    @classmethod
    def from_texts(
            cls,
            texts: Iterable[str],
            metadatas: Optional[Iterable[dict]] = None,
            bm25_params: Optional[Dict[str, Any]] = None,
            preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
            **kwargs: Any,
    ) -> "BM25Retriever":
        """
        Create a BM25Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
        )

    @classmethod
    def from_documents(
            cls,
            documents: Iterable[Document],
            *,
            bm25_params: Optional[Dict[str, Any]] = None,
            preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
            **kwargs: Any,
    ) -> "BM25Retriever":
        """
        Create a BM25Retriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            preprocess_func=preprocess_func,
            **kwargs,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        processed_query = self.preprocess_func(query)
        return_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
        return return_docs


async def async_download_html(url: str, headers: Dict, timeout: int):
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(timeout),
                                     max_field_size=65536) as session:
        try:
            resp = await session.get(url)
            return await resp.text(), url
        except UnicodeDecodeError:
            if not resp.headers['Content-Type'].startswith("text/html"):
                print(
                    f"LLM_Web_search | {url} generated an exception: Expected content type text/html. Got {resp.headers['Content-Type']}.")
        except TimeoutError:
            print('LLM_Web_search | %r did not load in time' % url)
        except Exception as exc:
            print('LLM_Web_search | %r generated an exception: %s' % (url, exc))
    return None


async def async_fetch_chunk_websites(urls: List[str],
                                     text_splitter: BoundedSemanticChunker or RecursiveCharacterTextSplitter,
                                     timeout: int = 10):
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
               "Accept-Language": "en-US,en;q=0.5",
               "Accept-Encoding": "gzip;q=1, *;q=0.5"}
    result_futures = [async_download_html(url, headers, timeout) for url in urls]
    chunks = []
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        for f in asyncio.as_completed(result_futures):
            result = await f
            if result:
                resp_html, url = result
                document = html_to_plaintext_doc(resp_html, url)
                new_chunks = await loop.run_in_executor(pool, text_splitter.split_documents, [document])
                chunks.extend(new_chunks)
    return chunks


def docs_to_pretty_str(docs) -> str:
    ret_str = ""
    for i, doc in enumerate(docs):
        ret_str += f"Result {i + 1}:\n"
        ret_str += f"{doc.page_content}\n"
        ret_str += f"Source URL: {doc.metadata['source']}\n"
    return ret_str


def html_to_plaintext_doc(html_text: str or bytes, url: str) -> Document:
    with warnings.catch_warnings(action="ignore"):
        soup = BeautifulSoup(html_text, features="lxml")
    for script in soup(["script", "style"]):
        script.extract()

    strings = '\n'.join([s.strip() for s in soup.stripped_strings])
    webpage_document = Document(page_content=strings, metadata={"source": url})
    return webpage_document


def weighted_reciprocal_rank(doc_lists: List[List[Document]], weights: List[float], c: int = 60) -> List[Document]:
    """
    Perform weighted Reciprocal Rank Fusion on multiple rank lists.
    You can find more details about RRF here:
    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

    Args:
        doc_lists: A list of rank lists, where each rank list contains unique items.
        weights: A list of weights corresponding to the rank lists. Defaults to equal
            weighting for all lists.
        c: A constant added to the rank, controlling the balance between the importance
            of high-ranked items and the consideration given to lower-ranked items.
            Default is 60.

    Returns:
        list: The final aggregated list of items sorted by their weighted RRF
                scores in descending order.
    """
    if len(doc_lists) != len(weights):
        raise ValueError(
            "Number of rank lists must be equal to the number of weights."
        )

    # Associate each doc's content with its RRF score for later sorting by it
    # Duplicated contents across retrievers are collapsed & scored cumulatively
    rrf_score: Dict[str, float] = defaultdict(float)
    for doc_list, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(doc_list, start=1):
            rrf_score[doc.page_content] += weight / (rank + c)

    # Docs are deduplicated by their contents then sorted by their scores
    all_docs = chain.from_iterable(doc_lists)
    sorted_docs = sorted(
        unique_by_key(all_docs, lambda doc: doc.page_content),
        reverse=True,
        key=lambda doc: rrf_score[doc.page_content],
    )
    return sorted_docs


def unique_by_key(iterable: Iterable, key: Callable) -> Iterator:
    """Yield unique elements of an iterable based on a key function.

    Args:
        iterable: The iterable to filter.
        key: A function that returns a hashable key for each element.

    Yields:
        Unique elements of the iterable based on the key function.
    """
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e
