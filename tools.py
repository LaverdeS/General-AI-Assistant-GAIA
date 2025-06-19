import base64
import json
import inspect
import time
from typing import Callable

import diskcache

from datetime import datetime, timezone

from langchain.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from markitdown import MarkItDown
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

from basic_agent import print_conversation

from dotenv import load_dotenv
from langchain.globals import set_debug
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


set_debug(False)
CUSTOM_DEBUG = True

load_dotenv()

cache = diskcache.Cache(".tools_cache")


def encode_image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def print_tool_call(tool: Callable, tool_name: str, args: dict):
    """Prints the tool call for debugging purposes."""
    sig = inspect.signature(tool)
    print_conversation(
        messages=[
            {
                'role': 'Tool-Call',
                'content': f"Calling `{tool_name}`{sig}"
            },
            {
                'role': 'Tool-Args',
                'content': args
            }
        ],
    )


def print_tool_response(response: str):
    """Prints the tool response for debugging purposes."""
    print_conversation(
        messages=[
            {
                'role': 'Tool-Response',
                'content': response
            }
        ],
    )


tavily_tool_community = TavilySearchResults(max_results=5,
                                  search_depth='advanced',
                                  include_answer=False,
                                  include_raw_content=True)
md = MarkItDown()  # Set to True to enable tools debug prints


@tool
def search_web_extract_info(query: str) -> list:
    """Search the web for a query and extract useful information from the search links"""
    results = tavily_tool_community.invoke(query)
    docs = []
    MAX_NUMBER_OF_CHARS = 10_000
    sig = inspect.signature(search_web_extract_info)
    if CUSTOM_DEBUG:
        print_conversation(
            messages=[
                {
                    'role': 'Tool-Call',
                    'content': f"Calling `search_web_extract_info`{sig}"
                },
                {
                    'role': 'Tool-Args',
                    'content': {'query': query, 'max_number_of_chars': MAX_NUMBER_OF_CHARS}
                }
            ],
        )
    for result in results:
            try:
                extracted_info = md.convert(result['url'])
                text_title = extracted_info.title.strip()
                text_content = extracted_info.text_content.strip()

                doc = result.copy() if isinstance(result, dict) else {}

                doc['title'] = text_title
                if 'content' in doc:
                    doc.pop('content')
                doc['content'] = text_content[:MAX_NUMBER_OF_CHARS] if len(text_content) > MAX_NUMBER_OF_CHARS else text_content

                if 'raw_content' in doc and isinstance(doc['raw_content'], str):
                    doc['raw_content'] = doc['raw_content'][:MAX_NUMBER_OF_CHARS] if len(doc['raw_content']) > MAX_NUMBER_OF_CHARS else doc['raw_content']

                doc['retrieved_at'] = datetime.now(timezone.utc).isoformat()

                docs.append(doc)

            except Exception:
                continue


    if CUSTOM_DEBUG:
        console_structured_results = [
            {k: v for k, v in result_dicti.items() if k != "raw_content"} for result_dicti in docs
        ]
        print_conversation(
            messages=[
                {
                    'role': 'Tool-Response',
                    'content': json.dumps(console_structured_results)
                }
            ],
        )
    return docs


search_tool = TavilySearch(max_results=5)
extract_tool = TavilyExtract()


@tool
def search_and_extract(query: str) -> list[dict]:
    """Performs a web search and returns structured content extracted from top results."""
    time.sleep(3)  # To avoid hitting the API rate limit in the llm-apis when calling the tool multiple times in a row.
    if query in cache:
        print(f"Cache hit for query: {query}")
        return cache[query]
    MAX_NUMBER_OF_CHARS = 10_000

    if CUSTOM_DEBUG:
        print_tool_call(
            search_and_extract,
            tool_name='search_and_extract',
            args={'query': query, 'max_number_of_chars': MAX_NUMBER_OF_CHARS},
        )

    results = search_tool.invoke({"query": query})
    raw_results = results.get("results", [])
    urls = [r["url"] for r in raw_results if r.get("url")]

    if not urls:
        return [{"error": "No URLs found to extract from."}]

    extracted = extract_tool.invoke({"urls": urls})
    results = extracted.get("results", [])

    structured_results = []
    raw_contents = [doc.get("raw_content", "") for doc in results]

    for result, doc_content in zip(raw_results, raw_contents):
        doc_content_trunc = doc_content[0:MAX_NUMBER_OF_CHARS] if len(doc_content) > MAX_NUMBER_OF_CHARS else doc_content
        structured_results.append({
            "title": result.get("title"),
            "url": result.get("url"),
            "snippet": result.get("content"),
            "raw_content": doc_content_trunc
        })

    if CUSTOM_DEBUG:
        console_structured_results = [{k: v for k, v in result_dicti.items() if k != "raw_content"} for result_dicti in
                                      structured_results]
        print_tool_response(json.dumps(console_structured_results))
    return structured_results



def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    return parse_qs(parsed.query).get("v", [""])[0]


@tool
def load_youtube_transcript(url: str) -> str:
    """Load a YouTube transcript using youtube_transcript_api."""

    video_id = extract_video_id(url)

    if CUSTOM_DEBUG:
        print_tool_call(
            load_youtube_transcript,
            tool_name='load_youtube_transcript',
            args={'url': url},
        )
    try:
        youtube_api_client = YouTubeTranscriptApi()
        fetched_transcript = youtube_api_client.fetch(video_id=video_id)
        transcript = " ".join(entry.text for entry in fetched_transcript if entry.text.strip())

        if transcript and CUSTOM_DEBUG:
            print_tool_response(transcript)

        return transcript

    except Exception as e:
        error_str = f"Error loading transcript: {e}. Assuming no transcript for this video."
        print_tool_response(error_str)
        return error_str



gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

@tool
def image_query_tool(image_path: str, question: str) -> str:
    """
    Uses Gemini Vision to answer a question about an image.
    - image_path: file path to the image to analyze (.png)
    - question: the query to ask about the image
    """
    try:
        base64_img = encode_image_to_base64(image_path)
    except OSError:
        response = f"OSError: Invalid argument (invalid image path or file format): {image_path}. Please provide a valid PNG image."
        print_tool_response(response)
        return response

    base64_img_str = f"data:image/png;base64,{base64_img}"
    if CUSTOM_DEBUG:
        print_tool_call(
            image_query_tool,
            tool_name='image_query_tool',
            args={'base64_image': base64_img_str[:100], 'question': question},
        )
    msg = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": base64_img_str},
    ])
    try:
        response = gemini.invoke([msg])
    except ChatGoogleGenerativeAIError:
        response = "ChatGoogleGenerativeAIError: Invalid argument provided to Gemini: 400 Provided image is not valid"
        print_tool_response(response)
        return response
    if CUSTOM_DEBUG:
        print_tool_response(response.content)
    return response.content


@tool
def search_and_extract_from_wikipedia(query: str) -> list:
    """Search Wikipedia for a query and extract useful information."""
    wiki_api_wrapper = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
    if CUSTOM_DEBUG:
        print_tool_call(
            search_and_extract_from_wikipedia,
            tool_name='search_and_extract_from_wikipedia',
            args={'query': query},
        )
    response = wiki_tool.invoke(query)
    if CUSTOM_DEBUG:
        print_tool_response(response)
    return response



if __name__ == "__main__":
    test = "D"
    if test=="A":
        query = "What are tardigrades?"
        _ = search_and_extract.invoke({"query": query})
        # _ = search_web_extract_info.invoke({"query": query})#

    elif test=="B":
        url = "https://www.youtube.com/watch?v=L1vXCYZAYYM"  # https://www.youtube.com/watch?v=dGby9BH9bMc   , https://www.youtube.com/watch?v=L1vXCYZAYYM
        _ = load_youtube_transcript.invoke({"url": url})

    elif test=="C":
        query = "Tardigrades"
        _ = search_and_extract_from_wikipedia.invoke(query)

    elif test=="D":

        image_path = "./chess_move.png"
        question = "What is the best move for white in this position?"
        c = image_query_tool.invoke({"image_path": image_path, "question": question})
        #print(f"c({type(json.loads(c))})", json.loads(c))


# todo: add wikipedia search tool
# todo: youtube tool search should be visual... use gemini
# summarize tool for tavily search results