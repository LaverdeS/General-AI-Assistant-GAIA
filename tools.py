import base64
import json
import os
import inspect
import requests
import time
from pathlib import PurePath

import diskcache

from datetime import datetime, timezone

import trafilatura
from langchain.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from markitdown import MarkItDown
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_google_community import SpeechToTextLoader
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.file_management.read import ReadFileTool
from youtube_transcript_api import YouTubeTranscriptApi
from typing import Callable, Union

from basic_agent import print_conversation

from dotenv import load_dotenv
from langchain.globals import set_debug
from urllib.parse import urlparse, parse_qs

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


set_debug(False)
CUSTOM_DEBUG = True

load_dotenv()

cache = diskcache.Cache(".tools_cache")


@tool
def aggregate_information(query: str, results: list[str]) -> str:
    """
    Processes a list of unstructured text chunks (e.g., search results) and produces a concise, query-specific summary.

    Each input text is filtered and summarized individually in the context of the provided query. Irrelevant results are discarded.
    Relevant content is aggregated and synthesized into a final, coherent answer that directly addresses the query.
    """
    if CUSTOM_DEBUG:
        print_tool_call(
            aggregate_information,
            tool_name='aggregate_information',
            args={'results': results, 'query': query},
        )
    if not results:
        response = "No search results provided."
        if CUSTOM_DEBUG:
            print_tool_response(response)
        return response

    # Convert to LangChain Document objects
    docs = [Document(page_content=chunk) for chunk in results]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 🔍 Map Prompt — Summarize each document in light of the query
    map_prompt = PromptTemplate.from_template(
        "You are analyzing a search result in the context of the question: '{query}'.\n\n"
        "Search result:\n{text}\n\n"
        "Instructions:\n"
        "- If the result contains information relevant to answering the query, summarize the relevant parts clearly.\n"
        "- If the result is not helpful or unrelated, return 'IGNORE'.\n"
        "- Do not include generic information or filler.\n"
        "- Focus on extracting facts, key statements, or numbers that directly support the query.\n\n"
        "Relevant Summary:"
    )

    # 🧠 Combine Prompt — Aggregate the summaries to one final answer
    combine_prompt = PromptTemplate.from_template(
        "You are aggregating information to provide context to answer the following question: '{query}'.\n\n"
        "Here are the summaries from filtered search results:\n{text}\n\n"
        "Use the provided summaries to construct a context that directly supports the query without answering it.\n"
        "Context:"
    )

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt.partial(query=query),
        combine_prompt=combine_prompt.partial(query=query),
    )

    summary = chain.invoke({'input_documents': docs})
    output_text = summary.get('output_text', str(summary))
    output_text = json.dumps({'summary': output_text})

    if CUSTOM_DEBUG:
        print_tool_response(output_text)

    return output_text


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


def print_tool_response(response: Union[str, list]):
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


@tool # deprecated
def search_web_extract_info(query: str) -> list:  # deprecated
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
    return docs   # Un


search_tool = TavilySearch(max_results=3)
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
def get_audio_from_youtube(urls: list[str], save_dir:str="./tmp/") -> list[str | PurePath | None]:
    """Extracts audio from a YouTube video URL."""

    if CUSTOM_DEBUG:
        print_tool_call(
            get_audio_from_youtube,
            tool_name='get_audio_from_youtube',
            args={'urls': urls, 'save_dir': save_dir},
        )
    loader = YoutubeAudioLoader(urls, save_dir)
    audio_blobs = list(loader.yield_blobs())
    paths = [str(blob.path) for blob in audio_blobs]

    if CUSTOM_DEBUG:
        print_tool_response(json.dumps({'paths': paths}))

    return paths


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


youtube_search_api = YouTubeSearchTool()

@tool
def youtube_search_tool(query: str, number_of_results:int=3) -> list:
    """Search YouTube for a query and return the top number_of_results."""
    if CUSTOM_DEBUG:
        print_tool_call(
            youtube_search_tool,
            tool_name='youtube_search_tool',
            args={'query': query, number_of_results: number_of_results},
        )
    response = youtube_search_api.run(f"{query},{number_of_results}")
    if CUSTOM_DEBUG:
        print_tool_response(response)
    return response


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


@tool
def transcribe_audio(file_path: str) -> list:
    """Transcribe audio from an audio file in file_path using Google Speech-to-Text."""
    docs, docs_content = [], []
    if CUSTOM_DEBUG:
        print_tool_call(
            transcribe_audio,
            tool_name='transcribe_audio',
            args={'file_path': file_path},
        )
    try:
        loader = SpeechToTextLoader(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
            file_path=file_path,
            is_long = False,  # Set to True for long audio files
        )

        docs = loader.load()
    except Exception as e:
        print(f"Error loading audio file: {e}")
        try:
            loader = SpeechToTextLoader(
                project_id=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
                file_path=file_path,
                is_long=True,  # Set to True for long audio files
            )

            docs = loader.load()
        except Exception as e:
            docs_content = [f"Error loading audio file: {e}"]

    docs_content = [doc.page_content for doc in docs] if docs else docs_content

    if CUSTOM_DEBUG:
        print_tool_response(docs_content)
    return docs_content


@tool
def extract_clean_text_from_url(url: str) -> str:
    """Extract the main readable content from a webpage using trafilatura."""
    if CUSTOM_DEBUG:
        print_tool_call(
            extract_clean_text_from_url,
            tool_name='extract_clean_text_from_url',
            args={'url': url},
        )
    downloaded = trafilatura.fetch_url(url)
    response = ""
    if not downloaded:
        response = "Failed to download the page. Please check the URL."

    if not "Failed" in response:
        response = trafilatura.extract(downloaded)

    response = response or "No meaningful content found."
    if CUSTOM_DEBUG:
        print_tool_response(response)
    return response


read_tool = ReadFileTool()


@tool
def smart_read_file(file_path: str) -> str:
    """
    Smart tool to read a file based on its type.

    - Use `read_file_tool` for simple text, CSV, code files.
    - Use MarkItDown for PDFs, Word, Excel, HTML, and other complex formats.
    """
    if CUSTOM_DEBUG:
        print_tool_call(
            smart_read_file,
            tool_name='smart_read_file',
            args={'file_path': file_path},
        )
    _, ext = os.path.splitext(file_path.lower())

    if ext in [".mp3", ".wav", ".m4a", ".flac"]:
        # If the file is an audio file, transcribe it
        return transcribe_audio.invoke({"file_path": file_path})

    if ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
        # If the file is an image, use image_query_tool to analyze it
        q = "What can you tell me about this image?"
        return image_query_tool.invoke({"image_path": file_path, "question": q})

    if any(ext in url_pattern for url_pattern in ["http://", "https://", "www."]):
        if "youtube.com/watch?v=" in file_path:
            transcript = load_youtube_transcript.invoke({"url": file_path})
            if "Error loading" in transcript:
                return get_audio_from_youtube.invoke({'urls': [file_path], 'save_dir': './tmp/'})
        else:
            return extract_clean_text_from_url.invoke(file_path)

    md = MarkItDown()
    try:
        result = md.convert(file_path)
        result = result.text_content
    except Exception as e:
        # print("Error reading file with MarkItDown:", e)
        result = read_tool.invoke({"file_path": file_path})

    if CUSTOM_DEBUG:
        print_tool_response(result)
    return result


@tool
def geocode_with_nominatim(place_name: str):
    """Uses Nominatim to geocode a place name and return its 'latitude, longitude'"""
    if CUSTOM_DEBUG:
        print_tool_call(
            geocode_with_nominatim,
            tool_name='geocode_with_nominatim',
            args={'place_name': place_name},
        )
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    response = requests.get(url, params=params, headers={"User-Agent": "LangGraph-Agent"})
    results = response.json()
    if results:
        location_info = {
            "lat": results[0]["lat"],
            "lon": results[0]["lon"],
            "display_name": results[0]["display_name"]
        }
    else:
        location_info = {"error": "No results found for the given place name."}
    if CUSTOM_DEBUG:
        print_tool_response(json.dumps(location_info))
    return location_info



@tool
def search_nearby_clothing_stores_by_latlon(lat_lon_location: str, radius_meters: int = 1000) -> list[dict]:
    """
    Uses Google Places API to find nearby clothing stores around the given location.
    Returns store names, addresses, and business data. The lat_lon_location must be in the format "latitude,longitude".
    """
    if CUSTOM_DEBUG:
        print_tool_call(
            search_nearby_clothing_stores_by_latlon,
            tool_name='search_nearby_clothing_stores_by_latlon',
            args={'location': lat_lon_location, 'radius_meters': radius_meters},
        )
    endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        "key": os.getenv("GOOGLE_API_KEY"),
        "location": lat_lon_location,
        "radius": radius_meters,
        "type": "clothing_store"
    }

    response = requests.get(endpoint, params=params)
    data = response.json()
    with open("google_places_response.json", "w") as f:
        json.dump(data, f, indent=2)

    area_info = [
        {
            "name": place["name"],
            "address": place.get("vicinity"),
            "business_status": place.get("business_status", "Unknown"),
            "market": place.get("types", []),
            "rating": place.get("rating"),
            "user_ratings_total": place.get("user_ratings_total", 0),
            "opening_hours": place.get("opening_hours", {}),

        }
        for place in data.get("results", [])
    ]
    if CUSTOM_DEBUG:
        print_tool_response(area_info)
    return area_info


if __name__ == "__main__":

    test = "L"

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

    elif test=="E":
        file_path = "./audio_sample.mp3"
        t = transcribe_audio.invoke({"file_path": file_path})
        print(f"Transcription: {t}")

    elif test=="F":
        query = "Tardigrades"
        _ = youtube_search_tool.invoke({"query": query, "number_of_results": 3})

    elif test=="G":
        file_paths = [
            "./username.csv",
            "./basic_agent.py",
            "./audio_sample.mp3",
            "./chess_move.png",
            "file_example.xlsx",
            "sample-local-pdf.pdf",
            "https://www.youtube.com/watch?v=L1vXCYZAYYM",
            "gaia_default_results_log.json",
            "https://en.wikipedia.org/wiki/Tardigrade",
            "README.md",
        ]
        for file_path in file_paths:
            _ = smart_read_file.invoke({"file_path": file_path})


    elif test=="I":
        urls = ["https://www.youtube.com/watch?v=L1vXCYZAYYM", "https://www.youtube.com/watch?v=dGby9BH9bMc"]
        _ = get_audio_from_youtube.invoke({"urls": urls, "save_dir": "./tmp/"})

    elif test=="J":
        # summarize_search_results
        results = [
            "Tardigrades, also known as water bears, are microscopic animals that can survive extreme conditions.",
            "They are found in various environments, including deep oceans and high mountains.",
            "Tardigrades can withstand temperatures from near absolute zero to over 300 degrees Fahrenheit."
        ]
        query = "What are the survival capabilities of tardigrades?"
        _ = aggregate_information.invoke({"query": query, "results": results})

    elif test=="K":
        leipzig_location = "51.3407317,12.371553"  # Latitude, Longitude for Leipzig, Germany
        radius_meters = 1000
        _ = search_nearby_clothing_stores_by_latlon.invoke({"location": leipzig_location, "radius_meters": radius_meters})

    elif test=="L":
        place_name = "Leipzig, Germany"
        _ = geocode_with_nominatim.invoke({"place_name": place_name})


# todo: youtube tool search should be visual... use gemini
# summarize tool for tavily search results