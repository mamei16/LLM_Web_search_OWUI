# LLM_Web_search_OWUI

This repo contains the source code of the LLM Web Search tool for Open WebUI. The code is [LLM_Web_search](https://github.com/mamei16/LLM_Web_search) ported to Open WebUI, retaining as much functionality as possible given the different environments. This means that the documentation on search types, keyword retrievers and chunking methods from the original project still apply to this  project.



* **[Table of Contents](#table-of-contents)**
  * [Initial Setup](#initial-setup)
  * [Settings (Valves)](#settings-valves)
    + [Ensemble Weighting](#ensemble-weighting)

## Initial Setup

### Configure Embedding Model Save Location

This tool processes search results locally on device. For this purpose, three embedding models need to be downloaded automatically when the first web search is executed. Before using this tool for the first time, you'll need to configure the "Embedding Model Save Path" setting (or "Valve"), which controls where the embedding models will be downloaded to.


### Update DuckDuckGo Search Python Package

By default, DuckDuckGo is used as the search engine, and the `duckduckgo-search` python package is used to get results from DuckDuckGo. However, as of this writing, the `open-webui` PyPI package ships with an old version of `duckduckgo-search` (see [here](https://github.com/open-webui/open-webui/blob/main/pyproject.toml#L92)). It is important to keep this package up to date to avoid `202 Ratelimit` exceptions, so before using this tool (and anytime you encounter said exception), you'll need to update `duckduckgo-search`:  
`pip install --upgrade duckduckgo-search`  

If you choose to use SearXNG as the search engine, you of course won't need to update `duckduckgo-search`.


## Settings (Valves)

Most settings are already explained briefly in the UI, and are expanded on in the readme of the [original project](https://github.com/mamei16/LLM_Web_search?tab=readme-ov-file#search-types). In the following subsection(s), some extra info that would be too verbose to put into the UI will be added.

### Ensemble Weighting 

Set this value to 0 to use keyword retrieval only, or 1 to use purely dense retrieval. If you choose to run the tool on CPU only, I recommend to either use BM25 as the keyword retrieval method or use purely dense retrieval, as the SPLADE document encoder will run slowly on a CPU.