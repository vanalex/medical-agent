# Medical Agent

A LangGraph-based decision agent that intelligently routes medical queries to either PubMed (for research-based questions) or Tavily (for general medical information).

## Features

- **Automatic Query Classification**: Classifies queries as 'research' or 'general' using LLM
- **Dual Search Sources**:
  - PubMed for scholarly articles, clinical trials, and research
  - Tavily for treatment options, symptoms, and patient-friendly information
- **Quality Checking**: Validates search results and refines queries if needed
- **Memory Checkpointing**: Maintains conversation state using LangGraph MemorySaver
- **Professional Summarization**: Provides citations (PMID/URLs) and medical disclaimers

## Installation

```bash
uv sync
```

## Environment Variables

Set your API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"
```

## Usage

```bash
python medical_agent.py
```

Enter your medical question when prompted, and the agent will search and summarize relevant information.

## Example Queries

- **Research**: "What are the latest clinical trials for CRISPR gene therapy?"
- **General**: "What are the common symptoms of type 2 diabetes?"

## Disclaimer

This tool provides informational content only and is not a substitute for professional medical advice.
