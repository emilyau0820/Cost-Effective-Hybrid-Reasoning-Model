# Project: Multi-Model AI Workflow
# File: hybridaiworkflow.py
# Purpose: Cost and latency-optimized hybrid AI workflow that combines caching, compression, and expansion to make high-quality reasoning accessible.
# Dependencies: openai>=1.2.0, Python 3.10+
# Author: Emily Au

from openai import OpenAI

# Your OpenAI API key
client = OpenAI(api_key = "", timeout = 600)

# Text files and markers for caching
queryFile = "queries.txt"
responseFile = "responses.txt"
targetMarker = "`"

# Breaks a text file into blocks based on a specified marker
def readBlocks(filePath, marker = targetMarker):
    blocks = []
    with open(filePath, "r") as lines:
        currentBlock = []
        for line in lines:
            line = line.strip()
            if line == marker:
                if currentBlock:
                    blocks.append("\n".join(currentBlock))
                    currentBlock = []
            else:
                currentBlock.append(line)
        if currentBlock:
            blocks.append("\n".join(currentBlock))
    return blocks

# Cleans and standardizes text for comparison
def normalize(text):
    import re
    return re.sub(r'\s+', ' ', text.replace('\r','')).strip().lower()

# STAGE 1: CACHING
# Read and parse the query and response files into blocks
queryLines = readBlocks(queryFile, targetMarker)
responseLines = readBlocks(responseFile, targetMarker)

# Creates a mapping (cache) of queries to corresponding responses
mapping = dict(zip(queryLines, responseLines))

# Gather user input and return the cached response if found
print("Please enter your queries: ")
userInput = input()
checkInput = userInput.strip().lower()
found = False
for query, response in mapping.items():
    if checkInput == normalize(query):
        # Cached response found: return directly, bypassing pipeline
        print(response)
        found = True
        break

if not found:
    # STAGE 2: COMPRESSION (gpt-4o-mini)
    # Triage/Keyword extraction
    prompt1 = "If the prompt DOES NOT require deep reasoning and research, output \"0\" (eg. \"0Here is an example response\") followed by your response to the prompt. Otherwise, adhere to the following instructions. From the following text extract all key terms, named entities, and relations that should be represented in a semantic embedding space. Output only the precise tokens that maximize vector representation quality as a comma‑separated list. Do not write a response blurb. Use the following prompt: " + userInput
    response1 = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt1}])
    compressedQueries1 = response1.choices[0].message.content
    if compressedQueries1.startswith("0"):
        print(compressedQueries1.lstrip("0").strip())
    else:
        # Semantic filtering
        prompt2 = "Select only the most semantically central 50'%' of keywords, entities, or relations that best describe the input’s overall meaning. Discard secondary, repetitive, or weakly connected tokens. Output minimal, high-importance items only as a comma‑separated list. Do not write a response blurb. Use the following text: " + compressedQueries1
        response2 = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt2}])
        compressedQueries2 = response2.choices[0].message.content

        # STAGE 3: RESEARCH (o4-mini-deep-research)
        prompt3 = "From the input text, list up to seven distinct key points capturing the most critical findings, ideas, or conclusions. Output only a numbered list of points: " + compressedQueries2
        response3 = client.responses.create(
            model="o4-mini-deep-research",
            input=prompt3,
            background=False,
            tools=[
                {"type": "web_search_preview"},
            ]
        )
        keyPoints = response3.output_text

        # STAGE 4: EXPANSION (gpt-4o-mini)
        prompt4 = "You are a professional in the field of the following list of topics. Take each key finding and expand and elaborate on them (eg. related insights, advancements, implications, examples, statistics, and supporting evidence), and teach them as if you were a professional. Do not write a response blurb and use these key points: " + keyPoints
        response4 = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt4}])
        expandedResponse = response4.choices[0].message.content
        print(expandedResponse)

        # Caches prompt and expanded response to respective files
        with open(queryFile, "a") as appendQueries, open(responseFile, "a") as appendResponses:
            appendQueries.write("\n" + userInput + "\n" + targetMarker)
            appendResponses.write("\n" + expandedResponse + "\n" + targetMarker)