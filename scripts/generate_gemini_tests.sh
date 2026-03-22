#!/bin/bash
codex exec "I have implemented Gemini API support in Tinkuy. Please research the implementation in src/tinkuy/formats/gemini.py and the updates in src/tinkuy/gateway/_gateway.py. Then, write comprehensive unit tests in tests/test_gemini_format.py using pytest. The tests should cover:
1. GeminiLiveAdapter: Synthesizing GenerateContentRequest from various projection states (R0-R4).
2. GeminiInboundAdapter: Correctly parsing Gemini requests into Tinkuy events.
3. GeminiResponseIngester: Correctly ingesting Gemini responses and extracting signals.
4. Gateway: Ensure prepare_gemini_request and ingest_gemini_response work as expected and correctly set the APIFormat.
Follow the existing test style in tests/."
