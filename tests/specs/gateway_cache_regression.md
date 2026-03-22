# Regression tests: gateway cache collapse (2026-03-22)

Three bugs interacted to collapse cache hit rate to 0% in the live projective gateway.

## Bug 1: count_tokens route missing

`POST /v1/messages/count_tokens` returned 404. Claude Code uses this endpoint
for the `/context` command.

### Test: `test_count_tokens_route_exists`

- Create the FastAPI app via `create_app()`
- Send `POST /v1/messages/count_tokens` with a valid body
- Assert status is NOT 404
- The route should forward to upstream (mock the httpx client)

## Bug 2: max_tokens=1 probe corrupts projection

When count_tokens 404'd, Claude Code fell back to sending a regular
`POST /v1/messages` with `max_tokens=1, stream=false`. The gateway
processed this as a real turn, ingesting a 1-token response and
corrupting the projection.

### Test: `test_max_tokens_1_probe_skips_projection`

- Create a Gateway instance with some conversation state
- Record the projection state (turn count, total_tokens, block count)
- Send a request through the server's `/v1/messages` handler with
  `{"max_tokens": 1, "stream": false, "model": "...", "messages": [...]}`
- Assert the projection state is UNCHANGED after the request
- Assert the response was forwarded to upstream without calling `prepare_request`

## Bug 3: page table in system block busts cache

The page table (`<yuyay-page-table>`) was injected into the system prompt
array. It changes every turn. Anthropic's API caches by prefix — any change
in the system block invalidates cache for everything after it.

### Test: `test_page_table_not_in_system_block`

- Create a Gateway, process several turns to build up projection state
- Call `gateway._synthesize(APIFormat.ANTHROPIC)` to get the payload
- Assert that NO element in `payload["system"]` contains `<yuyay-page-table>`
- Assert that `<yuyay-page-table>` appears in the messages array (in a user message)

### Test: `test_page_table_in_last_user_message`

- Same setup as above
- Find the last user message in `payload["messages"]`
- Assert its content (as list) contains a block with `<yuyay-page-table>`

## Bug 3b: cache_control stripped during system merge

`_merge_system()` was stripping `cache_control` from gateway system parts,
destroying the R0/R1 cache boundary.

### Test: `test_merge_system_preserves_cache_control`

- Call `_merge_system(client_system, gateway_system)` where gateway_system
  contains parts with `cache_control: {"type": "ephemeral"}`
- Assert the merged result still contains `cache_control` on those parts

### Test: `test_merge_system_deduplicates`

- Call `_merge_system` where client and gateway have overlapping text
- Assert the overlapping text appears only once in the result

## Bug 4: page table before tool_results breaks API pairing

When the last user message contains tool_result blocks (responding to
the prior assistant's tool_use), injecting the page table text block
before them breaks the API's requirement that tool_results come
immediately after their matching tool_use.

### Test: `test_page_table_after_tool_results`

- Create a Gateway, process a turn that produces tool_use blocks
- Feed tool_results back as the next turn
- Get the synthesized payload
- Find the last user message in the payload
- Assert that all `tool_result` content blocks come BEFORE any
  `text` block containing `<yuyay-page-table>`
- Assert the page table IS present (just not before tool_results)

## Validator tests

The pre-flight validator (`src/tinkuy/formats/validate.py`) should have
its own test file: `tests/test_validate.py`.

### Test: `test_validator_catches_alternation_violation`

- Payload with consecutive same-role messages
- Assert validation fails with rule "alternation"

### Test: `test_validator_catches_non_user_first`

- Payload where first message is assistant
- Assert validation fails with rule "user_first"

### Test: `test_validator_catches_missing_tool_result`

- Assistant with tool_use, next user message has no tool_result
- Assert validation fails with rule "tool_result_missing"

### Test: `test_validator_catches_orphan_tool_result`

- User message has tool_result for a tool_use ID that doesn't exist
- Assert validation fails with rule "tool_result_orphan"

### Test: `test_validator_catches_tool_result_ordering`

- Text block before tool_result in user message following tool_use
- Assert validation fails with rule "tool_result_ordering"
- Then verify correct ordering (tool_results first) passes

### Test: `test_validator_catches_cache_control_overflow`

- Payload with 5+ cache_control breakpoints
- Assert validation fails with rule "cache_control_budget"

### Test: `test_validator_accepts_valid_complex_payload`

- Multi-turn conversation with tool_use/tool_result, cache_control,
  system blocks, page table in user message
- Assert validation passes

## Files to reference

- `src/tinkuy/gateway/server.py` — count_tokens route, max_tokens=1 guard
- `src/tinkuy/gateway/_gateway.py` — `_inject_page_table`, `_merge_system`, `_synthesize`
- `src/tinkuy/formats/anthropic.py` — `_collect_system`, `_finalize_messages`, `_inject_page_table`
- `src/tinkuy/formats/validate.py` — pre-flight payload validator
- Existing test patterns: `tests/test_server.py`, `tests/test_gateway.py`
