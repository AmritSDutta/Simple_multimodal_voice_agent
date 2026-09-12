# MCP Server Integration Discussion

**Date:** 2026-02-13
**Topic:** Integrating MCP (Model Context Protocol) servers with LangChain Chat models in a LangGraph-based agent

---

## Original Question

> "How to add MCP server for ChatXXX construct of LangChain, or do I need LangChain agents?"

---

## Key Concepts

### MCP (Model Context Protocol)
- Protocol for connecting AI assistants to external data sources and tools
- Provides tools that can be used by LLMs
- Examples: filesystem access, web search, database queries

### Chat Models vs Agents
| Feature | Chat Model + Tools | LangChain Agent |
|---------|-------------------|-----------------|
| Tool Usage | Single-turn (LLM decides) | Multi-step orchestration |
| Tool Execution | Manual (you implement) | Automatic |
| LangGraph | ✅ Works great | ❌ Overkill (you already have LangGraph) |
| Control | Direct LLM control | Agent controls flow |

---

## Challenge #1: Chat Models Don't Auto-Invoke Tools

**Issue:** When you use `.bind_tools()` on a Chat model, it outputs tool calls but doesn't execute them.

### Solutions Discussed:

#### Option 1: Manual Tool Execution (in reasoning node)
- Check if LLM output has `tool_calls`
- Execute tools manually
- Re-invoke LLM with tool results
- ✅ Simple, direct control
- ⚠️ Mixes concerns in reasoning node

#### Option 2: Use AgentExecutor Inside Node
- Wrap LLM in LangChain AgentExecutor
- Let LangChain handle tool execution
- ✅ Automatic tool handling
- ❌ Hides tool execution from traces
- ❌ Adds unnecessary abstraction layer

#### Option 3: Separate Tool-Calling Node (Recommended)
- Create dedicated `execute_tools` node in LangGraph
- Conditional edge: reasoning → tools → reasoning
- ✅ LangGraph-native pattern
- ✅ Clean separation of concerns
- ✅ Visible in traces/LangSmith
- ✅ Fits existing architecture

---

## Challenge #2: Vision Models Don't Support Function Calling

**Issue:** The project uses excellent vision models that lack tool calling support:
- `gemma-3-27b-it` (Google Gemini)
- `GLM-4.6V-Flash` (Zhipu)
- `qwen3-vl:235b-instruct-cloud` (Ollama)

These models can understand images but can't output structured tool calls.

### Solutions Discussed:

#### Option 1: Two-Stage Pipeline (Recommended)
```
Input with Image → Vision Model (describe) → Tool-Calling Model (act) → Execute Tools → Final Response
```

**Flow:**
1. Vision model analyzes image and describes contents
2. Tool-calling model (OpenAI/Anthropic) decides if tools needed
3. Execute tools if required
4. Combine results for final answer

**Pros:**
- ✅ Best of both worlds (vision + tools)
- ✅ Clean separation of concerns
- ✅ Vision models do what they're good at
- ✅ Tool-capable models handle actions
- ✅ Fits weighted provider distribution

**Code Sketch:**
```python
if has_images and needs_tools:
    # Stage 1: Vision model describes image
    vision_llm = get_chat_llm(provider="ollama")  # qwen3-vl
    vision_response = await vision_llm.ainvoke([
        HumanMessage(content=[vision_prompt, images])
    ])

    # Stage 2: Tool-calling model takes action
    tool_llm = get_chat_llm(provider="openai")  # gpt-5-nano
    tool_llm = tool_llm.bind_tools(mcp_tools)
    response = await tool_llm.ainvoke([
        HumanMessage(content=f"Image: {vision_response}\nQuestion: {user_q}")
    ])
```

#### Option 2: Prompt-Based Tool Selection
- Ask vision model to output `[TOOL: name]` and `[ARGS: {...}]`
- Parse and execute manually
- Re-invoke with tool results
- ⚠️ Fragile parsing
- ⚠️ Model may not follow format reliably

#### Option 3: Provider-Specific Routing
- Detect if tools are needed (keyword heuristics)
- Route to tool-capable provider if yes
- Use vision models otherwise
- ✅ Simple implementation
- ⚠️ Heuristics may miss edge cases

#### Option 4: Image-First Processing
- Always run vision model on images first
- Extract image information
- Pass to tool-calling model for decision/action
- ✅ Good for vision-heavy workloads
- ⚠️ Extra latency for all image inputs

---

## Recommended Architecture for This Project

### Integration Points

1. **Create MCP Client Module** (`src/flow_agent/llms/mcp_tools.py`)
   ```python
   from langchain_mcp_adapters import MultiServerMCPClient

   async def get_mcp_tools():
       """Get tools from configured MCP servers."""
       mcp_client = MultiServerMCPClient({
           "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
           "brave-search": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"]}
       })
       await mcp_client.initialize()
       return await mcp_client.get_tools()
   ```

2. **Update Configuration** (`src/flow_agent/config.py`)
   ```python
   class Settings(BaseSettings):
       # MCP Configuration
       ENABLE_MCP_TOOLS: bool = False  # Master switch
       MCP_ALLOWED_DIR: str = "/tmp"
       MCP_TOOL_TRIGGER_KEYWORDS: list[str] = ["search", "find", "latest", "news", "price"]
   ```

3. **Implement Two-Stage Reasoning** (`src/flow_agent/utils/nodes.py`)
   ```python
   async def call_langchain_reasoning_model(state: State, config: RunnableConfig):
       """Two-stage: Vision model → Tool-calling model."""
       messages = state["messages"]
       last_human = [m for m in messages if m.type == "human"][-1]

       has_images = _has_image_content(last_human.content)
       needs_tools = _might_need_tools(last_human.content)

       if has_images and needs_tools:
           # Stage 1: Vision understanding
           vision_llm = get_chat_llm(provider="ollama")
           vision_response = await call_llm_safely(vision_llm, vision_input, config)

           # Stage 2: Tool calling with context
           tool_llm = get_chat_llm(provider="openai", with_tools=True)
           tool_response = await call_llm_safely(tool_llm, combined_input, config)

           # Execute tools if called
           if hasattr(tool_response, 'tool_calls') and tool_response.tool_calls:
               # Execute and get final response
               pass
       else:
           # Normal single-stage flow
           llm = get_chat_llm(provider=None)
           response = await call_llm_safely(llm, prepare_llm_input(messages), config)

       return process_response(state, response)
   ```

4. **Add Tool Execution Helper** (`src/flow_agent/utils/nodes.py`)
   ```python
   async def execute_tool_calls(tool_calls, llm, config):
       """Execute tool calls and return final response."""
       from src.flow_agent.llms.mcp_tools import get_mcp_tools
       tools = await get_mcp_tools()
       tools_dict = {t.name: t for t in tools}

       tool_results = []
       for tool_call in tool_calls:
           tool = tools_dict.get(tool_call['name'])
           if tool:
               result = await tool.ainvoke(tool_call['args'])
               tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))

       # Re-invoke LLM with tool results
       final_response = await llm.ainvoke(messages + tool_results)
       return final_response
   ```

---

## Configuration Options

### When MCP Tools Are Enabled

Update provider distribution to prioritize tool-capable models:

```python
# In src/flow_agent/config.py
class Settings(BaseSettings):
    # Vision models WITHOUT tool calling
    VISION_PROVIDER_DISTRIBUTION: dict = {
        'gemini': 0.7,   # gemma-3-27b-it
        'ollama': 0.2,   # qwen3-vl
        'zhipu': 0.1,    # GLM-4.6V-Flash
        'openai': 0.01   # gpt-5-nano (fallback)
    }

    # Tool-capable providers (for MCP tools)
    TOOL_CAPABLE_PROVIDERS: dict = {
        'openai': 0.7,     # gpt-5-nano (primary)
        'anthropic': 0.2,  # claude-4.5-sonnet
        'gemini': 0.1      # gemini-2.5-flash (has tools)
    }

    # Or use distribution override when tools enabled
    ENABLE_MCP_TOOLS: bool = False
```

---

## Implementation Checklist

When ready to implement:

- [ ] Create `src/flow_agent/llms/mcp_tools.py` with MCP client
- [ ] Update `src/flow_agent/config.py` with MCP settings
- [ ] Add `with_tools` parameter to `get_chat_llm()`
- [ ] Implement two-stage reasoning in `call_langchain_reasoning_model()`
- [ ] Add helper functions: `_has_image_content()`, `_might_need_tools()`, `_extract_images()`, `_extract_text()`
- [ ] Add `execute_tool_calls()` helper function
- [ ] Update `langgraph.json` dependencies (add `langchain-mcp-adapters`)
- [ ] Test with filesystem server
- [ ] Test with search server
- [ ] Add tracing for tool execution
- [ ] Update tests to cover tool execution paths

---

## Key Dependencies

```toml
# Add to pyproject.toml
langchain-mcp-adapters = "^0.1.0"

# Add to langgraph.json dependencies
"langchain-mcp-adapters"
```

---

## Testing Strategy

1. **Unit Tests**
   - Test `_has_image_content()` with various content formats
   - Test `_might_need_tools()` with trigger keywords
   - Mock MCP tool execution
   - Test two-stage flow with mock LLMs

2. **Integration Tests**
   - Test with real MCP server (e.g., filesystem)
   - Test vision → tool pipeline end-to-end
   - Verify tool results are incorporated correctly

3. **Evaluation Tests**
   - LangSmith evaluations for tool usage accuracy
   - Arize Phoenix evals for multimodal + tool scenarios

---

## Alternative: Skip MCP, Use LangChain Tools Directly

If MCP complexity isn't needed, LangChain has built-in tools:

```python
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun

# Already using DuckDuckGo in this project!
# Can add more: file system, database, APIs, etc.
```

**Consider this if:**
- MCP servers feel over-engineered for your needs
- You only need 2-3 tools
- You prefer direct Python implementation

---

## Summary

**Problem:** Integrate MCP tools with LangGraph agent using non-tool-capable vision models

**Solution:** Two-stage pipeline
1. Vision models (gemma, zhipu, qwen) analyze images
2. Tool-capable models (OpenAI, Anthropic) execute MCP tools
3. Results combined for final response

**Status:** ✅ Designed, ready to implement when needed

**Files to Modify:**
- `src/flow_agent/config.py` - Add MCP settings
- `src/flow_agent/llms/LangChainChatLLM.py` - Add `with_tools` parameter
- `src/flow_agent/utils/nodes.py` - Implement two-stage reasoning
- `langgraph.json` - Add MCP adapter dependency
- New: `src/flow_agent/llms/mcp_tools.py` - MCP client wrapper
