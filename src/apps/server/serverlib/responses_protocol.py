"""Responses API の入力検証と応答エンコード。

明示的に列挙した metadata は推論に使用しない。暗号化 reasoning は返さない。
web_search と namespace はモデルに渡さず、未知のキーは拒否する。
保存済み response の参照には対応しない。
"""

from __future__ import annotations

import json
import time
import uuid

from .openai_protocol import OpenAIProtocolError, parse_chat_request, wire_tool_calls


def reject(param, message=None):
    raise OpenAIProtocolError(message or f"{param} is not supported", param=param,
                              code="unsupported_parameter")


def fields(value, allowed, param):
    if not isinstance(value, dict):
        raise OpenAIProtocolError(f"{param} must be an object", param=param)
    for key in value:
        if key not in allowed:
            reject(f"{param}.{key}")


def text_content(value, param):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise OpenAIProtocolError(f"{param} must be text", param=param)
    parts = []
    for i, part in enumerate(value):
        p = f"{param}[{i}]"
        fields(part, {"type", "text", "annotations"}, p)
        if part.get("type") not in ("input_text", "output_text"):
            reject(f"{p}.type")
        if not isinstance(part.get("text"), str):
            raise OpenAIProtocolError(f"{p}.text must be a string", param=f"{p}.text")
        if part.get("annotations") not in (None, []):
            reject(f"{p}.annotations")
        parts.append(part["text"])
    return "".join(parts)


def input_messages(value):
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value:
        raise OpenAIProtocolError("input must be a string or non-empty array", param="input")
    messages = []
    pending_reasoning = None
    for i, item in enumerate(value):
        p = f"input[{i}]"
        if not isinstance(item, dict):
            raise OpenAIProtocolError(f"{p} must be an object", param=p)
        kind = item.get("type", "message")
        if kind == "reasoning":
            fields(item, {"type", "id", "summary", "content", "status", "encrypted_content"}, p)
            if item.get("encrypted_content") is not None:
                reject(p + ".encrypted_content")
            if pending_reasoning is not None:
                reject(p, "consecutive reasoning items are not supported")
            content = item.get("content") or item.get("summary") or []
            if not isinstance(content, list):
                raise OpenAIProtocolError("reasoning content must be an array", param=p)
            chunks = []
            for part in content:
                fields(part, {"type", "text"}, p + ".content")
                if part.get("type") not in ("reasoning_text", "summary_text") or not isinstance(part.get("text"), str):
                    reject(p + ".content")
                chunks.append(part["text"])
            pending_reasoning = "".join(chunks)
        elif kind == "message":
            fields(item, {"type", "role", "content", "id", "status"}, p)
            role = item.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                reject(p + ".role")
            message = {"role": role, "content": text_content(item.get("content"), p + ".content")}
            if pending_reasoning is not None:
                if role != "assistant":
                    reject(p, "reasoning must precede an assistant item")
                message["reasoning_content"] = pending_reasoning
                pending_reasoning = None
            messages.append(message)
        elif kind == "function_call":
            fields(item, {"type", "id", "call_id", "name", "arguments", "status"}, p)
            for key in ("call_id", "name", "arguments"):
                if not isinstance(item.get(key), str) or (key != "arguments" and not item[key]):
                    raise OpenAIProtocolError(f"{p}.{key} must be a string", param=f"{p}.{key}")
            call = {"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": item["arguments"]}}
            if messages and messages[-1]["role"] == "assistant" and messages[-1].get("tool_calls") and pending_reasoning is None:
                messages[-1]["tool_calls"].append(call)
            else:
                msg = {"role": "assistant", "content": "", "tool_calls": [call]}
                if pending_reasoning is not None:
                    msg["reasoning_content"] = pending_reasoning
                    pending_reasoning = None
                messages.append(msg)
        elif kind == "function_call_output":
            fields(item, {"type", "id", "call_id", "output", "status"}, p)
            if pending_reasoning is not None:
                reject(p, "reasoning must precede an assistant item")
            if not isinstance(item.get("call_id"), str) or not item["call_id"] or not isinstance(item.get("output"), str):
                raise OpenAIProtocolError("function output requires call_id and string output", param=p)
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
        else:
            reject(p + ".type")
    if pending_reasoning is not None:
        reject("input", "reasoning must precede an assistant item")
    return messages


def parse_responses_request(body, *, inference_budget_tokens):
    allowed = {"model", "input", "instructions", "tools", "tool_choice", "max_output_tokens", "temperature", "top_p", "seed", "reasoning", "store", "metadata", "user", "parallel_tool_calls", "truncation", "include", "stream", "previous_response_id", "conversation", "text", "prompt_cache_key", "client_metadata"}
    if not isinstance(body, dict):
        raise OpenAIProtocolError("request body must be a JSON object")
    for key in body:
        if key not in allowed:
            reject(key)
    for key in ("previous_response_id", "conversation"):
        if body.get(key) is not None:
            reject(key)
    response_format = None
    if body.get("text") is not None:
        fields(body["text"], {"format"}, "text")
        fmt = body["text"].get("format")
        if fmt is not None and fmt != {"type": "text"}:
            fields(fmt, {"type", "name", "description", "schema", "strict"}, "text.format")
            if fmt.get("type") != "json_schema":
                reject("text.format")
            response_format = {"type": "json_schema", "json_schema": {k: v for k, v in fmt.items() if k != "type"}}
    if body.get("max_output_tokens") is not None:
        value = body["max_output_tokens"]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= (1 << 63) - 1:
            raise OpenAIProtocolError("max_output_tokens must be a positive integer", param="max_output_tokens")
    messages = input_messages(body.get("input"))
    if body.get("instructions") is not None:
        if not isinstance(body["instructions"], str):
            raise OpenAIProtocolError("instructions must be a string", param="instructions")
        messages.insert(0, {"role": "system", "content": body["instructions"]})
    chat = {"model": body.get("model"), "messages": messages}
    if response_format is not None:
        chat["response_format"] = response_format
    for source, target in (("stream", "stream"), ("max_output_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p"), ("seed", "seed"), ("tool_choice", "tool_choice")):
        if source in body:
            chat[target] = body[source]
    choice = chat.get("tool_choice")
    if isinstance(choice, dict):
        fields(choice, {"type", "name"}, "tool_choice")
        if choice.get("type") != "function" or not isinstance(choice.get("name"), str):
            raise OpenAIProtocolError("invalid tool_choice", param="tool_choice")
        chat["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("reasoning") is not None:
        fields(body["reasoning"], {"effort", "summary"}, "reasoning")
        if body["reasoning"].get("summary") not in (None, "auto"):
            reject("reasoning.summary")
        chat["reasoning_effort"] = body["reasoning"].get("effort")
    if body.get("tools") is not None:
        if not isinstance(body["tools"], list):
            raise OpenAIProtocolError("tools must be an array", param="tools")
        converted = []
        for i, tool in enumerate(body["tools"]):
            p = f"tools[{i}]"
            if not isinstance(tool, dict):
                raise OpenAIProtocolError(f"{p} must be an object", param=p)
            if tool.get("type") == "web_search":
                fields(tool, {"type", "external_web_access"}, p)
                continue
            if tool.get("type") == "namespace":
                fields(tool, {"type", "name", "description", "tools"}, p)
                nested = tool.get("tools", [])
                if not isinstance(nested, list):
                    raise OpenAIProtocolError(f"{p}.tools must be an array", param=p + ".tools")
                for j, child in enumerate(nested):
                    if not isinstance(child, dict):
                        raise OpenAIProtocolError(f"{p}.tools[{j}] must be an object", param=f"{p}.tools[{j}]")
                    if child.get("strict") is not None and child.get("strict") is not False:
                        reject(f"{p}.tools[{j}].strict")
                continue
            fields(tool, {"type", "name", "description", "parameters", "strict"}, p)
            if tool.get("type") != "function":
                reject(p + ".type")
            converted.append({"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}})
        chat["tools"] = converted or None
    for key in ("store", "parallel_tool_calls"):
        if body.get(key) is not None and not isinstance(body[key], bool):
            raise OpenAIProtocolError(f"{key} must be a boolean", param=key)
    if body.get("store") is True:
        reject("store", "store=true is not supported")
    if body.get("prompt_cache_key") is not None and not isinstance(body["prompt_cache_key"], str):
        raise OpenAIProtocolError("prompt_cache_key must be a string", param="prompt_cache_key")
    if body.get("client_metadata") is not None and not isinstance(body["client_metadata"], dict):
        raise OpenAIProtocolError("client_metadata must be an object", param="client_metadata")
    if body.get("parallel_tool_calls") is False:
        reject("parallel_tool_calls")
    if body.get("truncation") not in (None, "disabled"):
        reject("truncation")
    if body.get("include") not in (None, [], ["reasoning.encrypted_content"]):
        reject("include")
    if body.get("metadata") is not None and not isinstance(body["metadata"], dict):
        raise OpenAIProtocolError("metadata must be an object", param="metadata")
    if body.get("user") is not None and not isinstance(body["user"], str):
        raise OpenAIProtocolError("user must be a string", param="user")
    try:
        return parse_chat_request(chat, inference_budget_tokens=inference_budget_tokens)
    except OpenAIProtocolError as exc:
        if exc.param and exc.param.startswith("response_format"):
            exc.param = exc.param.replace("response_format.json_schema.schema", "text.format.schema", 1).replace("response_format", "text.format", 1)
        exc.param = {"messages": "input", "max_tokens": "max_output_tokens", "reasoning_effort": "reasoning.effort"}.get(exc.param, exc.param)
        raise


def response_object(chat):
    return {"id": "resp_" + uuid.uuid4().hex, "object": "response", "created_at": int(time.time()), "status": "in_progress", "model": chat.model, "output": [], "error": None, "incomplete_details": None, "usage": None}


def usage(prompt, completion, cached, reasoning=None):
    if prompt is None or completion is None:
        return None
    result = {"input_tokens": prompt, "output_tokens": completion, "total_tokens": prompt + completion, "input_tokens_details": {"cached_tokens": cached or 0}}
    if reasoning is not None:
        result["output_tokens_details"] = {"reasoning_tokens": reasoning}
    return result


def encode_response(chat, result):
    response = response_object(chat)
    output = response["output"]
    if result.reasoning_content:
        output.append({"id": "rs_" + uuid.uuid4().hex, "type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": result.reasoning_content}]})
    if result.content:
        output.append({"id": "msg_" + uuid.uuid4().hex, "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": result.content, "annotations": []}]})
    for call in wire_tool_calls(result.tool_calls):
        output.append({"id": "fc_" + uuid.uuid4().hex, "type": "function_call", "status": "completed", "call_id": call["id"], "name": call["function"]["name"], "arguments": call["function"]["arguments"]})
    response["status"] = "incomplete" if result.finish_reason == "length" else "completed"
    if response["status"] == "incomplete":
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    response["usage"] = usage(result.prompt_tokens, result.completion_tokens, result.cached_tokens, result.reasoning_tokens)
    return response
