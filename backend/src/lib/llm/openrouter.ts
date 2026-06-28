import type {
  LlmMessage,
  NormalizedToolCall,
  NormalizedToolResult,
  OpenAIToolSchema,
  StreamChatParams,
  StreamChatResult,
} from "./types";
import { createRawLlmStreamRecorder, logRawLlmStream } from "./rawStreamLog";

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";
const MAX_TOKENS = 16384;

// Strip the "openrouter/" namespace prefix before sending to the API.
// Model IDs in this codebase are stored as e.g. "openrouter/meta-llama/llama-3.1-70b-instruct"
// but OpenRouter expects just "meta-llama/llama-3.1-70b-instruct".
function toApiModelId(model: string): string {
  return model.startsWith("openrouter/") ? model.slice("openrouter/".length) : model;
}

type ChatMessage =
  | { role: "system"; content: string }
  | { role: "user" | "assistant"; content: string }
  | { role: "tool"; tool_call_id: string; content: string };

type ChatTool = {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
  };
};

type ToolCallDelta = {
  index: number;
  id?: string;
  type?: "function";
  function?: { name?: string; arguments?: string };
};

type StreamChunk = {
  choices?: {
    delta?: {
      content?: string | null;
      tool_calls?: ToolCallDelta[];
    };
    finish_reason?: string | null;
  }[];
  error?: { message?: string; code?: string | number };
};

function apiKey(override?: string | null): string {
  const key = override?.trim() || process.env.OPENROUTER_API_KEY?.trim() || "";
  if (!key) {
    throw new Error(
      "OpenRouter API key is not configured. Set OPENROUTER_API_KEY or add a user OpenRouter key.",
    );
  }
  return key;
}

function toMessages(
  systemPrompt: string,
  messages: LlmMessage[],
): ChatMessage[] {
  const result: ChatMessage[] = [];
  if (systemPrompt) {
    result.push({ role: "system", content: systemPrompt });
  }
  for (const m of messages) {
    result.push({ role: m.role, content: m.content });
  }
  return result;
}

function toChatTools(tools: OpenAIToolSchema[]): ChatTool[] {
  return tools.map((t) => ({
    type: "function",
    function: {
      name: t.function.name,
      description: t.function.description,
      parameters: t.function.parameters,
    },
  }));
}

function extractSseJson(buffer: string): { events: unknown[]; rest: string } {
  const events: unknown[] = [];
  const chunks = buffer.split(/\n\n/);
  const rest = chunks.pop() ?? "";

  for (const chunk of chunks) {
    const dataLines = chunk
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim());

    for (const data of dataLines) {
      if (!data || data === "[DONE]") continue;
      try {
        events.push(JSON.parse(data));
      } catch {
        // Incomplete events stay buffered until the next read.
      }
    }
  }

  return { events, rest };
}

function openRouterStreamFailureMessage(chunk: StreamChunk): string | null {
  if (chunk.error) {
    const message =
      typeof chunk.error.message === "string" && chunk.error.message.trim()
        ? chunk.error.message.trim()
        : "OpenRouter stream failed.";
    const code =
      chunk.error.code !== undefined && chunk.error.code !== null
        ? String(chunk.error.code).trim()
        : null;
    return code
      ? `OpenRouter error (${code}): ${message}`
      : `OpenRouter error: ${message}`;
  }
  return null;
}

function abortError(): Error {
  const err = new Error("Stream aborted.");
  err.name = "AbortError";
  return err;
}

function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) throw abortError();
}

export async function streamOpenRouter(
  params: StreamChatParams,
): Promise<StreamChatResult> {
  const {
    model,
    systemPrompt,
    tools = [],
    callbacks = {},
    runTools,
    apiKeys,
  } = params;
  const maxIter = params.maxIterations ?? 10;
  const key = apiKey(apiKeys?.openrouter);
  const apiModelId = toApiModelId(model);
  const chatTools = toChatTools(tools);

  let messages: ChatMessage[] = toMessages(systemPrompt, params.messages);
  let fullText = "";
  const rawStreamRecorder = createRawLlmStreamRecorder({
    provider: "openrouter",
    model,
  });

  try {
    for (let iter = 0; iter < maxIter; iter++) {
      throwIfAborted(params.abortSignal);

      const response = await fetch(OPENROUTER_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${key}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: apiModelId,
          messages,
          tools: chatTools.length ? chatTools : undefined,
          tool_choice: chatTools.length ? "auto" : undefined,
          max_tokens: MAX_TOKENS,
          stream: true,
        }),
        signal: params.abortSignal,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        const err = new Error(
          `OpenRouter request failed (${response.status}): ${text || response.statusText}`,
        );
        (err as { status?: number }).status = response.status;
        throw err;
      }

      if (!response.body) throw new Error("OpenRouter response had no body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      // Accumulators for this iteration.
      let buffer = "";
      let assistantText = "";
      // Map from tool-call index → accumulated state.
      const toolCallMap = new Map<
        number,
        { id: string; name: string; argumentsRaw: string }
      >();
      const startedToolCallIds = new Set<string>();

      while (true) {
        throwIfAborted(params.abortSignal);
        const { done, value } = await reader.read();
        if (done) break;

        const decoded = decoder.decode(value, { stream: true });
        logRawLlmStream({
          provider: "openrouter",
          model,
          iteration: iter,
          label: "sse_chunk",
          payload: decoded,
        });
        rawStreamRecorder?.record({
          iteration: iter,
          label: "sse_chunk",
          payload: decoded,
        });
        buffer += decoded;
        const extracted = extractSseJson(buffer);
        buffer = extracted.rest;

        for (const event of extracted.events as StreamChunk[]) {
          logRawLlmStream({
            provider: "openrouter",
            model,
            iteration: iter,
            label: "sse_event",
            payload: event,
          });
          rawStreamRecorder?.record({
            iteration: iter,
            label: "sse_event",
            payload: event,
          });

          const failureMessage = openRouterStreamFailureMessage(event);
          if (failureMessage) throw new Error(failureMessage);

          const delta = event.choices?.[0]?.delta;
          if (!delta) continue;

          // Text content delta.
          if (typeof delta.content === "string" && delta.content) {
            assistantText += delta.content;
            fullText += delta.content;
            callbacks.onContentDelta?.(delta.content);
          }

          // Tool call deltas — OpenAI-compatible streaming format.
          if (delta.tool_calls) {
            for (const tc of delta.tool_calls) {
              if (!toolCallMap.has(tc.index)) {
                toolCallMap.set(tc.index, {
                  id: tc.id ?? `tool-${tc.index}`,
                  name: tc.function?.name ?? "",
                  argumentsRaw: "",
                });
              }
              const entry = toolCallMap.get(tc.index)!;
              if (tc.id) entry.id = tc.id;
              if (tc.function?.name) entry.name = tc.function.name;
              if (tc.function?.arguments) {
                entry.argumentsRaw += tc.function.arguments;
              }

              // Fire onToolCallStart as soon as we have an id + name.
              if (entry.id && entry.name && !startedToolCallIds.has(entry.id)) {
                startedToolCallIds.add(entry.id);
                callbacks.onToolCallStart?.({
                  id: entry.id,
                  name: entry.name,
                  input: {},
                });
              }
            }
          }
        }
      }

      throwIfAborted(params.abortSignal);

      // Build the finalized tool calls from accumulated deltas.
      const toolCalls: NormalizedToolCall[] = [];
      for (const [, entry] of toolCallMap) {
        let input: Record<string, unknown> = {};
        try {
          const parsed = JSON.parse(entry.argumentsRaw || "{}") as unknown;
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
            input = parsed as Record<string, unknown>;
          }
        } catch {
          input = {};
        }
        toolCalls.push({ id: entry.id, name: entry.name, input });
      }

      if (!toolCalls.length || !runTools) {
        break;
      }

      // Append the assistant turn (with tool_calls) and the tool results.
      messages.push({
        role: "assistant",
        content: assistantText,
        // OpenAI-compatible tool_calls field on the assistant message.
        ...(toolCalls.length
          ? {
              tool_calls: toolCalls.map((c) => ({
                id: c.id,
                type: "function",
                function: {
                  name: c.name,
                  arguments: JSON.stringify(c.input),
                },
              })),
            }
          : {}),
      } as ChatMessage);

      const results: NormalizedToolResult[] = await runTools(toolCalls);
      throwIfAborted(params.abortSignal);

      for (const result of results) {
        messages.push({
          role: "tool",
          tool_call_id: result.tool_use_id,
          content: result.content,
        });
      }
    }

    await rawStreamRecorder?.flush("completed");
    return { fullText };
  } catch (error) {
    await rawStreamRecorder?.flush("error", error);
    throw error;
  }
}

export async function completeOpenRouterText(params: {
  model: string;
  systemPrompt?: string;
  user: string;
  maxTokens?: number;
  apiKeys?: { openrouter?: string | null };
}): Promise<string> {
  const key = apiKey(params.apiKeys?.openrouter);
  const apiModelId = toApiModelId(params.model);

  const messages: ChatMessage[] = [];
  if (params.systemPrompt) {
    messages.push({ role: "system", content: params.systemPrompt });
  }
  messages.push({ role: "user", content: params.user });

  const response = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: apiModelId,
      messages,
      max_tokens: params.maxTokens ?? 512,
      stream: false,
    }),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(
      `OpenRouter request failed (${response.status}): ${text || response.statusText}`,
    );
  }

  const json = (await response.json()) as {
    choices?: { message?: { content?: string | null } }[];
    error?: { message?: string };
  };

  if (json.error?.message) {
    throw new Error(`OpenRouter error: ${json.error.message}`);
  }

  return json.choices?.[0]?.message?.content ?? "";
}

export type { NormalizedToolResult };
