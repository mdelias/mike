import type { Provider } from "./types";

// ---------------------------------------------------------------------------
// Canonical model IDs
// ---------------------------------------------------------------------------
// Main-chat tier (top-end) — user picks one of these per message.
export const CLAUDE_MAIN_MODELS = [
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
] as const;
export const GEMINI_MAIN_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
] as const;
export const OPENAI_MAIN_MODELS = ["gpt-5.5", "gpt-5.4"] as const;

// Mid-tier (used for tabular review) — user picks one in account settings.
export const CLAUDE_MID_MODELS = ["claude-sonnet-4-6"] as const;
export const GEMINI_MID_MODELS = ["gemini-3.5-flash", "gemini-3-flash-preview"] as const;
export const OPENAI_MID_MODELS = ["gpt-5.4"] as const;

// Low-tier (used for title generation, lightweight extractions) — user picks
// one in account settings.
export const CLAUDE_LOW_MODELS = ["claude-haiku-4-5"] as const;
export const GEMINI_LOW_MODELS = ["gemini-3.1-flash-lite-preview"] as const;
export const OPENAI_LOW_MODELS = ["gpt-5.4-lite"] as const;

// OpenRouter models — prefixed with "openrouter/" to distinguish them from
// native provider model IDs. The prefix is stripped before sending to the API.
export const OPENROUTER_MAIN_MODELS = [
    "openrouter/anthropic/claude-3.5-sonnet",
    "openrouter/openai/gpt-4o",
    "openrouter/google/gemini-2.0-flash-001",
    "openrouter/meta-llama/llama-3.1-70b-instruct",
    "openrouter/mistralai/mistral-large",
    "openrouter/deepseek/deepseek-chat",
    "openrouter/x-ai/grok-2-1212",
] as const;
export const OPENROUTER_MID_MODELS = [
    "openrouter/anthropic/claude-3.5-haiku",
    "openrouter/openai/gpt-4o-mini",
    "openrouter/meta-llama/llama-3.1-8b-instruct",
    "openrouter/mistralai/mistral-small",
] as const;
export const OPENROUTER_LOW_MODELS = [
    "openrouter/google/gemini-2.0-flash-lite-001",
    "openrouter/meta-llama/llama-3.2-3b-instruct",
] as const;

export const DEFAULT_MAIN_MODEL = "gemini-3-flash-preview";
export const DEFAULT_TITLE_MODEL = "gemini-3.1-flash-lite-preview";
export const DEFAULT_TABULAR_MODEL = "gemini-3-flash-preview";

const ALL_MODELS = new Set<string>([
    ...CLAUDE_MAIN_MODELS,
    ...GEMINI_MAIN_MODELS,
    ...OPENAI_MAIN_MODELS,
    ...CLAUDE_MID_MODELS,
    ...GEMINI_MID_MODELS,
    ...OPENAI_MID_MODELS,
    ...CLAUDE_LOW_MODELS,
    ...GEMINI_LOW_MODELS,
    ...OPENAI_LOW_MODELS,
    ...OPENROUTER_MAIN_MODELS,
    ...OPENROUTER_MID_MODELS,
    ...OPENROUTER_LOW_MODELS,
]);

// ---------------------------------------------------------------------------
// Provider inference
// ---------------------------------------------------------------------------

export function providerForModel(model: string): Provider {
    if (model.startsWith("claude")) return "claude";
    if (model.startsWith("gemini")) return "gemini";
    if (model.startsWith("gpt-")) return "openai";
    if (model.startsWith("openrouter/")) return "openrouter";
    throw new Error(`Unknown model id: ${model}`);
}

export function resolveModel(id: string | null | undefined, fallback: string): string {
    if (id && ALL_MODELS.has(id)) return id;
    return fallback;
}
