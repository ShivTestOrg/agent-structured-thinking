import OpenAI from "openai";
import { v4 as uuidv4 } from "uuid";

// --- Configuration and Utility Types ---

export type LlmStepResponse = {
  output: string;
  reasoning: string;
  confidence: number;
};

export type ThinkingStepType = "analysis" | "decomposition" | "synthesis" | "evaluation" | "conclusion" | "exploration";

export interface ThinkingStep {
  id: string;
  type: ThinkingStepType;
  description: string;
  input: string;
  output: string;
  reasoning: string;
  confidence: number;
  timestamp: string;
}

export interface ChainMetadata {
  startTime: string;
  endTime: string;
  duration: number;
  totalSteps: number;
}

export interface ThinkingChain {
  id: string;
  problem: string;
  steps: ThinkingStep[];
  conclusion: string;
  overallConfidence: number;
  metadata: ChainMetadata;
}

export interface AnalysisResult {
  originalInput: string;
  structuredAnalysis: ThinkingChain;
  summary: string;
  keyInsights: string[];
  recommendations: string[];
  confidence: number;
}

export interface AnalysisCallbacks {
  onStepComplete?: (step: ThinkingStep) => void;
}

export interface ThinkingContext {
  objectives?: string[];
}

interface StepConfig {
  type: ThinkingStepType;
  description: string;
}

interface ThinkingFramework {
  steps: StepConfig[];
}

export interface LlmConfig {
  apiKey: string;
  model: string;
  temperature?: number;
  maxTokens?: number;
  baseURL?: string;
  useDummyData?: boolean; // New flag for dummy data
}

export interface SearchClientConfig {
  apiKey: string;
  useDummyData?: boolean;
}

const DUMMY_SEARCH_RESULTS: { [key: string]: string } = {
  "latest AI trends": "The latest AI trends include multimodal models, edge AI optimization, and a growing focus on ethical AI and regulation.",
  "structured thinking plan": "A standard structured thinking plan involves Analysis, Decomposition, Exploration, Synthesis, Evaluation, and Conclusion.",
  "impact of remote work":
    "Remote work has increased employee flexibility but also introduced challenges in team cohesion and data security. Productivity reports are mixed.",
};

const DUMMY_PLAN: ThinkingFramework = {
  steps: [
    { type: "analysis", description: "Understand the core problem and its constraints." },
    { type: "decomposition", description: "Break the problem into manageable sub-questions." },
    { type: "exploration", description: "Search for external, up-to-date data relevant to the problem." },
    { type: "synthesis", description: "Combine gathered information to form initial hypotheses or solutions." },
    { type: "evaluation", description: "Critically assess hypotheses against constraints and evidence." },
    { type: "conclusion", description: "Formulate the final answer, summary, and action plan." },
  ],
};

const DUMMY_LLM_STEP_RESPONSE: LlmStepResponse = {
  output: "This is a dummy output for the current step.",
  reasoning: "The reasoning is based on simulated data to bypass actual API calls.",
  confidence: 0.9,
};

export class SearchClient {
  private readonly _apiKey: string;
  private readonly _useDummyData: boolean;

  constructor(config: SearchClientConfig) {
    if (!config.apiKey && !config.useDummyData) throw new Error("SearchClient requires an apiKey unless useDummyData is true.");
    this._apiKey = config.apiKey;
    this._useDummyData = !!config.useDummyData;
  }

  public async search(query: string): Promise<string> {
    if (this._useDummyData) {
      console.log(`[DUMMY MODE] Simulating search for: "${query}"`);
      // simulate network delay
      await new Promise((resolve) => setTimeout(resolve, 6000));
      return DUMMY_SEARCH_RESULTS[query] || `[DUMMY RESULT] Fictional data for query: "${query}"`;
    }

    try {
      const response = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key: this._apiKey,
          query: query,
          search_depth: "advanced",
          include_answer: true,
          max_results: 5,
        }),
      });
      if (!response.ok) throw new Error(`Tavily API error: ${response.statusText}`);
      const data = await response.json();
      return data.answer || JSON.stringify(data.results);
    } catch (error) {
      console.error("Search failed:", error);
      return `Search failed with query "${query}". Reason: ${error instanceof Error ? error.message : "Unknown error"}`;
    }
  }
}

export class ChatbotClient {
  private readonly _openai: OpenAI | null;
  private readonly _config: LlmConfig;
  private readonly _searchClient: SearchClient;
  private readonly _useDummyData: boolean;

  constructor(llmConfig: LlmConfig, searchClient: SearchClient) {
    this._useDummyData = !!llmConfig.useDummyData;
    if (!this._useDummyData) {
      this._openai = new OpenAI({ apiKey: llmConfig.apiKey, baseURL: llmConfig.baseURL, maxRetries: 3 });
      if (!llmConfig.apiKey) throw new Error("ChatbotClient requires an apiKey unless useDummyData is true.");
    } else {
      this._openai = null;
    }

    this._config = { temperature: 0.1, maxTokens: 4096, ...llmConfig };
    this._searchClient = searchClient;
  }

  public async generateStructuredResponse<T>(systemPrompt: string, userPrompt: string, responseSchema: object, toolName: string): Promise<T> {
    if (this._useDummyData) {
      return this._handleDummyStructuredResponse<T>(toolName);
    }

    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ];

    const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
      {
        type: "function",
        function: {
          name: "search_the_web",
          description: "Searches the web for current, relevant information on a topic.",
          parameters: { type: "object", properties: { query: { type: "string", description: "A concise, targeted search query." } }, required: ["query"] },
        },
      },
      {
        type: "function",
        function: {
          name: toolName,
          description: "Formats the final structured output.",
          parameters: { type: "object", ...responseSchema, additionalProperties: false },
        },
      },
    ];

    let attempts = 0;
    while (attempts < 5) {
      if (!this._openai) throw new Error("OpenAI client not initialized.");
      const response = await this._openai.chat.completions.create({
        model: this._config.model,
        messages: messages,
        temperature: this._config.temperature,
        max_tokens: this._config.maxTokens,
        tools: tools,
        tool_choice: "auto",
      });

      const responseMessage = response.choices[0].message;
      messages.push(responseMessage);

      const toolResult = await this._processToolCalls<T>(responseMessage, messages, toolName);
      if (toolResult !== undefined) {
        return toolResult;
      }
      attempts++;
    }

    throw new Error("LLM failed to produce the required structured output after multiple attempts.");
  }

  private async _handleDummyStructuredResponse<T>(toolName: string): Promise<T> {
    // simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 6000));
    console.log(`[DUMMY MODE] Simulating LLM call for tool: ${toolName}`);
    if (toolName === "create_thinking_plan") {
      return DUMMY_PLAN as T;
    }
    if (toolName === "execute_thinking_step") {
      return DUMMY_LLM_STEP_RESPONSE as T;
    }
    if (toolName === "extract_insights") {
      return { insights: ["Dummy Insight 1", "Dummy Insight 2"] } as T;
    }
    return { output: "Dummy response", reasoning: "Simulated", confidence: 0.5 } as T;
  }

  private async _processToolCalls<T>(
    responseMessage: OpenAI.Chat.Completions.ChatCompletionMessage,
    messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
    toolName: string
  ): Promise<T | undefined> {
    if (responseMessage.tool_calls) {
      for (const toolCall of responseMessage.tool_calls) {
        if (toolCall.type === "function" && toolCall.function.name === "search_the_web") {
          const args = JSON.parse(toolCall.function.arguments);
          const searchResult = await this._searchClient.search(args.query);
          messages.push({ tool_call_id: toolCall.id, role: "tool", content: `Search results for "${args.query}":\n${searchResult}` });
        } else if (toolCall.type === "function" && toolCall.function.name === toolName) {
          return JSON.parse(toolCall.function.arguments) as T;
        }
      }
    } else if (responseMessage.content) {
      throw new Error("LLM provided a text response instead of the required structured output tool call.");
    }
    return undefined;
  }
}

export class StructuredThinkingEngine {
  private readonly _llmClient: ChatbotClient;
  private readonly _contextCharLimit = 16000;

  constructor(llmClient: ChatbotClient) {
    this._llmClient = llmClient;
  }

  public async analyzeStructured(input: string, context?: ThinkingContext, callbacks?: AnalysisCallbacks): Promise<AnalysisResult> {
    const framework = await this._generateExecutionPlan(input);
    const thinkingChain = await this._executeThinkingChain(input, framework, context, callbacks);
    const keyInsights = await this._extractKeyInsights(thinkingChain);

    return {
      originalInput: input,
      structuredAnalysis: thinkingChain,
      summary: this._generateSummary(thinkingChain),
      keyInsights: keyInsights,
      recommendations: this._generateRecommendations(thinkingChain),
      confidence: thinkingChain.overallConfidence,
    };
  }

  private async _generateExecutionPlan(originalInput: string): Promise<ThinkingFramework> {
    const systemPrompt =
      "You are an elite AI meta-strategist. Your purpose is to deconstruct a user's problem into a sequence of logical reasoning steps. Formulate a plan that is efficient, comprehensive, and leads to a clear, actionable outcome. If the problem requires external, real-time, or factual data that you do not possess, you MUST include an 'exploration' step in your plan.";

    const userPrompt = `Generate a structured thinking plan to address the following problem: "${originalInput}".

The available step types are:
- analysis: Examine the problem's components, context, and unspoken assumptions.
- decomposition: Break the problem into smaller, manageable parts.
- exploration: Search the web to gather necessary external or real-time information.
- synthesis: Combine information to generate solutions, hypotheses, or insights.
- evaluation: Assess options or ideas against criteria and evidence.
- conclusion: Formulate a final summary and action plan.

Your plan must be a logical sequence of these steps. Be strategic about when to use 'exploration' to gather facts.`;

    const planSchema = {
      type: "object",
      properties: {
        steps: {
          type: "array",
          items: {
            type: "object",
            properties: {
              type: { type: "string", enum: ["analysis", "decomposition", "exploration", "synthesis", "evaluation", "conclusion"] },
              description: { type: "string" },
            },
            required: ["type", "description"],
          },
        },
      },
      required: ["steps"],
    };

    const plan = await this._llmClient.generateStructuredResponse<ThinkingFramework>(systemPrompt, userPrompt, planSchema, "create_thinking_plan");
    if (!plan.steps || plan.steps.length === 0) throw new Error("LLM generated an empty plan.");
    return plan;
  }

  private async _executeThinkingChain(
    input: string,
    framework: ThinkingFramework,
    context?: ThinkingContext,
    callbacks?: AnalysisCallbacks
  ): Promise<ThinkingChain> {
    const startTime = new Date();
    const steps: ThinkingStep[] = [];
    for (const stepConfig of framework.steps) {
      const step = await this._executeThinkingStep(stepConfig, input, steps, context);
      steps.push(step);
      callbacks?.onStepComplete?.(step);
    }
    const endTime = new Date();
    return {
      id: uuidv4(),
      problem: input,
      steps,
      conclusion: steps.find((s) => s.type === "conclusion")?.output || "Analysis complete.",
      overallConfidence: this._calculateOverallConfidence(steps),
      metadata: {
        startTime: startTime.toISOString(),
        endTime: endTime.toISOString(),
        duration: endTime.getTime() - startTime.getTime(),
        totalSteps: steps.length,
      },
    };
  }

  private async _executeThinkingStep(
    config: StepConfig,
    originalInput: string,
    previousSteps: ThinkingStep[],
    context?: ThinkingContext
  ): Promise<ThinkingStep> {
    const inputContext = this._formatPreviousSteps(previousSteps);
    const result = await this._applyThinkingOperation(config, originalInput, inputContext, context);
    return {
      id: uuidv4(),
      type: config.type,
      description: config.description,
      input: inputContext,
      output: result.output,
      reasoning: result.reasoning,
      confidence: result.confidence,
      timestamp: new Date().toISOString(),
    };
  }

  private async _applyThinkingOperation(config: StepConfig, originalInput: string, inputContext: string, context?: ThinkingContext): Promise<LlmStepResponse> {
    const systemPrompt =
      "You are an expert AI reasoner executing one step in a larger plan. You have access to a web search tool. Use it if you need external information to fulfill your objective. Think step-by-step and be rigorous in your final output.";

    const responseSchema = {
      type: "object",
      properties: { output: { type: "string" }, reasoning: { type: "string" }, confidence: { type: "number", minimum: 0.0, maximum: 1.0 } },
      required: ["output", "reasoning", "confidence"],
    };

    const contextInfo = context?.objectives ? "\nOBJECTIVES TO CONSIDER:\n" + context.objectives.map((obj) => "- " + obj).join("\n") : "";

    const userPrompt = `Original Problem: "${originalInput}"

PREVIOUS STEPS & CONTEXT:
${inputContext}${contextInfo}

YOUR CURRENT TASK: ${config.type.toUpperCase()}
Objective: "${config.description}"

Execute this task now. If you lack the necessary information, use the 'search_the_web' tool. Provide your final answer by calling the 'execute_thinking_step' tool.`;

    return this._llmClient.generateStructuredResponse<LlmStepResponse>(systemPrompt, userPrompt, responseSchema, "execute_thinking_step");
  }

  private async _extractKeyInsights(chain: ThinkingChain): Promise<string[]> {
    const systemPrompt =
      "You are an AI expert in executive communication. Your purpose is to read a detailed analytical report and distill it into its most critical, high-impact insights. An insight is a strategically significant finding, not a simple summary.";
    const userPrompt = `Review the following reasoning chain and extract the 3-5 most critical insights.
Problem: "${chain.problem}"
---
${this._formatPreviousSteps(chain.steps)}
---`;

    const insightSchema = { type: "object", properties: { insights: { type: "array", items: { type: "string" } } }, required: ["insights"] };

    try {
      const result = await this._llmClient.generateStructuredResponse<{ insights: string[] }>(systemPrompt, userPrompt, insightSchema, "extract_insights");
      return result.insights || [];
    } catch (error) {
      console.error("LLM-powered insight extraction failed. Using fallback.", error);
      return chain.steps
        .filter((s) => s.confidence > 0.8)
        .map((s) => s.output.split(".")[0])
        .slice(0, 3);
    }
  }

  private _formatPreviousSteps(steps: ThinkingStep[]): string {
    if (steps.length === 0) return "No previous steps. This is the first step.";
    const fullText = steps.map((s) => `Step: ${s.type.toUpperCase()}\nObjective: ${s.description}\nOutput: ${s.output}`).join("\n\n---\n\n");
    if (fullText.length <= this._contextCharLimit) return fullText;
    const firstStep = steps[0];
    const lastSteps = steps.slice(-2);
    return (
      `[First Step]:\n${firstStep.output}\n\n---\n...[${steps.length - 3} intermediate steps summarized]...\n---\n\n` +
      lastSteps.map((s) => `[Recent Step: ${s.type.toUpperCase()}]\n${s.output}`).join("\n\n---\n\n")
    );
  }

  private _calculateOverallConfidence(steps: ThinkingStep[]): number {
    if (steps.length === 0) return 0;
    const weights: Record<ThinkingStepType, number> = { analysis: 1.2, decomposition: 0.9, exploration: 1.1, synthesis: 1.0, evaluation: 1.2, conclusion: 1.5 };
    const confidences = steps.map((s) => s.confidence);
    const stepWeights = steps.map((s) => weights[s.type]);
    const totalWeight = stepWeights.reduce((a, b) => a + b, 0);

    const weightedMean = steps.reduce((acc, step, i) => acc + step.confidence * stepWeights[i], 0) / totalWeight;
    const variance = steps.reduce((acc, step, i) => acc + Math.pow(step.confidence - weightedMean, 2) * stepWeights[i], 0) / totalWeight;
    const stdDev = Math.sqrt(variance);
    const minConfidence = Math.min(...confidences);

    const minConfidencePenalty = minConfidence < 0.6 ? 0.75 : 1.0;
    const consistencyFactor = 1 - stdDev * 1.5;
    const overallConfidence = weightedMean * consistencyFactor * minConfidencePenalty;

    return Math.max(0, Math.min(1, Math.round(overallConfidence * 100) / 100));
  }

  private _generateSummary(chain: ThinkingChain): string {
    return `Dynamic analysis of "${chain.problem}" completed in ${chain.metadata.duration}ms using a custom ${chain.metadata.totalSteps}-step plan, including internet exploration. The overall confidence in the conclusion is ${Math.round(chain.overallConfidence * 100)}%.`;
  }

  private _generateRecommendations(chain: ThinkingChain): string[] {
    const conclusionStep = chain.steps.find((s) => s.type === "conclusion");
    return conclusionStep ? [conclusionStep.output] : ["Review the full thinking chain for detailed action items."];
  }
}
