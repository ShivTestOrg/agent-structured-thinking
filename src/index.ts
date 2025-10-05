import { Context } from "./types";
import { StructuredThinkingEngine, SearchClient, ThinkingStep, ChatbotClient } from "./structured-thinking";

export async function runPlugin(context: Context) {
  const { logger, eventName, command, env } = context;
  logger.info(`Event received: ${eventName}`);

  // Check for the specific 'analyze' command
  if (command?.name !== "analyze") {
    logger.info("No 'analyze' command found. Skipping.");
    return;
  }

  const { jobId, callbackUrl, text: prompt } = command.parameters;
  logger.info(`Command parameters: jobId=${jobId}, callbackUrl=${callbackUrl}, prompt length=${prompt?.length}, command=${JSON.stringify(command, null, 2)}`);
  const { OPENAI_API_KEY, TAVILY_API_KEY } = env;

  if (!jobId || !callbackUrl || !prompt) {
    logger.error("Missing required parameters: jobId, callbackUrl, or text.");
    return;
  }

  if (!OPENAI_API_KEY || !TAVILY_API_KEY) {
    const errorMessage = "Missing required environment variables: OPENAI_API_KEY and/or TAVILY_API_KEY must be configured.";
    logger.error(errorMessage);
    await updateCallbackUrl(callbackUrl, jobId, "failed", { error: errorMessage });
    return;
  }

  logger.info(`Recognized 'analyze' command for jobId: ${jobId}`);

  try {
    await updateCallbackUrl(callbackUrl, jobId, "started");
    const searchClient = new SearchClient({ apiKey: TAVILY_API_KEY });
    const llmClient = new ChatbotClient(
      {
        apiKey: OPENAI_API_KEY,
        model: "gpt-4o",
      },
      searchClient
    );
    const engine = new StructuredThinkingEngine(llmClient);
    async function onStepComplete(step: ThinkingStep) {
      logger.info(`[Job ${jobId}] Step '${step.type}' completed.`);
      logger.info(`Step details: ${JSON.stringify(step, null, 2)}`);
      await updateCallbackUrl(callbackUrl, jobId, "processing", {
        message: `Step '${step.type}' complete.`,
        step: {
          type: step.type,
          description: step.description,
          output: step.output,
        },
      });
    }

    logger.info(`[Job ${jobId}] Starting structured analysis...`);
    const analysisResult = await engine.analyzeStructured(prompt, {}, { onStepComplete });

    await updateCallbackUrl(callbackUrl, jobId, "completed", {
      message: "Analysis complete.",
      data: analysisResult,
    });
    logger.info(`Job ${jobId} completed successfully.`);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
    logger.error(`Error processing job ${jobId}: ${errorMessage}`);
    await updateCallbackUrl(callbackUrl, jobId, "failed", { error: errorMessage });
  }
}

async function updateCallbackUrl(
  callbackUrl: string,
  jobId: string,
  status: "started" | "processing" | "completed" | "failed",
  result?: Record<string, unknown>
) {
  const payload = { jobId, status, outputs: result || {} };
  try {
    const response = await fetch(callbackUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      console.error(`Callback to ${callbackUrl} failed with status: ${response.status}`);
    }
  } catch (error) {
    console.error(`Failed to send callback to ${callbackUrl}:`, error);
  }
}
