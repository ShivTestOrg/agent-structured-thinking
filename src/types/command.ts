import { Type as T } from "@sinclair/typebox";
import { StaticDecode } from "@sinclair/typebox";

export const commandSchema = T.Object({
  name: T.String(),
  parameters: T.Object({
    callbackUrl: T.String({ format: "uri" }),
    jobId: T.String(),
    text: T.Optional(T.String()),
  }),
});

export type Command = StaticDecode<typeof commandSchema>;
