"""Prompt text of the evaluated configuration. A view is
    [system] [16 thumbnails with their time stamps, oldest first] [the latest frame, "(now)"] [question] [final-view sentence]
and the assistant turn is forced to start with "<answer>", so the answer follows directly (no reasoning)."""

SYSTEM = ("You are an expert multimodal assistant for streaming video QA. Answer the question "
          "using the video frames shown, which cover the stream up to the current moment.\n"
          "Place your final answer inside <answer> and </answer>.")

FINAL_VIEW = ("This is the final and richest view (the full timeline). <need_more_context/> is NO LONGER allowed. "
              "Give your best-effort final answer inside <answer> and </answer>.")

MC_INSTRUCTION = "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer></answer> tags."

OE_HEAD = "Here is the question:"
OE_INSTRUCTION = ("Give a COMPLETE, self-contained answer (a full sentence or two) inside <answer> and </answer>; "
                  "do not answer with a single word or refuse unless the question is truly unanswerable.")

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"


def question_block(question: str, options: list[str] | None) -> str:
    """The text after the last image. Options are full strings such as "A. a red car" (one per line)."""
    if options:
        return "\n" + question + "\n" + "\n".join(options) + "\n\n" + FINAL_VIEW + "\n" + MC_INSTRUCTION
    return "\n" + OE_HEAD + "\n" + question + "\n\n" + FINAL_VIEW + "\n" + OE_INSTRUCTION
