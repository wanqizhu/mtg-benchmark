# Methodology

## Puzzles

Problems come from [Possibility Storm](https://www.patreon.com/mtgpuzzles) (public puzzle pages on [possibilitystorm.com](https://www.possibilitystorm.com/)). Each published problem page shows the original image, a structured transcription of the board state, and a link to the public source page. Transcriptions are done with a combination of models and cross-verified manually, though it's possible there are mistakes. Official solutions are taken from patreon and used for grading.

## How the agent sees a problem

Every attempt gets the same transcribed puzzle in a `<puzzle>` block and must return a numbered line in `<solution>` tags.

The two **versions** on this site differ only in how the Comprehensive Rules are provided:

- **grep rules**: the model can `grep` and `read` a local copy of the Comprehensive Rules. The full document is not in the prompt.
- **full rules in context**: the entire Comprehensive Rules text is included in the system prompt. There are no search tools. This needs a long-context model.

Puzzle conventions (what the image assumes about life totals, empty zones, blocking, and so on) are included in the system prompt in both modes.

## Grading

A separate judge model (currently GPT-5.6 Sol, high reasoning) sees the puzzle, the official solution, and the model's `<solution>` text. It does not see the solver's chain of thought or tool traces.

A solution **passes** if it is rules-legal and achieves the objective for possible opponent responses and blocks. It does not need to match the official line step for step. Alternate correct lines can still pass; the judge is asked to be extra careful when the line differs.