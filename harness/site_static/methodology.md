# Methodology

## Puzzles

Problems come from [Possibility Storm](https://www.patreon.com/mtgpuzzles). We transcribe the original image with a combination of models and cross-verified manually. Official solutions are taken from patreon and used for grading. To prevent solution leakage, I am only publishing details for a subset of problems.  

In private eval I am including puzzles from the most recent sets which are definitely outside of any model's training cutoff and includes newly defined rules interactions.

## Eval

Models get a stock system prompt describing general puzzle assumptions (win this turn, assum no extra cards, etc), the transcribed puzzle, and must return detailed steps in `<solution>` tags, accounting for all opponent decisions. All cards in the puzzle have full oracle text included.

To make sure models have full information available, we provide the full comprehensive rules. This is over 200k tokens, so we give the model ability to `grep` and `read` a local copy. (A second mode, full rules in context, pastes the entire rules document into the system prompt. I found it did not have a meaningful effect on model performance and costs significantly more, as frontier models mostly memorize the rules.)

Full rollout transcripts are provided for a subset of problems, see problems detail page.

All models are with thinking effort high unless specified.

## Grading

A separate judge model (currently GPT-5.6 Sol, high reasoning) sees the puzzle, the official solution, and the model's `<solution>` text, and grades based on correctness. I found the judge model does not matter too much after a baseline intelligence.