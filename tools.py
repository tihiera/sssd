from google.genai import types

def submit_scores_tool() -> types.Tool:
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="submit_scores",
                description="Return standardized worth sub-scores for a profile.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "human":        types.Schema(type=types.Type.INTEGER),
                        "social":       types.Schema(type=types.Type.INTEGER),
                        "network":      types.Schema(type=types.Type.INTEGER),
                        "contribution": types.Schema(type=types.Type.INTEGER),
                        "confidence":   types.Schema(type=types.Type.INTEGER),
                        "reasons":      types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.STRING),
                        ),
                    },
                    required=["human", "social", "network", "contribution", "confidence"],
                ),
            )
        ]
    )

SYSTEM_INSTRUCTION = (
    "You are WorthScorer. Read the user context and return strict scores via "
    "the function `submit_scores`. Scores are integers 1-10 for: human, social, network, contribution. "
    "Add confidence (0-100) and up to 4 short reasons. No extra text outside the function call."
)
