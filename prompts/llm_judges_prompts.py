from langchain_core.prompts import ChatPromptTemplate

cc_prompt = ChatPromptTemplate.from_messages([
    ("system", """Evaluate the Contextual Coherence between the generated response and the retrieved context.

Task: Assess the logical consistency and coherence of the response
with respect to the provided context. Ensure the response logically
follows from and does not contradict the context.

Scoring:
Score from 0 to 100, where 0 is completely incoherent or contradictory, and 100 is perfectly coherent and consistent.
If no response is generated, the score is 0.

Output:
Valid outpus are a integer from 0 to 100.
"Score: X" where X is the score is not a valid output.
only the integer score should be printed and nothing else.
"""),
    ("user", """Retrieved Context:
{context}

Generated Response:
{response}
""")
])

qr_prompt = ChatPromptTemplate.from_messages([
    ("system", """Evaluate the Question Relevance of the generated response to the user query.

Task:
Assess how well the response directly addresses the user's query.
Consider if the response answers the question asked.

Scoring:
Score from 0 to 100, where 0 is completely irrelevant and 100 is perfectly relevant and directly answers the query.
If no response is generated, the score is 0.

Output:
Valid outpus are a integer from 0 to 100.
"Score: X" where X is the score is not a valid output.
only the integer score should be printed and nothing else.
"""),
    ("user", """User Query:
{question}

Generated Response:
{response}
""")
])

id_prompt = ChatPromptTemplate.from_messages([
    ("system", """Evaluate the Information Density of the generated response.

Task:
Assess the balance of conciseness and informativeness in the response, considering both the context and the query.
The response should provide necessary information without being overly verbose or unnecessarily brief.

Scoring:
Score from 0 to 100, where 0 is either too verbose (contains excessive irrelevant detail) or uninformative (lacks necessary information), and 100 is optimally concise and informative for the query.
If no response is generated, the score is 0.

Output:
Valid outpus are a integer from 0 to 100.
"Score: X" where X is the score is not a valid output.
only the integer score should be printed and nothing else.
"""),
    ("user", """User Query:
{question}

Retrieved Context:
{context}

Generated Response:
{response}
""")
])

ac_prompt = ChatPromptTemplate.from_messages([
    ("system", """Evaluate the Answer Correctness of the generated response.

Task:
Assess the factual accuracy of the information presented in the response compared to the ground truth answer, considering the provided context.
Do not penalize differences in phrasing if the core factual meaning is preserved and accurate according to the ground truth.

Scoring:
Score from 0 to 100, where 0 is completely incorrect or contains significant factual errors, and 100 represents perfect factual accuracy (semantically equivalent to the ground truth).
If no response is generated, the score is 0.

Output:
Valid outpus are a integer from 0 to 100.
"Score: X" where X is the score is not a valid output.
only the integer score should be printed and nothing else.
"""),
    ("user", """Retrieved Context:
{context}

Generated Response:
{response}

Ground Truth Answer:
{ground_truth}
""")
])

ir_prompt = ChatPromptTemplate.from_messages([
    ("system", """Evaluate the Information Recall of the generated response.

Task:
Assess how much of the essential information present in the ground truth answer is captured in the generated response, considering the provided context.
Focus on whether key facts or points from the ground truth are included.

Scoring:
Score from 0 to 100, where 0 means no essential information from the ground truth is recalled, and 100 means all essential information is fully captured.
If no response is generated, the score is 0.

Output:
Valid outpus are a integer from 0 to 100.
"Score: X" where X is the score is not a valid output.
only the integer score should be printed and nothing else.
"""),
    ("user", """Retrieved Context:
{context}

Generated Response:
{response}

Ground Truth Answer:
{ground_truth}
""")
])
