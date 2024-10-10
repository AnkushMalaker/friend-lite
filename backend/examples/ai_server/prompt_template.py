qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
    "Query: {query}\n"
    "Answer: "
)

log_taker_prompt_tmpl_str = (
    "The transcript of a conversation or monologue is below.\n"
    "---------------------\n"
    "{transcript}\n"
    "---------------------\n"
    "Refactor the transcript into structured logs. Include a section for tracking of chores."
    "Mark chores as [x] if mentioned as completed in the conversation. "
    "The only known chores are: {chores}"
    "If no chores are mentioned, do not create a chores section."
    "So in total, you should have 3 sections in the output: "
    "1. '# Raw transcript' - after removing any filler words like 'um', 'uh', 'like' etc. "
    "2. An optional '# Tasks' section with a list of tasks or events to do. "
    "3. A '# Chores' section with a list of chores, marked [x] if mentioned as completed in the transcription, [ ] if not mentioned."
    "Format the output in markdown.\n"
    "Output: "
)
