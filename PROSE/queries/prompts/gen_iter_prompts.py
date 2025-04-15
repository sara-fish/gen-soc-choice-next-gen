INITIAL_PROMPT_V1 = """
You will be given some information about {num_agents} users on a specific topic. Your goal is to write some text. After you write your text, the users will rate your text on the following scale:
{approval_levels_str}
Your goal is to have all users give your text a rating of at least {target_approval} (that is, "{target_approval_text}"). Additionally, it's important you obey the length limit -- your text must be at most {target_budget} words. 

****************
User information
****************
{agent_descriptions}

*******************
Output instructions
*******************

Remember your goal is to write a text using at most {target_budget} words that all users will give a rating of at least {target_approval} (that is, "{target_approval_text}") to. To do this follow these steps:

1. Read the user information carefully. Where do the users agree? Where do they disagree? Weigh how important each of the points are to the overall topic. Write your reasoning here in HTML tags <step1>...</step1>.

2. Considering your word limit of {target_budget} words, decide what points to include in your text and what ones to omit. Write your reasoning here in HTML tags <step2>...</step2>.

3. Finally, write your text. **To make sure it can be parsed, include your text in HTML tags <text>...</text>.** The word count will be determined by whatever is in these HTML tags, so follow this instruction carefully.
""".strip()

RESPONSE_PROMPT_V1 = """
(Attempt number {num_iterations}/{max_iterations}.) Recall that your task is to write a text using at most {target_budget} words that all users will give a rating of at least {target_approval} (that is, "{target_approval_text}") to. 

Last time, you wrote the following text:
{previous_text}

This text has {previous_text_target_budget} words, and your target word count was {target_budget}. 

This text was rated as follows by the users:
{agent_approvals}
Your goal was to have all users give your text a rating of at least {target_approval} (that is, "{target_approval_text}").

Now, you should modify your text to make it fit the specifications better (both on word count and on user ratings). To do this, follow these steps:

1. Read the user ratings carefully. For which users was the text not good enough? What modifications could you make to improve the text for these users, without sacrificing other users' approval? Write your reasoning here in HTML tags <step1>...</step1>.

2. If applicable, does the length of the text need to be modified to better fit the word limit? Write your reasoning here in HTML tags <step2>...</step2>.

3. Finally, write your new text. **To make sure it can be parsed, include your text in HTML tags <text>...</text>.** The word count will be determined by whatever is in these HTML tags, so follow this instruction carefully.
""".strip()


def get_prompts(prompt_type: str) -> tuple[str, str]:
    if prompt_type == "v1":
        return INITIAL_PROMPT_V1, RESPONSE_PROMPT_V1
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
