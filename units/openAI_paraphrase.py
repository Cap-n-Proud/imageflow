import os
import re
import openai
# a message with a elephant in the background. a large brown giraffe with a message on its head in the green field. a red bird sitting on a branch with leaves in the jungle behind it. a young girl wearing glasses holding a pink bag in the woods. a young girl wearing glasses holding her mouth in the background. a young girl standing near a fence with hay in the green field behind it. a young person kissing a brown giraffe in a wooden fence. a young girl wearing glasses looking into a elephant in the mirror. a young girl wearing glasses standing near a wooden fence with her hand behind it. a smiling young girl holding her hand near a log in the background. a smiling young woman wearing glasses holding her hand in the background. a couple of brown sheep standing together with a pile of hay behind it. a little girl playing on a wooden bench in the woods with plants behind it. a little girl playing on a wooden bench with plants next to the path. a couple of little kids playing on a wooden path in the woods


# Set up OpenAI API credentials
openai.api_key = "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"

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="a message with a elephant in the background. a large brown giraffe with a message on its head in the green field. a red bird sitting on a branch with leaves in the jungle behind it. a young girl wearing glasses holding a pink bag in the woods. a young girl wearing glasses holding her mouth in the background. a young girl standing near a fence with hay in the green field behind it. a young person kissing a brown giraffe in a wooden fence. a young girl wearing glasses looking into a elephant in the mirror. a young girl wearing glasses standing near a wooden fence with her hand behind it. a smiling young girl holding her hand near a log in the background. a smiling young woman wearing glasses holding her hand in the background. a couple of brown sheep standing together with a pile of hay behind it. a little girl playing on a wooden bench in the woods with plants behind it. a little girl playing on a wooden bench with plants next to the path. a couple of little kids playing on a wooden path in the woods\nTl;dr: \nA young girl wearing glasses stands near a fence with hay behind it, while a smiling woman holds her hand in the background. A couple of brown sheep stand together with a pile of hay behind them, and a little girl plays on a wooden bench with plants nearby. In the jungle, a red bird sits on a branch with leaves and a large brown giraffe has a message on its head. \n Tl;dr: ",
    temperature=0.7,
    max_tokens=2500,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=1
)

print(response)
