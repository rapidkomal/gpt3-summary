import openai
import os
from nltk.tokenize import  word_tokenize
import pandas as pd

# Set up OpenAI API credentials

"""
Account creation
To use OpenAI through API, you must create a free account and generate keys. 
Fortunately, it is pretty straightforward.

1. Sign up here https://beta.openai.com/signup. 

2. Now, visit your OpenAI key page https://beta.openai.com/account/api-keys or click the menu item "View API keys"

"""

openai.api_key = os.getenv("OPENAI_API_KEY")

contents = pd.read_csv("./csv_folder/63edb7ffc28d09a86a541a82.csv")

cl_sent = []
sent = ''
for i in range(len(contents)):
    if i==10:
        break
    if len(word_tokenize(contents['message'][i].replace('.','').replace(',','').replace('?',''))) > 3:
        sent += contents['message'][i]


    
# Define the input text
input_text = sent #"Insert your input text here"

"""

Next, you use the openai.Completion.create() method to generate a summary of 
the input text using the davinci language model, which is one of OpenAI's most powerful models. 
You set the max_tokens parameter to 60, which controls the length of the summary, and the temperature parameter to 0.5,
which controls the "creativity" of the model (higher values result in more diverse output).

"""
# Generate a summary of the input text
output = openai.Completion.create(
    model="text-davinci-003",
    prompt="please summarize the attached meeting conversation:\n{}".format(input_text.strip()),
    temperature=0.1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# Extract the summary text from the API response
summary = output.choices[0].text.strip()

# Print the summary
print(summary)
