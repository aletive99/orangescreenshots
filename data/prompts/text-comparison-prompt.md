You are an expert in text analysis and comparison. I have two texts, and I need you to provide a similarity score from 0 to 10 based on their content. A score of 1 means the texts are completely different in content, while a score of 10 means they are identical in content. Please consider the following texts and only provide the similarity score. Your output should be of the type "SCORE", where the text in uppercase works as a place holder for your answer. Here are the texts.

Here is an example:

## Input:
'Text 1:
"The cat sat on the mat. It was a sunny day."

Text 2:
"A dog lay on the rug. The weather was bright and clear."

Score:'

## Output:
'7'


## Input:
'Text 1:
"The team lost, beacuse Brian played poorly."

Text 2:
"The team won, although Brian played poorly."

Score:'

## Output:
'5'


## Input:
'Text 1:
"The exam was so easy that every student got an A on it."

Text 2:
"The exam was extremely hard and no one passed it."

Score:'

## Output:
'0'

## Input:
'Text 1:
"Computers have made people's lives better by reducing a lot the time needed to perfrom many tasks."

Text 2:
"Thanks to the rise of computers, people have to spend less time doing many tasks, thus making their lives better."

Score:'

## Output:
'10'