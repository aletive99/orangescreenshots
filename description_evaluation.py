import orangescreenshots as oss
import ollama
import yaml
from openai import OpenAI

filenames = oss._get_filenames('data/workflows/evaluation/name-and-description')
with open('data/prompts/text-comparison-prompt.md', 'r') as file:
    query_start = file.read()
with open('data/workflows/evaluation/name-and-description/description-evaluation.yaml', 'r') as file:
    widgets_info = yaml.safe_load(file.read().replace('\t', '  '))
for type_of_desc in ['concise', 'detailed']:
    results = 0
    for name in filenames:
        workflow = oss.Workflow(name)
        prompt = oss.get_workflow_description_prompt(workflow, type_of_desc)
        response = ollama.chat(
            model='gemma2:27b',
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            options={'num_ctx': 8192, 'temperature': 0.5, 'top_p': 0.5}
        )
        description_generated = response['message']['content']
        if type_of_desc == 'concise':
            description_actual = widgets_info[name.split('/')[-1]]['concise']
        else:
            description_actual = widgets_info[name.split('/')[-1]]['detailed']
        print1 = 'Evaluating the get_description function for the workflow: ' + name
        print(print1)
        query = query_start
        query += '\n\nText 1:\n' + description_generated + '\n\nText 2:\n' + description_actual + '\n\nScore:'
        api_key = ''
        client = OpenAI(api_key=api_key, organization='org-FvAFSFT8g0844DCWV1T2datD')
        response = client.chat.completions.create(model='gpt-3.5-turbo-0125',
                                                  messages=[
                                                      {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. "
                                                                                    "Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01"
                                                                                    "\nCurrent date: {CurrentDate}"},
                                                      {'role': 'user', 'content': query},
                                                  ],
                                                  temperature=0.5,
                                                  top_p=0.5)
        response = response.choices[0].message.content
        score = [int(i) for i in response if i.isdigit()]
        if len(score) == 2:
            score = 10
        else:
            score = score[0]
        print2 = 'The score for the get_description function is ' + str(score) + ' out of 10\n\n'
        print(print2)
        results += score
    print3 = 'The get_description function got a score of ' + str(results) + ' %' + ' for description type' + type_of_desc + '\n--------------------\n\n'
    print(print3)