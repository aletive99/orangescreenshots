from orangescreenshots import *
import ollama

filenames = get_filenames('data/workflows/evaluation/name-and-description')
with open('data/prompts/text-comparison-prompt.md', 'r') as file:
    query_start = file.read()
results = 0
with open('data/workflows/evaluation/name-and-description/description-evaluation.yaml', 'r') as file:
    widgets_info = yaml.safe_load(file.read().replace('\t', '  '))
for type_of_desc in ['concise', 'detailed']:
    for name in filenames:
        workflow = Workflow(name)
        prompt = get_workflow_description_prompt(workflow, type_of_desc)
        response = ollama.chat(
            model='gemma2:27b',
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            options={'num_ctx': 8192}
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
        response = ollama.chat(
            model='gemma2:27b',
            messages=[{'role': 'user', 'content': query}],
            stream=False,
            options={'num_ctx': 8192}
        )
        response = response['message']['content']
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



n_correct = 0
ignored = 0
count = 0
filenames = get_filenames('data/workflows/evaluation/new-widgets')
with open('data/workflows/evaluation/new-widgets/new-widget-evaluation.yaml', 'r') as file:
    workflows_info = yaml.safe_load(file.read().replace('\t', '  '))
for name in filenames:
    try:
        workflow = Workflow(name)
    except ValueError:
        ignored += 1
        continue
    possible_widgets, _ = find_similar_workflows(workflow, return_workflows=False)
    if isinstance(workflows_info[name.split('/')[-1]]['goal'], list):
        for goal in workflows_info[name.split('/')[-1]]['goal']:
            possible_widgets = augment_widget_list(possible_widgets, present_widgets=workflow.get_widgets(), goal=goal)
            print1 = 'Evaluating the new_widget_prompt function for the workflow: ' + name + ' for goal ' + goal
            print(print1)
            target_widget = workflows_info[name.split('/')[-1]]['widget'][goal]
            prompt = get_new_widget_prompt(workflow, goal, True)
            response = ollama.chat(
                model='gemma2:27b',
                messages=[{'role': 'user', 'content': prompt}],
                stream=False,
                options={'num_ctx': 8192}
            )
            response = list(yaml.safe_load(response['message']['content'].split('yaml\n')[-1].split('`')[0]).keys())
            count += 1
            if isinstance(target_widget, list):
                found = 0
                present = False
                for i in target_widget:
                    if Widget(i) in possible_widgets:
                        present = present or True
                    if i in response:
                        found += 1
                        n_correct += 1
                        break
                if found == 0:
                    print2 = 'The response does not contain the removed widget for the workflow: ' + name + ' for goal ' + goal
                    print(print2)
                    if not present:
                        print3 = 'The widget is not in the possible widgets'
                        print(print3)
            else:
                if target_widget in response:
                    n_correct += 1
                else:
                    print2 = 'The response does not contain the removed widget for the workflow: ' + name + ' for goal ' + goal
                    print(print2)
                    if Widget(target_widget) not in possible_widgets:
                        print3 = 'The widget is not in the possible widgets'
                        print(print3)
            print()
    else:
        print('Evaluating the new_widget_prompt function for the workflow: ' + name)
        possible_widgets = augment_widget_list(possible_widgets, present_widgets=workflow.get_widgets(), goal=workflows_info[name.split('/')[-1]]['goal'])
        target_widget = workflows_info[name.split('/')[-1]]['widget']

        prompt = get_new_widget_prompt(workflow, workflows_info[name.split('/')[-1]]['goal'], True)
        response = ollama.chat(
            model='gemma2:27b',
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            options={'num_ctx': 8192}
        )
        response = list(yaml.safe_load(response['message']['content'].split('yaml\n')[-1].split('`')[0]).keys())
        count += 1
        if isinstance(target_widget, list):
            found = 0
            present = False
            for i in target_widget:
                if Widget(i) in possible_widgets:
                    present = present or True
                if i in response:
                    found += 1
                    n_correct += 1
                    break
            if found == 0:
                print2 = 'The response does not contain the removed widget for the workflow: ' + name + ' for goal ' + goal
                print(print2)
                if not present:
                    print3 = 'The widget is not in the possible widgets'
                    print(print3)
            print('\n')
        else:
            if target_widget not in response:
                print2 = 'The response does not contain the removed widget for the workflow: ' + name + ' for goal ' + goal
                print(print2)
                if Widget(target_widget) not in possible_widgets:
                    print3 = 'The widget is not in the possible widgets'
                    print(print3)
            else:
                n_correct += 1
            print('\n')
print4 = 'The new_widget_prompt function predicts ' + str(n_correct) + ' out of ' + str(count) + ' workflows correctly'
print5 = 'The accuracy of the new_widget_prompt function is ' + str(n_correct/count*100)[:4] + '%'
print(print4)
print(print5)
