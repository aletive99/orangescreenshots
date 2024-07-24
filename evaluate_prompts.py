from orangescreenshots import *

filenames = _get_filenames('data/workflows/evaluation/name-and-description')
with open('data/prompts/text-comparison-prompt.md', 'r') as file:
    query_start = file.read()
results = 0
with open('data/workflows/evaluation/name-and-description/description-evaluation.yaml', 'r') as file:
    widgets_info = yaml.safe_load(file)
for type_of_desc in ['concise', detailed]
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
        if concise_description:
            description_actual = widgets_info[name.split('/')[-1]]['concise']
        else:
            description_actual = widgets_info[name.split('/')[-1]]['detailed']
        print('Evaluating the get_description function for the workflow: ' + name)
        query = query_start
        query += '\n\nText 1:\n' + description_generated + '\n\nText 2:\n' + description_actual + '\n\nScore:'
        response = _get_response(query, 'gpt-3.5-turbo-0125')
        score = [int(i) for i in response if i.isdigit()]
        if len(score) == 2:
            score = 10
        else:
            score = score[0]
        print('The score for the get_description function is ' + str(score) + ' out of 10\n\n')
        results += score
    print('The get_description function got a score of ' + str(results) + ' %' + ' for description type' + type_of_desc + '\n--------------------\n\n')



n_correct = 0
ignored = 0
count = 0
filenames = _get_filenames('data/workflows/evaluation/new-widgets')
with open('data/workflows/evaluation/new-widgets/new-widget-evaluation.yaml', 'r') as file:
    workflows_info = yaml.safe_load(file)
for name in filenames:
    try:
        workflow = Workflow(name)
    except ValueError:
        ignored += 1
        continue
    possible_widgets, _ = find_similar_workflows(workflow, return_workflows=False)
    if isinstance(workflows_info[name.split('/')[-1]]['goal'], list):
        for goal in workflows_info[name.split('/')[-1]]['goal']:
            possible_widgets = _augment_widget_list(possible_widgets, present_widgets=workflow.get_widgets(), goal=goal)
            print('Evaluating the new_widget_prompt function for the workflow: ' + name + ' for goal ' + goal)
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
                    print('The response does not contain the removed widget for the workflow: ' + name + ' for goal ' + goal)
                    if not present:
                        print('The widget is not in the possible widgets')
                        if not check_response:
                            for i in possible_widgets:
                                print(str(i))
            else:
                if target_widget in response:
                    n_correct += 1
                else:
                    print('The response does not contain the removed widget for the workflow: ' + name + ' for goal ' + goal)
                    if Widget(target_widget) not in possible_widgets:
                        print('The widget is not in the possible widgets')
            print()
    else:
        print('Evaluating the new_widget_prompt function for the workflow: ' + name)
        possible_widgets = _augment_widget_list(possible_widgets, present_widgets=workflow.get_widgets(), goal=workflows_info[name.split('/')[-1]]['goal'])
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
                print('The response does not contain the removed widget for the workflow: ' + name)
                if not present:
                    print('The widget is not in the possible widgets')
                    if not check_response:
                        for i in possible_widgets:
                            print(str(i))
            print('\n')
        else:
            if target_widget not in response:
                print('The response does not contain the removed widget for the workflow: ' + name)
                if Widget(target_widget) not in possible_widgets:
                    print('The widget is not in the possible widgets')
            else:
                n_correct += 1
            print('\n')
print('The new_widget_prompt function predicts ' + str(n_correct) + ' out of ' + str(count) + ' workflows correctly')
print('The accuracy of the new_widget_prompt function is ' + str(n_correct/count*100)[:4] + '%')