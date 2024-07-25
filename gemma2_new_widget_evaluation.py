import orangescreenshots as oss
import yaml
import ollama

n_correct = 0
ignored = 0
count = 0
filenames = oss._get_filenames('data/workflows/evaluation/new-widgets')
with open('data/workflows/evaluation/new-widgets/new-widget-evaluation.yaml', 'r') as file:
    workflows_info = yaml.safe_load(file.read().replace('\t', '  '))
for name in filenames:
    try:
        workflow = oss.Workflow(name)
    except ValueError:
        ignored += 1
        continue
    possible_widgets, _ = oss.find_similar_workflows(workflow, return_workflows=False)
    if isinstance(workflows_info[name.split('/')[-1]]['goal'], list):
        for goal in workflows_info[name.split('/')[-1]]['goal']:
            possible_widgets = oss._augment_widget_list(possible_widgets, present_widgets=workflow.get_widgets(), goal=goal)
            print1 = 'Evaluating the new_widget_prompt function for the workflow: ' + name + ' for goal ' + goal
            print(print1)
            target_widget = workflows_info[name.split('/')[-1]]['widget'][goal]
            prompt = oss.get_new_widget_prompt(workflow, goal, return_query=True)
            response = ollama.chat(
                model='gemma2:27b',
                messages=[{'role': 'user', 'content': prompt}],
                stream=False,
                options={'num_ctx': 8192, 'temperature': 0.3, 'top_p': 0.1}
            )
            response = response['message']['content'].split('yaml\n')[-1].split('`')[0]
            print(response)
            response = list(yaml.safe_load(response).keys())
            count += 1
            if isinstance(target_widget, list):
                found = 0
                present = False
                for i in target_widget:
                    if oss.Widget(i) in possible_widgets:
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
                    if oss.Widget(target_widget) not in possible_widgets:
                        print3 = 'The widget is not in the possible widgets'
                        print(print3)
            print()
    else:
        goal = workflows_info[name.split('/')[-1]]['goal']
        print('Evaluating the new_widget_prompt function for the workflow: ' + name + ' for goal ' + goal)
        possible_widgets = oss._augment_widget_list(possible_widgets, present_widgets=workflow.get_widgets(), goal=goal)
        target_widget = workflows_info[name.split('/')[-1]]['widget']
        prompt = oss.get_new_widget_prompt(workflow, goal, return_query=True)
        response = ollama.chat(
            model='gemma2:27b',
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            options={'num_ctx': 8192, 'temperature': 0.3, 'top_p': 0.1}
        )
        response = response['message']['content'].split('yaml\n')[-1].split('`')[0]
        print(response)
        response = list(yaml.safe_load(response).keys())
        count += 1
        if isinstance(target_widget, list):
            found = 0
            present = False
            for i in target_widget:
                if oss.Widget(i) in possible_widgets:
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
                if oss.Widget(target_widget) not in possible_widgets:
                    print3 = 'The widget is not in the possible widgets'
                    print(print3)
            else:
                n_correct += 1
            print('\n')
print4 = 'The new_widget_prompt function predicts ' + str(n_correct) + ' out of ' + str(count) + ' workflows correctly'
print5 = 'The accuracy of the new_widget_prompt function is ' + str(n_correct/count*100)[:4] + '%'
print(print4)
print(print5)
