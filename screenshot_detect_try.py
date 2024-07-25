from orangescreenshots import *

yaml_direct = 'image-analysis-results'
try:
    with open(yaml_direct+'/image-links.yaml', 'r') as file:
        links = yaml.safe_load(file)
except FileNotFoundError:
    print('There is no yaml file to read, the program will stop')
    exit(0)

list_of_links = []
for key in links:
    if links[key]['links'] is not None:
        for i in range(len(links[key]['links'])):
            link = links[key]['links'][i].split('/')[0]
            list_of_links.append(link)

a = np.unique(list_of_links)

