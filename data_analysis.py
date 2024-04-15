import yaml
from scipy.stats import hypergeom

# yaml files reading for results analysis
yaml_direct = 'image-analysis-results'

try:
    with open(yaml_direct+'/image-list.yaml', 'r') as file:
        chapters = yaml.safe_load(file)
except FileNotFoundError:
    print('There is no yaml file to read, the program will stop')
    exit(0)
try:
    with open(yaml_direct+'/image-widgets.yaml', 'r') as file:
        widgets = yaml.safe_load(file)
except FileNotFoundError:
    print('There is no yaml file to read, the program will stop')
    exit(0)
try:
    with open(yaml_direct+'/image-links.yaml', 'r') as file:
        links = yaml.safe_load(file)
except FileNotFoundError:
    print('There is no yaml file to read, the program will stop')
    exit(0)

# dictionaries creation
general_dict = dict()
chapter_dict = dict()
total_widgets = 0
for key in chapters.keys():
    chapter_dict[key] = {'n_widgets': 0}
    file_list = list([])
    keys_list = list([])
    for filename in chapters[key]['images']:
        file_list.append(key + '/' + filename)
        keys_list.append(key + '---' + filename)
    for image_key in keys_list:
        if widgets[image_key]['widgets'] is not None:
            for widget in widgets[image_key]['widgets']:
                widget_list = widget.split('/')[0]
                n_times = int(widget.split('/')[1])
                total_widgets += n_times
                if widget_list not in general_dict.keys():
                    general_dict[widget_list] = n_times
                else:
                    general_dict[widget_list] += n_times
                if widget_list not in chapter_dict[key].keys():
                    chapter_dict[key][widget_list] = n_times
                else:
                    chapter_dict[key][widget_list] += n_times
                chapter_dict[key]['n_widgets'] += n_times

general_link_dict = dict()
chapter_link_dict = dict()
total_links = 0
for key in chapters.keys():
    chapter_link_dict[key] = {'n_links': 0}
    file_list = list([])
    keys_list = list([])
    for filename in chapters[key]['images']:
        file_list.append(key + '/' + filename)
        keys_list.append(key + '---' + filename)
    for image_key in keys_list:
        if links[image_key]['links'] is not None:
            for link in links[image_key]['links']:
                link_list = link.split('/')[0]
                n_times = int(link.split('/')[1])
                total_links += n_times
                if link_list not in general_link_dict.keys():
                    general_link_dict[link_list] = n_times
                else:
                    general_link_dict[link_list] += n_times
                if link_list not in chapter_link_dict[key].keys():
                    chapter_link_dict[key][link_list] = n_times
                else:
                    chapter_link_dict[key][link_list] += n_times
                chapter_link_dict[key]['n_links'] += n_times

# p-values calculation
N = total_widgets
for key in chapter_dict:
    n = chapter_dict[key]['n_widgets']
    for widget in chapter_dict[key].keys():
        if widget != 'n_widgets':
            K = general_dict[widget]
            x = chapter_dict[key][widget]
            p_val = hypergeom.sf(x-1, N, K, n)
            chapter_dict[key][widget] = {'n_times': x, 'p-value': p_val}

N = total_links
for key in chapter_link_dict:
    n = chapter_link_dict[key]['n_links']
    for link in chapter_link_dict[key].keys():
        if link != 'n_links':
            K = general_link_dict[link]
            x = chapter_link_dict[key][link]
            p_val = hypergeom.sf(x-1, N, K, n)
            chapter_link_dict[key][link] = {'n_times': x, 'p-value': p_val}

f = open("image-analysis-results/widgets-analysis.txt", "w+")

# print results
enriched_widgets = []
for key in chapter_dict:
    f.write('\nChapter: ' + chapters[key]['document-title'] + ' (path: ' + key + ', images: ' + str(len(chapters[key]['images'])) +
            ', widgets: ' + str(chapter_dict[key]['n_widgets']) + ')\n')
    for widget in chapter_dict[key].keys():
        if widget != 'n_widgets' and chapter_dict[key][widget]['p-value'] < 0.1:
            enriched_widgets.append(widget)
            if chapter_dict[key][widget]['p-value'] < 0.001:
                f.write('    - ' + widget + ' (<0.001, k=' + str(chapter_dict[key][widget]['n_times']) + ')\n')
            else:
                f.write('    - ' + widget + ' (' + str(chapter_dict[key][widget]['p-value'])[0:5] + ', k=' +
                        str(chapter_dict[key][widget]['n_times']) + ')\n')
enriched_widgets = list(set(enriched_widgets))

enriched_links = []
for key in chapter_link_dict:
    f.write('\nChapter: ' + chapters[key]['document-title'] + ' (path: ' + key + ', images: ' + str(len(chapters[key]['images'])) +
          ', links: ' + str(chapter_link_dict[key]['n_links']) + ')\n')
    for link in chapter_link_dict[key].keys():
        if link != 'n_links' and chapter_link_dict[key][link]['p-value'] < 0.1:
            enriched_links.append(link)
            if chapter_link_dict[key][link]['p-value'] < 0.001:
                f.write('    - ' + link + ' (<0.001, k=' + str(chapter_link_dict[key][link]['n_times']) + ')\n')
            else:
                f.write('    - ' + link + ' (' + str(chapter_link_dict[key][link]['p-value'])[0:5] + ', k=' +
                        str(chapter_link_dict[key][link]['n_times']) + ')\n')
enriched_links = list(set(enriched_links))

f.close

with open(yaml_direct+'/widgets-analysis.yaml', 'w') as file:
    yaml.dump(enriched_widgets, file)

with open(yaml_direct+'/links-analysis.yaml', 'w') as file:
    yaml.dump(enriched_links, file)


#%%
