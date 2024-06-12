# Orangescreenshot library
This library is created to help users understand and learn to use [Orange Data Mining] faster. Given a workflow screenshot it allows users to get a description of what action the workflow performs and how it achieves that, it can prompt the user with suggestions of widgets that might be added to the workflow, or it can simply find a fitting name for the workflow.

[Orange Data Mining]: https://orange.biolab.si/


## Installing

1. Navigate with your computer's terminal to the folder you want this library to be downloaded in.
2. Clone this repository by typing `git clone https://github.com/aletive99/orangescreenshots.git`.
3. Install dependencies using `pip install -r requirements.txt`.


## Usage

In order to use the library it just needs to be imported inside Python. During the first run of this library a setup function will start to download the widget images from the [catalog] and will take around two minutes.

To wait as little time as possible when the screenshots are processed please only provide the portion of your screen that includes the workflow.

[catalog]: https://orangedatamining.com//widget-catalog/


### User callable functions
Following are the functions callable by the user:

#### draw_locations: 
- Description: Plots the screenshot with positions of the widgets and their name highlighted in red. Then press any button to close the image.
- Inputs: img_name: string with the path to the screenshot
(Internal input variable: return_img=False: bool)
- Example run: `draw_locations('data/screenshots/workflow.png')` will create:
![image](https://github.com/aletive99/orangescreenshots/blob/main/data/screenshots/example-function-usages/draw_locations.png)

#### draw_links: 
- Description: Plots the screenshot with the identified links highlighted in red. Then press any button to close the image.
- Inputs: img_name: string with the path to the screenshot
- Example run: `draw_links('data/screenshots/workflow.png')` will create:
![image](https://github.com/aletive99/orangescreenshots/blob/main/data/screenshots/example-function-usages/draw_links.png)

#### draw_links_and_locations: 
- Description: Plots the screenshot with both links and names and locations of the widgets highlighted in red. Then press any button to close the image.
- Inputs: img_name: string with the path to the screenshot
- Example run: `draw_links_and_locations('data/screenshots/workflow.png')` will create:
![image](https://github.com/aletive99/orangescreenshots/blob/main/data/screenshots/example-function-usages/draw_links_and_locations.png)

#### widgets_from_image: 
- Description: Returns a list of the widgets contained in the screenshot
- Inputs: img_name: string with the path to the screenshot
(Internal input variable: return_list=True: bool)
- Example run:<br>
`widgets_from_image('data/screenshots/workflow.png')` will output
`[('Text Mining', 'Concordance'),
 ('Text Mining', 'Word Cloud'),
 ('Text Mining', 'Corpus'),
 ('Text Mining', 'Preprocess Text')]`

#### extract_workflow_from_image
- Description: Returns a list of the links contained in the screenshot written as tuple of tuples: tuple(outputting widget, receiving widget) where both widgets are of the type tuple(module, name). If show_process is set to True the function will also plot the `link_detection` algorithm process when the links are multiple.
- Inputs: img_name: string with the path to the screenshot
show_process=False: bool
- Example run:`extract_workflow_from_image('data/screenshots/workflow.png')` will output
`[(('Text Mining', 'Word Cloud'), ('Text Mining', 'Concordance')),
 (('Text Mining', 'Corpus'), ('Text Mining', 'Concordance')),
 (('Text Mining', 'Preprocess Text'), ('Text Mining', 'Word Cloud')),
 (('Text Mining', 'Corpus'), ('Text Mining', 'Preprocess Text'))]`

#### get_workflow_description_prompt
- Description: Will create and print a ChatGPT prompt to use to generate a description of the workflow given in the input screenshot. If the use_api value is set to True, then the function will automatically ask the OpenAI API for an answer to the prompt and will print it after the given prompt. 
- Inputs: img_name: string with the path to the screenshot
use_api=False: bool
- Example runs: <br>
`get_workflow_description_prompt('path/to/screenshot/workflow.png')` will print the prompt<br>
`get_workflow_description_prompt('path/to/screenshot/workflow.png', True)` will print the prompt as well as the ChatGPT response


#### get_workflow_name_prompt
- Description: Will create and print a ChatGPT prompt to use to generate a plausible name of the workflow given in the input screenshot. If the use_api value is set to True, then the function will automatically ask the OpenAI API for an answer to the prompt and will print it after the given prompt. 
- Inputs: img_name: string with the path to the screenshot
use_api=False: bool
- Example runs: <br>
`get_workflow_name_prompt('path/to/screenshot/workflow.png')` will print the prompt<br>
`get_workflow_name_prompt('path/to/screenshot/workflow.png', True)` will print the prompt as well as the ChatGPT response

#### get_new_widget_prompt
- Description: Will create and print a ChatGPT prompt to use to get 3 possible widgets that could make sense to be added to the workflow given in the input screenshot. If the use_api value is set to True, then the function will automatically ask the OpenAI API for an answer to the prompt and will print it after the given prompt. 
- Inputs: img_name: string with the path to the screenshot
use_api=False: bool
- Example run:  <br>
`get_new_widget_prompt('path/to/screenshot/workflow.png')` will print the prompt<br>
`get_new_widget_prompt('path/to/screenshot/workflow.png', True)` will print the prompt as well as the ChatGPT response


### Classes

#### Widget:
- Initialization: requires two inputs: module and name in order to be created.<br>
Example: `wid1 = Widget('Data', 'File')`
- String: `str(wid1)` will output the string `'Data/File'`
- `wid1.get(description)` will output a list containing: description of the widget, list of possible inputs and list of possible outputs.

#### Workflow:
- Initialization: requires a list of tuples of tuples, a list of tuples of Widgets or a string as input<br>
Examples: <br>
`work1 = Workflow('path/to/folder/screenshot-name.png`) will automatically call the extract workflow from image function to create the object<br>
`work1 = Workflow([(wid1, wid2), (wid2, wid3), (wid3, wid4)])` where `wid` are Widget objects<br>
`work1 = Workflow([(('Data','File'),('Unsupervised','Distances')),(('Unsupervised','Distances'),('Unsupervised', 'Hierarchical Clustering')), (('Unsupervised', 'Hierarchical Clustering'), ('Unsupervised', 'Dendrogram'))])`<br>
- String: `str(work1)` will output a string containing a characterization of the links in topological order, for example: <br>
```
work1 = Workflow([(('Unsupervised','Distances'),('Unsupervised', 'Hierarchical Clustering')), (('Unsupervised', 'Hierarchical Clustering'), (('Data','File'),('Unsupervised','Distances')), ('Unsupervised', 'Dendrogram'))])
str(work1) = 'Data/File -> Unsupervised/Distances\nUnsupervised/Distances -> Unsupervised/Hierarchical Clustering\n'Unsupervised/Hierarchical Clustering -> Unsupervised/Dendrogram
```
- Length: `len(work1)` will output an integer saying how many links are in the workflow
- Get Item: `work1[i]` will output the i-th link in the workflow as a tuple of tuples
- `work1.get_widgets` will output a list of the Widget objects used in the workflow
- `work1.remove_widget(wid1)` will remove `wid1` from the workflow by removing its links. If `wid1` is not specified then the function will remove randomly one of the last 2 widgets in the workflow (chosen by topological order) and the respective links.


### Internal functions
Following are the internal functions used to perform the operations used by the user callable functions or used to create yaml files that allow these functions to work. We suggest no to use these functions.

| Function name             	| Inputs      		  							| Description                                                                 									|
| ----------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| download_widgets           	| None											| Downloads the widget images from the widget catalog                         									|
| size_identification           | img_name: str<br> show_circles=False: bool 		| Returns the size of the widget in the screenshot received as input                          					|
| get_sizes            			| img_name: str            						| Returns the final size of the widget images based on the identified widget size in the screenshot 			|
| get_filenames                 | direct: str<br>ext='Image': str     			| Returns all the file paths in the specified folder                                                            |
| widget_loading        		| img_names_tgt: list of str<br>img_name: str     | Returns a 3D-array containing all the widget images cropped and scaled 										|
| get_widget_description        | None                							| Creates a yaml file containing a dictionary with the description of the widgets' functions, inputs and outputs|
| screenshot_loading            | img_name: str             					| Returns the greyscale image of the screenshot                                     							|
| is_there_widget_creation   	| img_name: str<br>value_thresh=0.8: float		| Returns an array containing information about the widgets present in the screenshot         					|
| find_circle_intersection      | label_binary_image: np.array<br>center: tuple<br>radius_size: int<br>prev_direction: float<br>connect_type=8: int | Returns the intersections, found closest direction and best intersection index between a binary image and a manually built circle|
| link_detection                | img_name: str<br>show_process=False, bool		| Returns an array containing info about the links between the widget											|
| update_image_list             | None											| Parses the images present in the "orange-lecture-notes-web/public/chapters" directory and creates a yaml file containing information about the found chapters|
| update_widget_list            | None											| Parses the images present in the "orange-lecture-notes-web/public/chapters" directory and creates a yaml file containing information about the widgets present in each image|
| update_image_links            | None											| Parses the images present in the "orange-lecture-notes-web/public/chapters" directory and creates a yaml file containing information about the links present in each image|
| crop_workflows        		| directory_to_check='orange-lecture-notes-web/public/chapters': str<br>img_name=None: bool<br>no_yaml=False: bool | Crops the workflows from the original screenshot and saves it in a dedicated directory|
| workflow_to_code        		| workflow: Workflow<br>return_labels=False: bool<br>only_enriched=True: bool | Returns an array of numbers that characterizes the workflow 						|
| create_dataset 				| orange_dataset=True<br>min_thresh=3<br>only_enriched=True | Creates an excel file with the codes of the workflows in the documentation based on the inputs given	|
| get_example_workflows		   	| None											| Loads the sample workflows and returns their names, their links and their description							|
| find_closest_workflows	   	| workflow: Workflow<br>remove_widget=False: bool<br>k=10: int | Returns the widget that are not in the given workflow, but are present in the k-most similar ones	|


## Contributing

Contributions are welcome! Please submit bug reports or feature requests through the issue tracker.


## Credits

- Alessandro Tiveron
- Blaz Zupan
