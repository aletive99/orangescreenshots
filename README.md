# Widget and link recognition library for Orange

This library, given an input image to check, aims at identifying and locating both widgets and links between them in screenshots of the Orange Data Mining app workplace. 

## Installation

1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.

## Usage

To use this project, you need to do the following:
1. Import all the functions contained in the library 'orangescreenshots.py'.
2. Call the function 'widgets_from_image('img_name.png')'. The output will be a list of the widgets present in the image 'img_name.png'
3. Call the function 'widget_pairs_from_image('img_name.png')'. The output will be a list of tuples containing the widget pairs, with the first element in each tuple being the link origin and the second one being the link destination.
4. To visualize where the widgets and the links are detected in the image the functions draw_positions('img_name.png'), draw_links('img_name.png') and draw_links_and_positions('img_name.png') can be called and they will show, respectively: an image with only the widgets and their names highlighted, an image with only the links between widgets highlighted and an image with both the links and the widgets highlighted.

## Callable functions
- size_identification: returns the size of the widget in the screenshot we gave as input
- get_sizes: returns the final size of the widget images based on the identified widget size in the screenshot
- get_filenames: returns all the file paths in the secified folder.
- widget_loading: returns an array containing all the widget images cropped and scaled
- get_widget_description: returns a dictionary containing the description of what each widget does
- screenshot_loading: returns the greyscale image of the screenshot
- is_there_widget_creation: returns an array containing information about the widgets present in the screenshot
- draw_locations: shows the identified positions of the widgets in the screenshot
- draw_links: shows the identified links in the screenshot
- draw_links_and_locations: shows both links and locations of the widgets in the screenshot
- widgets_from_images: returns a list of the widgets contained in the screenshot
- find_circle_intersection: returns the intersections between a binary image and a circle draws based on the inputs of the functions
- link_detection: returns an array containing info about the links between the widgets
- widget_pairs_from_image: returns a list of the links contained in the screenshot
- extract_workflows: crops the workflows from the original screenshot and saves it in a dedicated directory
- workflow_to_code: returns an array of numbers that describe the workflow
- update_image_list: parses the images present in the "orange-lecture-notes-web/public/chapters" directory and creates a yaml file containing information about the found chapters
- update_widget_list: parses the images present in the "orange-lecture-notes-web/public/chapters" directory and creates a yaml file containing information about the widgets present in each image
- update_image_links: parses the images present in the "orange-lecture-notes-web/public/chapters" directory and creates a yaml file containing information about the links present in each image

## Contributing

Contributions are welcome! Please submit bug reports or feature requests through the issue tracker.


## Credits

- Alessandro Tiveron
- Blaz Zupan