# Widget and link recognition library for Orange

This library, given an input image to check, aims at identifying and locating both widgets and links between them in screenshots of the Orange Data Mining app workplace. 

## Installation

1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.

## Usage

To use this project, you need to do the following:
1. Import all the functions contained in the library 'orangescreenshots.py'.
2. Call the function 'get_widgets()' in order to download the widget icons from the website 'https://orangedatamining.com/widget-catalog/'. This function will create a subfolder in the current directory containing all the widget image files. Once it is called, if the working directory stays the same it will not be needed to call this function again.
3. Call the function 'widgets_from_image('img_name.png')'. The output will be a list of the widgets present in the image 'img_name.png'
4. Call the function 'widget_pairs_from_image('img_name.png')'. The output will be a list of tuples containing the widget pairs, with the first element in each tuple being the link origin and the second one being the link destination.
5. To visualize where the widgets and the links are detected in the image the functions draw_positions('img_name.png'), draw_links('img_name.png') and draw_links_and_positions('img_name.png') can be called and they will show, respectively: an image with only the widgets and their names highlighted, an image with only the links between widgets highlighted and an image with both the links and the widgets highlighted.

## Contributing

Contributions are welcome! Please submit bug reports or feature requests through the issue tracker.


## Credits

- Alessandro Tiveron
- Blaz Zupan