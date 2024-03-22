# Widget and link recognition library for Orange

This library, given an input image to check, aims at identifying and locating both widgets and links between them in screenshots of the Orange Data Mining app workplace. 

## Installation

1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.

## Usage

To use this project, you need to do the following:
1. Import all the functions contained in the library 'orangescreenshots.py'. Importing the library will create a subdirectory inside the working directory called 'widgets', containing the widget images which are downloaded from the website 'https://orangedatamining.com/widget-catalog/'. If the subdirectory already exists it is checked and if it is not updated, the new widgets' icons will be downloaded.
2. Call the function 'widgets_from_image('img_name.png')'. The output will be a list of the widgets present in the image 'img_name.png'
3. Call the function 'widget_pairs_from_image('img_name.png')'. The output will be a list of tuples containing the widget pairs, with the first element in each tuple being the link origin and the second one being the link destination.

## Contributing

Contributions are welcome! Please submit bug reports or feature requests through the issue tracker.


## Credits

- Alessandro Tiveron
- Blaz Zupan
