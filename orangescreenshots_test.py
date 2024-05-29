import unittest
from orangescreenshots import *


class WidgetsFromImageTest(unittest.TestCase):
    def test_widgets_from_image(self):
        yaml_file_widgets = get_filenames('data/widget_test/', '.yaml')
        remember_test = True
        for yaml_name in yaml_file_widgets:
            is_good = True
            general_name = yaml_name.split('/')[-1].removesuffix('.yaml')
            print('\nTesting', general_name)
            with open(yaml_name, 'r') as file:
                widget_list = yaml.safe_load(file)
            screenshot_name = 'data/screenshots/' + general_name
            try:
                try:
                    widget_list_detected = widgets_from_image(screenshot_name + '.png')
                except FileNotFoundError:
                    widget_list_detected = widgets_from_image(screenshot_name + '.jpg')
                if widget_list_detected is None and widget_list != 'None':
                    remember_test = False
                    is_good = False
                    print('The image', general_name, 'has no widgets detected in it but it should have some')
                elif widget_list_detected is not None:
                    sum_widgets = 0
                    for i in range(len(widget_list)):
                        sum_widgets = sum_widgets + widget_list[i][1]
                        check_presence = widget_list_detected.count(tuple(widget_list[i][0])) == widget_list[i][1]
                        if not check_presence:
                            print('The widget', tuple(widget_list[i][0]), 'was not detected the right amount of times '
                                                                          'in the image', general_name)
                            remember_test = False
                            is_good = False
                    if len(widget_list_detected) > sum_widgets:
                        print('There are more widgets detected than expected in the image', general_name)
                        print('There are', len(widget_list_detected), 'detected widgets, but there should be', sum_widgets)
                        remember_test = False
                        is_good = False
                if is_good:
                    print('The widgets in the image', general_name, 'are detected correctly')
            except FileNotFoundError:
                print('It appears the file', general_name, 'has a yaml file, but no screenshot')
        self.assertTrue(remember_test)

    def test_widget_pairs_from_image(self):
        remember_test = True
        yaml_file_pairs = get_filenames('data/link_test/', '.yaml')
        for yaml_name in yaml_file_pairs:
            is_good = True
            general_name = yaml_name.split('/')[-1].removesuffix('.yaml')
            print('\nTesting', general_name)
            with open(yaml_name, 'r') as file:
                widget_pairs = yaml.safe_load(file)
            screenshot_name = 'data/screenshots/' + general_name
            try:
                try:
                    widget_pair_detected = extract_workflow_from_image(screenshot_name + '.png', show_process=False)
                except FileNotFoundError:
                    widget_pair_detected = extract_workflow_from_image(screenshot_name + '.jpg', show_process=False)
                if widget_pair_detected is None and widget_pairs != 'None':
                    remember_test = False
                    is_good = False
                    print('The image', general_name, 'has no widgets detected in it but it should have some')
                elif widget_pair_detected is not None:
                    sum_links = 0
                    for i in range(len(widget_pairs)):
                        sum_links = sum_links + widget_pairs[i][1]
                        check_presence = widget_pair_detected.count(tuple([tuple(widget_pairs[i][0][0]), tuple(
                                                                    widget_pairs[i][0][1])])) == widget_pairs[i][1]
                        if not check_presence:
                            print('The link', tuple(widget_pairs[i][0]), 'was not detected the right amount of times '
                                                                           'in the image', general_name)
                            print()
                            remember_test = False
                            is_good = False
                    if len(widget_pair_detected) > sum_links:
                        print('There are more links detected than expected in the image', general_name)
                        print('There are', len(widget_pair_detected), 'detected links, but there should be', sum_links)
                        remember_test = False
                        is_good = False
                if is_good:
                    print('The links in the image', general_name, 'are detected correctly')
            except FileNotFoundError:
                print('It appears the file', general_name, 'has a yaml file, but no screenshot')
        self.assertTrue(remember_test)


if __name__ == '__main__':
    unittest.main()
