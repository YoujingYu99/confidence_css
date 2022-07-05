"""Extract categories from xml files.

This script finds the questions from json transcripts and gather all
information in a csv file.
"""
import xml.etree.ElementTree as ET
from ..data_preparation_utils import *

folder_number = 3


home_dir = os.path.join("/home", "yyu")
rss_folder_dir = os.path.join(
    home_dir, "data", "Spotify-Podcasts", "podcasts-no-audio-13GB", "show-rss",
)

xml_path = os.path.join(rss_folder_dir, "0", "0", "show_00grj9F8Ql4H67kY80YTBm.xml")


def get_show_category(xml_path):
    # # top folder
    # ogg_file_top = file_name.split("/")[-4]
    # # sub folder list
    # ogg_file_sub = file_name.split("/")[-3]
    # # show name list
    # show_file = file_name.split("/")[-2] + ".xml"
    # xml_path = os.path.join(rss_folder_dir, ogg_file_top, ogg_file_sub, show_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    print(
        root.find("./channel/{http://www.itunes.com/dtds/podcast-1.0.dtd}category").get(
            "text"
        )
    )

    # # print the text contained within first subtag of the 5th tag from the parent
    # print(root[5][0].text)


get_show_category(xml_path)
