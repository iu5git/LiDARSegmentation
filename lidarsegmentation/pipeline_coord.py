import os
from lidarsegmentation.coordinates import coordinates
from lidarsegmentation.merge_coordinates import merge_coordinates
from lidarsegmentation.clear_excess_stumps import clear_excess_stumps
from lidarsegmentation.settings.coord_settings import CS


if __name__ == "__main__" :
    cs = CS()
    yml_path = "settings\settings.yaml"
    cs.set(yml_path)
    coordinates(7000, cs)
    coordinates(5000, cs)
    coordinates(1000, cs)
    merge_coordinates(cs)
    clear_excess_stumps(cs)
