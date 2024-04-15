import xlrd
import numpy as np
import math, random

class WorkBook:
    def __init__(self):
        self.workbook = None
        self.worksheet = None
        self.info_base = {"Название трубопровода": None,
                     "Длина инспектируемого участка трубопровода": None,
                     "Диаметр трубопровода": None,
                     "Диапазон толщины стенки": None,
                     "Превалирующий тип сварного шва": None,
                     "Мин. радиус изгиба колена": None,
                     "Марка стали": None,
                     "Условия во время нормального режима": 
                        {"•Давление": None,
                         "•Температура": None,
                         "•Среда": None,
                         "•Скорость потока среды": None},
                     "Условия во время инспекции:":
                        {"•Давление": None,
                         "•Температура": None,
                         "•Контактная среда": None,
                         "•Скорость звука в контактной среде": None,
                         "•Затухание в контактной среде": None,
                         "•Скорость инспекции (средняя)": None}}
        self.db = []

    def process_the_db(self):
        ALPHABET = "ANCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.db = []
        for j in range(4, 21):
            start = list(map(int, self.worksheet.cell_value(0, j).split(";")))
            end = list(map(int, self.worksheet.cell_value(1, j).split(";")))
            db_add = []
            for x in range(start[1], end[1]):
                try:
                    inside = self.worksheet.cell_value(x, start[0] - 1)
                    db_add.append(inside)
                except:
                    db_add.append("")
            self.db.append(db_add)


    def open(self, path, worksheet=0):
        self.workbook = xlrd.open_workbook(path)
        self.worksheet = self.workbook.sheet_by_index(worksheet)

    def get_info_string_1(self):
        info = "Условия во время нормального режима"
        return "\nУсловия во время нормального режима:" + "\n•Давление: " + self.info_base[info]["•Давление"] \
                        + "\n•Температура: " + self.info_base[info]["•Температура"] \
                        + "\n•Среда: " + self.info_base[info]["•Среда"] \
                        + "\n•Скорость потока среды: " + self.info_base[info]["•Скорость потока среды"]

    def get_info_string_2(self):
        info = "Условия во время инспекции:"
        return "\nУсловия во время инспекции:" + "\n•Давление: " + self.info_base[info]["•Давление"] \
                        + "\n•Температура: " + self.info_base[info]["•Температура"] \
                        + "\n•Контактная Среда: " + self.info_base[info]["•Контактная Среда"] \
                        + "\n•Скорость звука в контактной среде: " + self.info_base[info]["•Скорость звука в контактной среде"] \
                        + "\n•Затухание в контактной среде: " + self.info_base[info]["•Затухание в контактной среде"] \
                        + "\n•Скорость инспекции (средняя): " + self.info_base[info]["•Скорость инспекции (средняя)"]

    def get_info(self):  
        info_index = 0
        for info in self.info_base:
            if info == "Условия во время нормального режима":
                self.info_base[info]["•Давление"] = self.worksheet.cell_value(info_index, 0)
                self.info_base[info]["•Температура"] = self.worksheet.cell_value(info_index + 1, 0)
                self.info_base[info]["•Среда"] = self.worksheet.cell_value(info_index + 2, 0)
                self.info_base[info]["•Скорость потока среды"] = self.worksheet.cell_value(info_index + 3, 0)
                info_index += 4
            elif info == "Условия во время инспекции:":
                self.info_base[info]["•Давление"] = self.worksheet.cell_value(info_index, 0)
                self.info_base[info]["•Температура"] = self.worksheet.cell_value(info_index + 1, 0)
                self.info_base[info]["•Контактная Среда"] = self.worksheet.cell_value(info_index + 2, 0)
                self.info_base[info]["•Скорость звука в контактной среде"] = self.worksheet.cell_value(info_index + 3, 0)
                self.info_base[info]["•Затухание в контактной среде"] = self.worksheet.cell_value(info_index + 4, 0)
                self.info_base[info]["•Скорость инспекции (средняя)"] = self.worksheet.cell_value(info_index + 5, 0)
                info_index += 6
            else:
                self.info_base[info] = self.worksheet.cell_value(info_index, 0)
                info_index += 1

def generate_black_image_2(width, height):
    return np.zeros((width, height))

def generate_color_spectrum_image_2(width, height, tube_length, tube_width, wall, tube_depth_percent, spot_angle, spot_start_x, diametr):
    map = np.zeros((width, height))
    spot_start_y = round(spot_angle * math.pi * diametr / 360)
    k_max = ((tube_length//2) * (tube_length//2) + (tube_width//2) * (tube_width//2)) ** (1/2)

    center_x = spot_start_x + tube_length//2
    center_y = spot_start_y + tube_width//2

    for x in range(spot_start_x, spot_start_x + tube_length - 1):
        for y in range(spot_start_y, spot_start_y + tube_width - 1):
            k_it = (4 * (center_x - x) * (center_x - x) + 4 * (center_y - y) * (center_y - y)) ** (1/2)
            if (k_max - k_it) < 0: continue
            k = (k_max - k_it) * random.randint(0, 10) / 10
            try:
                map[y][x] = (wall * tube_depth_percent) * k
            except:
                pass
    
    return map
