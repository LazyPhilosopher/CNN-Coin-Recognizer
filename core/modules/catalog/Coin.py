from os import path


class Coin:
    year: str
    country: str
    training_params: dict
    coin_params: dict
    name: str = None
    pictures: dict[str, dict]

    def __init__(self, name: str, year: str = None, country: str = None):
        self.name = name
        self.year = year
        self.country = country
        self.training_params = {}
        self.pictures = {}
        self.coin_params = {}

    def add_training_param(self, param_name: str, value: int):
        self.training_params[param_name] = value
        return True

    def add_picture(self, picture_file: str) -> bool:
        self.pictures[picture_file] = {"contour": []}
        return True

    def add_picture_contour(self, picture_file: str, contour: list[tuple[int, int]]) -> bool:
        self.pictures[picture_file]["contour"] = contour
        return True

    def add_coin_param(self, param_name: str, value: int):
        self.coin_params[param_name] = value
        return True

    def add_contour_to_picture(self, picture_file: str, contour: list[list[int, int]]) -> bool:
        contour_pixel_list: list[tuple[float, float]] = []

        for pixel in contour:
            x: int = pixel[0]
            y: int = pixel[1]
            if not isinstance(x, int) or not isinstance(y, int):
                return False
            contour_pixel_list.append((x, y))

        self.pictures[picture_file]["contour"] += contour_pixel_list
        return True

    def to_dict(self):
        return {
            "training_params": self.training_params,
            "coin_params": self.coin_params,
            "pictures": self.pictures
        }

    def coin_dir_path(self) -> str:
        return path.join(self.year, self.country, self.name)
