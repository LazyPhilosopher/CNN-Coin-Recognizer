

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
        self.pictures[picture_file] = {"vertices": []}
        return True

    def add_coin_param(self, param_name: str, value: int):
        self.coin_params[param_name] = value
        return True

    def add_vertices_to_picture(self, picture_file: str, vertices: list[list[int, int]]) -> bool:
        vertices_list: list[tuple[float, float]] = []

        for vertex in vertices:
            x: float = vertex[0]
            y: float = vertex[1]
            if not isinstance(x, float) or not isinstance(y, float):
                return False
            vertices_list.append((x, y))

        self.pictures[picture_file]["vertices"] += vertices_list
        return True

    def to_dict(self):
        return {
            "training_params": self.training_params,
            "coin_params": self.coin_params,
            "pictures": self.pictures
        }
