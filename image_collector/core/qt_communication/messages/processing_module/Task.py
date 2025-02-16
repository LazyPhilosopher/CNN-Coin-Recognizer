class Task:
    def __init__(self, func: callable, arg_dict: dict | None, pipe_dict: dict):
        self.func = func
        self.arg_dict = arg_dict
        self.pipe_dict: dict = pipe_dict
