from core.qt_threading.headers.MessageBase import MessageBase


class PictureResponse(MessageBase):
    def __init__(self,  picture, vertices, source=None, destination=None):
        super().__init__()
        self.picture = picture
        self.vertices = vertices
        self.source = source
        self.destination = destination
