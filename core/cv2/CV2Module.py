import cv2


class CV2Module:
    def __init__(self) -> None:
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def detect_markers_on_frame(self, frame):
        (corners, ids, rejected) = self.detector.detectMarkers(frame)
        return (corners, ids, rejected)

    def colorize_markers_on_frame(self, corners, frame):
        for corner_set in corners:
            # Draw lines around each detected marker
            for i in range(4):
                start_point = tuple(map(int, corner_set[0][i]))
                end_point = tuple(map(int, corner_set[0][(i + 1) % 4]))
                color = (0, 255, 0)  # Green color for the lines
                thickness = 2
                frame = cv2.line(frame, start_point, end_point, color, thickness)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    # def corect_histogram_on_frame(ids, corners, frame):