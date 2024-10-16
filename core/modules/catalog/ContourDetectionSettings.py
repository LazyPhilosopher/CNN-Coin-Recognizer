import json


class ContourDetectionSettings:
    def __init__(self,
                 blur_kernel=3,
                 blur_sigma=2,
                 canny_threshold_1=50,
                 canny_threshold_2=9,
                 dilate_kernel_1=4,
                 dilate_kernel_2=2,
                 erode_kernel_1=13,
                 erode_kernel_2=7,
                 dilate_iteration=11,
                 erode_iteration=4):

        self.blur_kernel: int = blur_kernel
        self.blur_sigma: int = blur_sigma
        self.canny_threshold_1: int = canny_threshold_1
        self.canny_threshold_2: int = canny_threshold_2
        self.dilate_kernel_1: int = dilate_kernel_1
        self.dilate_kernel_2: int = dilate_kernel_2
        self.erode_kernel_1: int = erode_kernel_1
        self.erode_kernel_2: int = erode_kernel_2
        self.dilate_iteration: int = dilate_iteration
        self.erode_iteration: int = erode_iteration

    def to_dict(self):
        return {
            "blur_kernel": self.blur_kernel,
            "blur_sigma": self.blur_sigma,
            "canny_threshold_1": self.canny_threshold_1,
            "canny_threshold_2": self.canny_threshold_2,
            "dilate_kernel_1": self.dilate_kernel_1,
            "dilate_kernel_2": self.dilate_kernel_2,
            "erode_kernel_1": self.erode_kernel_1,
            "erode_kernel_2": self.erode_kernel_2,
            "dilate_iteration": self.dilate_iteration,
            "erode_iteration": self.erode_iteration
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
