from enum import Enum

class ResultsType(Enum):
    FRAME_ONLY = "FrameOnly"
    DETECTIONS = "Detections"
    ALL = "All"

class ResultsCollections:
    """
    Results collection class for storing and managing results.
    """
    def __init__(self, max_size=40):
        self.results = {}
        self.last_consumed = None
        self.latest_frame_id = -1
        self.latest_results_id = {
            ResultsType.FRAME_ONLY: -1,
            ResultsType.DETECTIONS: -1,
            ResultsType.ALL: -1
        }
        self.max_size = max_size
    
    def __getitem__(self, frame_id):
        if frame_id in self.results:
            result = self.results[frame_id]
            self.last_consumed = result
            self.latest_frame_id = frame_id
            return result
        else:
            return None

    def add_frame(self, frame_id, rgb, depth, camera_pose):
        self.results[frame_id] = [ResultsType.FRAME_ONLY, rgb, depth, camera_pose]
        if len(self.results) >= self.max_size:
            # get the oldest frame
            oldest_frame_id = min(self.results.keys(), key=int)
            del self.results[oldest_frame_id]
        self.latest_frame_id = frame_id
        return self.results[frame_id]

    def pop(self, frame_id):
        if frame_id in self.results:
            result = self.results[frame_id]
            del self.results[frame_id]
            return result
        else:
            return None

    def update(self, frame_id: int, result_type: ResultsType, result: list):
        if frame_id in self.results:
            self.results[frame_id] = [result_type] + result
            self.latest_results_id[result_type] = frame_id
            return result
        else:
            return None