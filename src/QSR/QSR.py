from qsrlib_io.world_trace import World_Trace, Object_State
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message

class QSRPipeline:

    def __init__(self):

        self.qsrlib = QSRlib()
        self.world = World_Trace()

    def build_world_trace(self, collected_frames):

        # self.world = World_Trace()
        object_tracks = {}

        # Organize data per object
        for frame in collected_frames:
            id = frame["frame_id"]
            for obj in frame["objects"]:
                name = obj["label"]

                if name not in object_tracks:
                    object_tracks[name] = []

                object_tracks[name].append(
                    Object_State(
                        name=name,
                        timestamp=id,
                        x=obj["x"],
                        y=obj["y"]
                    )
                )

        for obj_states in object_tracks.values():
            self.world.add_object_state_series(obj_states)

        return self.world


    def compute_qtc(self, world):


        request = QSRlib_Request_Message(
            which_qsr="qtcbs",
            input_data=world
        )

        response = self.qsrlib.request_qsrs(request)
        return response


    def print_qtc(self, response):
        for t, qsrs in response.qsrs.trace.items():
            print(f"Time {t}:")
            for pair, relation in qsrs.qsrs.items():
                print(f"  {pair}: {relation.qsr}")


    detections = [
        {"timestamp": 0, "objects": [
            {"label": "robot", "x": 0, "y": 0},
            {"label": "person", "x": 5, "y": 0}
        ]},
        {"timestamp": 1, "objects": [
            {"label": "robot", "x": 1, "y": 0},
            {"label": "person", "x": 4, "y": 0}
        ]}
    ]
