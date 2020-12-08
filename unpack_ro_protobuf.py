# Attempting to get into a pbRadarOdometryState
import sys
import os

ro_state_path = "/workspace/Desktop/ro_state.monolithic"

# Magic
sys.path.insert(-1, "/workspace/code/corelibs/src/tools-python")
sys.path.insert(-1, "/workspace/code/corelibs/build/datatypes")
sys.path.insert(-1, "/workspace/code/radar-navigation/build/radarnavigation_datatypes_python")

from mrg.logging.indexed_monolithic import IndexedMonolithic


class RadarOdometryState(object):

    def __init__(self):
        """Creates an empty radar odometry state with no parameters set.
        """
        self.timestamp = None


def get_ro_state_from_pb(pb_ro_state):
    print("Running get_ro_state_from_pb...")
    radar_odometry_state = RadarOdometryState()
    radar_odometry_state.timestamp = pb_ro_state.timestamp

    return radar_odometry_state


# Open monolithic and iterate frames
print("reading ro_state_path: " + ro_state_path)

# You need to run this: ~/code/corelibs/build/tools-cpp/bin/MonolithicIndexBuilder
# -i /Users/roberto/Desktop/ro_state.monolithic -o /Users/roberto/Desktop/ro_state.monolithic.index
radar_state_mono = IndexedMonolithic(ro_state_path)
idx = 0
pb_state, name_scan, _ = radar_state_mono[idx]
ro_state = get_ro_state_from_pb(pb_state)
print(ro_state.timestamp)
print("Finished!")

# iterate mono
# se3s = []
# timestamps = []
# for pb_serialised_transform, _, _ in monolithic_decoder:
#     # adapt
#     serialised_transform = PbSerialisedTransformToPython(
#         pb_serialised_transform)
#     se3s.append(serialised_transform[0])
#     timestamps.append(serialised_transform[1])
#
# print("Finished reading", len(timestamps), "poses.")
