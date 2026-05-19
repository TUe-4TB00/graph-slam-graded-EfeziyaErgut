import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    # Determine the correct rotation (bearing) and distance from X(4) to L(2)
    # We use the optimized result so the measurement is consistent with the current best
    # estimate of where X(4) and L(2) actually are in the world.
    pose_4 = result.atPose2(X(4))
    landmark_2 = result.atPoint2(L(2))

    # gtsam.Pose2 exposes helpers that compute the bearing and range from this pose
    # to a 2D point, expressed in the pose's local frame.
    bearing = pose_4.bearing(landmark_2)            # gtsam.Rot2 (radians)
    distance = pose_4.range(landmark_2)             # float (meters)
    rotation = math.degrees(bearing.theta())        # convert to degrees for fromDegrees(...)

    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2.fromDegrees(rotation), distance, MEASUREMENT_NOISE))
    return graph
