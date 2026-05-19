
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # The robot at X(3) rotates 45 deg CCW, moves 2 m forward (in the new heading),
    # then rotates another 45 deg CCW. Expressed as a single relative motion in X(3)'s
    # local frame, this is a translation of (2*cos(45 deg), 2*sin(45 deg)) = (sqrt(2), sqrt(2))
    # followed by a total heading change of 90 deg = pi/2.
    relative_motion = gtsam.Pose2(math.sqrt(2.0), math.sqrt(2.0), math.pi / 2.0)

    # TODO: Add the odometry factor between X(4) and X(5) to the graph (BetweenFactorPose2)
    # (The comment in the template refers to X(5) by mistake; the factor goes between X(3) and X(4).)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), relative_motion, ODOMETRY_NOISE))

    # TODO: Based on the odometry, find the initial estimate for the pose of X(5) and add it to the graph
    # If X(3) were perfectly at (4, 0, 0) (its true value), applying the relative motion
    # would put X(4) at exactly (4 + sqrt(2), sqrt(2), pi/2). We use that as the initial guess
    # for X(4); optimization will refine it together with the noisy X(1)/X(2)/X(3) estimates.
    pose_4 = gtsam.Pose2(4.0 + math.sqrt(2.0), math.sqrt(2.0), math.pi / 2.0)
    initial_estimate.insert(X(4), pose_4)

    return graph, initial_estimate
