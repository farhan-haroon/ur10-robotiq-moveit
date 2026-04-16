#!/usr/bin/env python3
"""
Move the UR10 manipulator through 10 named positions and capture
RGB + depth images at each one.

Prerequisites:
    - Full launch is running (spawn_ur10_camera_gripper_moveit.launch.py)
    - Simulation is unpaused

Usage (run from the directory where you want images saved):
    ros2 run ur10_robotiq capture_views
"""

import os
import time
import xml.etree.ElementTree as ET
from threading import Event, Thread

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from moveit_msgs.action import MoveGroup as MoveGroupAction
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, JointConstraint,
    PlanningOptions, MoveItErrorCodes,
)
from ament_index_python.packages import get_package_share_directory


# ── Config ────────────────────────────────────────────────────────────────────
POSITIONS = [
    "top_view",
    "top_1", "top_2", "top_3", "top_4", "top_5", "top_6",
    "front_view",
    "left_view",
    "right_view",
]

# Maps SRDF state name → short image label suffix
POSITION_LABEL = {
    "top_view":   "top",
    "top_1":      "top_1",
    "top_2":      "top_2",
    "top_3":      "top_3",
    "top_4":      "top_4",
    "top_5":      "top_5",
    "top_6":      "top_6",
    "front_view": "front",
    "left_view":  "left",
    "right_view": "right",
}

PLANNING_GROUP        = "ur10_manipulator"
SETTLE_TIME           = 0.5    # seconds to wait after motion completes
IMAGE_TIMEOUT         = 10.0   # seconds to wait for a camera frame
MOVE_TIMEOUT          = 60.0   # seconds before declaring a move timed-out
ALLOWED_PLANNING_TIME = 15.0
MAX_VEL_SCALE         = 1.0
MAX_ACC_SCALE         = 1.0


# ── SRDF helpers ──────────────────────────────────────────────────────────────
def parse_named_states(srdf_path: str) -> dict:
    """Return {state_name: {joint_name: float}} from an SRDF file."""
    root = ET.parse(srdf_path).getroot()
    states = {}
    for gs in root.findall("group_state"):
        name = gs.get("name")
        states[name] = {j.get("name"): float(j.get("value"))
                        for j in gs.findall("joint")}
    return states


# ── Image-saving helpers ──────────────────────────────────────────────────────
def save_images(rgb, depth, label: str, out_dir: str):
    """
    label is already the final stem, e.g. 'it2_top'.
    Saves  it2_top.png  and  it2_top_depth.png
    """
    if rgb is not None:
        p = os.path.join(out_dir, f"{label}.png")
        cv2.imwrite(p, rgb)
        print(f"    saved {p}")
    else:
        print(f"    [WARN] no RGB frame for '{label}'")

    if depth is not None:
        # Convert to meters
        if depth.dtype == np.float32:
            d = depth.copy()
        else:
            d = depth.astype(np.float32) / 1000.0

        # Remove invalid values
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        # Valid pixels only (ignore zeros)
        valid = (d > 0.0)

        depth_8u = np.zeros_like(d, dtype=np.uint8)

        if valid.any():
            d_valid = d[valid]

            # Percentile-based scaling (like rqt)
            d_min = np.percentile(d_valid, 1)
            d_max = np.percentile(d_valid, 99)

            if d_max > d_min:
                norm = (d - d_min) / (d_max - d_min)
                norm = np.clip(norm, 0, 1)

                depth_8u[valid] = (norm[valid] * 255).astype(np.uint8)

        # OPTIONAL: invert to match rqt visual style
        # depth_8u = 255 - depth_8u

        p = os.path.join(out_dir, f"{label}_depth.png")
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        cv2.imwrite(p, depth_color)
        print(f"    saved {p}")

    else:
        print(f"    [WARN] no depth frame for '{label}'")

# ── Main node ─────────────────────────────────────────────────────────────────
class CaptureNode(Node):

    def __init__(self):
        super().__init__(
            "capture_views",
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    "use_sim_time",
                    rclpy.parameter.Parameter.Type.BOOL,
                    True,
                )
            ],
        )
        self._bridge = CvBridge()

        # image latches
        self._rgb:   np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._rgb_ready   = Event()
        self._depth_ready = Event()

        # action client to the already-running move_group
        self._ac = ActionClient(self, MoveGroupAction, "/move_action")

        self.create_subscription(Image, "/camera/image_raw",
                                 self._on_rgb,   10)
        self.create_subscription(Image, "/camera/depth/image_raw",
                                 self._on_depth, 10)

    # ── image callbacks ───────────────────────────────────────────────────────
    def _on_rgb(self, msg: Image):
        if not self._rgb_ready.is_set():
            self._rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._rgb_ready.set()

    def _on_depth(self, msg: Image):
        if not self._depth_ready.is_set():
            enc = "16UC1" if msg.encoding == "16UC1" else "32FC1"
            self._depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding=enc)
            self._depth_ready.set()

    # ── motion ────────────────────────────────────────────────────────────────
    def move_to(self, label: str, joint_values: dict) -> bool:
        """
        Send a MoveGroup goal and block (via threading.Event) until done.
        Returns True on SUCCESS.
        """
        done  = Event()
        ok    = [False]

        def on_result(future):
            val = future.result().result.error_code.val
            ok[0] = (val == MoveItErrorCodes.SUCCESS)
            if not ok[0]:
                self.get_logger().error(
                    f"  move_group error code {val} for '{label}'")
            done.set()

        def on_goal(future):
            gh = future.result()
            if gh is None or not gh.accepted:
                self.get_logger().error(f"  goal rejected for '{label}'")
                done.set()
                return
            gh.get_result_async().add_done_callback(on_result)

        # build goal
        jcs = []
        for name, pos in joint_values.items():
            jc = JointConstraint()
            jc.joint_name       = name
            jc.position         = pos
            jc.tolerance_above  = 0.01
            jc.tolerance_below  = 0.01
            jc.weight           = 1.0
            jcs.append(jc)

        req = MotionPlanRequest()
        req.group_name                    = PLANNING_GROUP
        req.num_planning_attempts         = 10
        req.allowed_planning_time         = ALLOWED_PLANNING_TIME
        req.max_velocity_scaling_factor   = MAX_VEL_SCALE
        req.max_acceleration_scaling_factor = MAX_ACC_SCALE
        req.goal_constraints              = [Constraints(joint_constraints=jcs)]

        opts = PlanningOptions()
        opts.plan_only      = False
        opts.replan         = True
        opts.replan_attempts = 3
        opts.replan_delay   = 2.0

        goal_msg = MoveGroupAction.Goal()
        goal_msg.request         = req
        goal_msg.planning_options = opts

        self._ac.send_goal_async(goal_msg).add_done_callback(on_goal)
        done.wait(timeout=MOVE_TIMEOUT)
        return ok[0]

    # ── image capture ─────────────────────────────────────────────────────────
    def capture(self):
        self._rgb = None
        self._depth = None
        self._rgb_ready.clear()
        self._depth_ready.clear()
        self._rgb_ready.wait(IMAGE_TIMEOUT)
        self._depth_ready.wait(IMAGE_TIMEOUT)
        return self._rgb, self._depth


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    rclpy.init()

    out_dir = os.getcwd()
    print(f"Output directory: {out_dir}")

    iteration = input("Iteration no.? ").strip()
    if not iteration.isdigit():
        print("[ERROR] Iteration number must be an integer.")
        rclpy.shutdown()
        return
    prefix = f"it{iteration}"
    print(f"Using prefix: {prefix}")

    srdf_path = os.path.join(
        get_package_share_directory("ur10_camera_gripper_moveit_config"),
        "config", "ur.srdf",
    )
    named_states = parse_named_states(srdf_path)
    print(f"Loaded {len(named_states)} named states from SRDF")

    node = CaptureNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    Thread(target=executor.spin, daemon=True).start()

    print("Waiting for /move_action action server …")
    if not node._ac.wait_for_server(timeout_sec=30.0):
        print("[ERROR] /move_action not available – is the launch running?")
        rclpy.shutdown()
        return

    for position in POSITIONS:
        if position not in named_states:
            print(f"\n[ERROR] '{position}' not in SRDF – skipping")
            continue

        # Via-point: pass through top_view before right_view to guarantee
        # a reachable IK solution from left_view's configuration.
        if position == "right_view":
            print(f"\n── 'top_view' (waypoint, no capture) ──")
            ok = node.move_to("top_view", named_states["top_view"])
            if not ok:
                print("  Waypoint motion to top_view failed – still attempting right_view")
            else:
                time.sleep(SETTLE_TIME)

        print(f"\n── '{position}' ──")
        ok = node.move_to(position, named_states[position])
        if not ok:
            print(f"  Motion failed – skipping capture")
            continue

        print(f"  Settling for {SETTLE_TIME} s …")
        time.sleep(SETTLE_TIME)

        print(f"  Capturing …")
        rgb, depth = node.capture()
        short = POSITION_LABEL.get(position, position)
        save_images(rgb, depth, f"{prefix}_{short}", out_dir)

    print("\nDone.")
    executor.shutdown(timeout_sec=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
