#!/usr/bin/env python3
"""
Print the current pose of a Gazebo model in the 'workspace' frame.

Usage:
    ros2 run ur10_robotiq get_model_pose
"""

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState


class PoseQuery(Node):

    def __init__(self):
        super().__init__("get_model_pose")
        self._cli = self.create_client(GetEntityState, "/gazebo/get_entity_state")

        if not self._cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                "/gazebo/get_entity_state not available – is Gazebo running?"
            )
            raise RuntimeError("Service unavailable")

    def query(self, model_name: str, reference_frame: str = "workspace"):
        req = GetEntityState.Request()
        req.name = model_name
        req.reference_frame = reference_frame

        future = self._cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is None:
            print(f"[ERROR] Service call timed out for '{model_name}'")
            return

        resp = future.result()
        if not resp.success:
            print(f"[ERROR] Model '{model_name}' not found in simulation")
            return

        p = resp.state.pose.position
        o = resp.state.pose.orientation

        print(f"\nModel     : {model_name}")
        print(f"Frame     : {reference_frame}")
        print(f"Position  : x={p.x:.6f}  y={p.y:.6f}  z={p.z:.6f}")
        print(f"Quaternion: x={o.x:.6f}  y={o.y:.6f}  z={o.z:.6f}  w={o.w:.6f}")


def main():
    rclpy.init()

    model_name = input("Model name? ").strip()
    if not model_name:
        print("[ERROR] No model name provided.")
        rclpy.shutdown()
        return

    try:
        node = PoseQuery()
    except RuntimeError:
        rclpy.shutdown()
        return

    node.query(model_name)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
