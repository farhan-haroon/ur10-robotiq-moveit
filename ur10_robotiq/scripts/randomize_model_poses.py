#!/usr/bin/env python3
import rclpy
import random
import math

from rclpy.node import Node
from gazebo_msgs.srv import GetModelList, GetEntityState, SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose


EXCLUDED_MODELS = {
    "ground_plane",
    "table_marble",
    "table",
    "tray_green",
    "tray_red",
    "tray_blue",
    "camera",
    "workspace"
}


class RandomizeModels(Node):

    def __init__(self):
        super().__init__("randomize_models")

        self.get_models = self.create_client(GetModelList, "/get_model_list")
        self.get_state = self.create_client(GetEntityState, "/gazebo/get_entity_state")
        self.set_state = self.create_client(SetEntityState, "/gazebo/set_entity_state")

        self.wait_for_services()
        self.randomize_all_models()

    def wait_for_services(self):
        for client in [self.get_models, self.get_state, self.set_state]:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Waiting for {client.srv_name}...")

    def randomize_all_models(self):
        req = GetModelList.Request()
        future = self.get_models.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        models = future.result().model_names
        self.get_logger().info(f"Detected models: {models}")

        for model in models:
            base_name = model.split("::")[0]

            if base_name in EXCLUDED_MODELS:
                self.get_logger().info(f"Skipping: {model}")
                continue

            self.get_logger().info(f"Processing: {model}")

            pose = self.get_model_pose(model)
            if pose is None:
                continue

            new_pose = self.randomize_pose(pose)
            self.set_model_pose(model, new_pose)

        self.get_logger().info("✅ Randomization complete")

    def get_model_pose(self, model_name):
        req = GetEntityState.Request()
        req.name = model_name

        future = self.get_state.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is None or not result.success:
            self.get_logger().warn(f"Failed to get state for {model_name}")
            return None

        return result.state.pose

    def randomize_pose(self, pose: Pose):
        new_pose = Pose()

        # 🎯 Sample within circle (radius ≤ 0.05 m) centered at (0,0)
        r = random.uniform(0.0, 0.05)
        theta = random.uniform(0.0, 2 * math.pi)

        new_pose.position.x = r * math.cos(theta)
        new_pose.position.y = r * math.sin(theta)

        # 🎯 Height perturbation
        new_pose.position.z = pose.position.z + random.uniform(0.2, 0.3)

        # 🎯 Full random orientation
        roll = random.uniform(-0.3, 0.3)
        pitch = random.uniform(-0.3, 0.3)
        yaw = random.uniform(-math.pi, math.pi)

        qx, qy, qz, qw = self.euler_to_quaternion(roll, pitch, yaw)

        new_pose.orientation.x = qx
        new_pose.orientation.y = qy
        new_pose.orientation.z = qz
        new_pose.orientation.w = qw

        return new_pose

    def set_model_pose(self, model_name, pose):
        req = SetEntityState.Request()
        state = EntityState()

        state.name = model_name
        state.pose = pose
        state.reference_frame = "workspace"

        req.state = state

        future = self.set_state.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is None or not result.success:
            self.get_logger().warn(f"Failed to set pose for {model_name}")

    def euler_to_quaternion(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return qx, qy, qz, qw


def main():
    rclpy.init()
    node = RandomizeModels()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()