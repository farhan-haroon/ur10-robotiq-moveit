#!/usr/bin/env python3
import math
import os
import sys
import struct
import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class PointCloudCapture(Node):

    def __init__(self, output_dir: str):
        super().__init__("pointcloud_capture")
        self._output_dir = output_dir
        self._done = False

        self._sub = self.create_subscription(
            PointCloud2,
            "/camera/points",
            self._callback,
            10,
        )

        self.get_logger().info("Waiting for /camera/points ...")

    def _callback(self, msg: PointCloud2):
        if self._done:
            return

        self.get_logger().info(
            f"Received cloud: {msg.width * msg.height} points"
        )

        field_names = [f.name for f in msg.fields]

        # Read all points safely
        points = list(point_cloud2.read_points(msg, skip_nans=False))

        # Generate filename
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self._output_dir, f"pointcloud_{stamp}.ply")

        # Write PLY safely
        valid_count = self._write_ply(filename, points, field_names)

        self.get_logger().info(f"Saved {valid_count} valid points → {filename}")

        # Mark done ONLY after successful write
        self._done = True

    def _write_ply(self, path: str, points, field_names: list):
        has_rgb = "rgb" in field_names or "rgba" in field_names
        rgb_key = "rgba" if "rgba" in field_names else "rgb" if has_rgb else None
        rgb_idx = field_names.index(rgb_key) if has_rgb else None

        valid_points = []

        for pt in points:
            try:
                x, y, z = float(pt[0]), float(pt[1]), float(pt[2])

                # Filter invalid values
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                    continue

                if has_rgb and rgb_idx is not None:
                    rgb_raw = pt[rgb_idx]

                    # Safe unpack
                    packed = struct.pack("f", float(rgb_raw))
                    bgr = struct.unpack("I", packed)[0]

                    r = (bgr >> 16) & 0xFF
                    g = (bgr >> 8) & 0xFF
                    b = bgr & 0xFF

                    valid_points.append((x, y, z, r, g, b))
                else:
                    valid_points.append((x, y, z))

            except Exception:
                continue  # skip bad points completely

        # Write file (fully consistent)
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(valid_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")

            if has_rgb:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            f.write("end_header\n")

            for pt in valid_points:
                f.write(" ".join(map(str, pt)) + "\n")

        return len(valid_points)


def main():
    output_dir = os.getcwd()

    rclpy.init(args=sys.argv)
    node = PointCloudCapture(output_dir)

    while rclpy.ok() and not node._done:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()