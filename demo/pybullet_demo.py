import os
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import open3d as o3d
from scipy.spatial.transform import Rotation

from src.shepherd.shepherd import Shepherd
from src.shepherd.shepherd_config import ShepherdConfig
from src.shepherd.utils.camera import CameraUtils


class PyBulletEnv(gym.Env):
    """RL environment wrapper for PyBullet simulation."""

    def __init__(self, shepherd: Shepherd):
        super().__init__()

        # Store Shepherd instance
        self.shepherd = shepherd

        # Initialize PyBullet
        p.connect(p.GUI)  # or p.DIRECT for headless mode
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load some basic objects
        self.objects = self._load_objects()

        # Setup camera parameters
        self.width = 256
        self.height = 256
        self.fov = 90
        self.aspect = self.width / self.height
        self.near = 0.1
        self.far = 10

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # move_forward, turn_left, turn_right, do nothing
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0, high=np.inf, shape=(256, 256, 1), dtype=np.float32)
        })

        # Camera state
        self.camera_pos = [0, 0, 1.5]  # x, y, height
        self.camera_yaw = 0  # rotation around z-axis
        self.move_step = 0.1
        self.turn_step = 0.1

        # Frame processing control
        self.last_frame_results = None
        self.last_frame_time = None
        self.frame_skip = 2
        self.frame_count = 0

        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), "pybullet_output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Add color map for visualization
        self.color_map = {}
        self.next_color_idx = 0
        self.color_palette = [
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

    def _load_objects(self) -> Dict[str, int]:
        """Load objects into the simulation."""
        objects = {}
        
        # Load a table
        table_pos = [1, 0, 0]
        table_ori = p.getQuaternionFromEuler([0, 0, 0])
        objects['table'] = p.loadURDF("table/table.urdf", table_pos, table_ori, globalScaling=1.0)

        # Load some basic shapes using primitive shapes
        # Cube
        cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        cube_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])
        objects['cube'] = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cube_col, 
                                          baseVisualShapeIndex=cube_vis, basePosition=[1, 0.3, 1])

        # Sphere
        sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
        sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 1])
        objects['sphere'] = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_col,
                                            baseVisualShapeIndex=sphere_vis, basePosition=[1, -0.3, 1])

        # Load a duck (comes with pybullet_data)
        duck_pos = [1.5, 0, 1]
        duck_ori = p.getQuaternionFromEuler([0, 0, 0])
        objects['duck'] = p.loadURDF("duck_vhacd.urdf", duck_pos, duck_ori, globalScaling=1.0)

        return objects

    def get_object_color(self, object_id: str, similarity: float = None) -> Tuple[int, int, int]:
        """Get color based on query similarity or assign distinct color if no query."""
        if similarity is not None:
            intensity = int(similarity * 255)
            return (0, 0, intensity)
        else:
            if object_id not in self.color_map:
                color = self.color_palette[self.next_color_idx % len(self.color_palette)]
                self.color_map[object_id] = color
                self.next_color_idx += 1
            return self.color_map[object_id]

    def _get_camera_pose(self) -> Dict:
        """Get current camera pose in world coordinates."""
        # Calculate camera target position (looking forward)
        forward_x = self.camera_pos[0] + np.cos(self.camera_yaw)
        forward_y = self.camera_pos[1] + np.sin(self.camera_yaw)
        target_pos = [forward_x, forward_y, self.camera_pos[2]]

        # Get view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
        )

        # Convert view matrix to position and orientation
        # Note: This is a simplified conversion
        rot = Rotation.from_euler('z', self.camera_yaw)
        quat = rot.as_quat()

        return {
            "x": float(self.camera_pos[0]),
            "y": float(self.camera_pos[1]),
            "z": float(self.camera_pos[2]),
            "qx": float(quat[0]),
            "qy": float(quat[1]),
            "qz": float(quat[2]),
            "qw": float(quat[3])
        }

    def _get_camera_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get RGB and depth images from PyBullet camera."""
        # Calculate camera target position
        forward_x = self.camera_pos[0] + np.cos(self.camera_yaw)
        forward_y = self.camera_pos[1] + np.sin(self.camera_yaw)
        target_pos = [forward_x, forward_y, self.camera_pos[2]]

        # Compute view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far
        )

        # Get camera image
        width, height, rgb, depth, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Convert RGB to BGR for OpenCV compatibility
        rgb = rgb[:, :, :3]  # Remove alpha channel
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Convert depth to meters
        depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
        depth = depth.reshape(height, width)

        return rgb, depth

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        """Execute action and return new observation."""
        # Execute action
        if action == 0:  # move forward
            self.camera_pos[0] += self.move_step * np.cos(self.camera_yaw)
            self.camera_pos[1] += self.move_step * np.sin(self.camera_yaw)
        elif action == 1:  # turn left
            self.camera_yaw += self.turn_step
        elif action == 2:  # turn right
            self.camera_yaw -= self.turn_step

        # Step physics simulation
        p.stepSimulation()

        # Get observation
        obs = self._get_observation()

        return obs, 0.0, False, False, {}

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation including vision processing results."""
        # Get camera images
        rgb, depth = self._get_camera_image()

        # Get camera pose
        camera_pose = self._get_camera_pose()

        # Process frame with Shepherd
        self.frame_count += 1
        if self.frame_count % self.frame_skip == 0:
            print("\n=== Processing New Frame ===")
            print(f"Camera Position: ({camera_pose['x']:.2f}, {camera_pose['y']:.2f}, {camera_pose['z']:.2f})")

            # Process frame and update point cloud
            results = self.shepherd.process_frame(rgb, depth, camera_pose)
            self.last_frame_results = results
            self.last_frame_time = time.time()

        return {
            "rgb": rgb,
            "depth": depth,
            "results": self.last_frame_results,
            "camera_pose": camera_pose
        }

    def reset(self, *, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed, options=options)
        
        # Reset camera position and orientation
        self.camera_pos = [0, 0, 1.5]
        self.camera_yaw = 0

        # Reset object positions if needed
        # Add code here if you want to randomize object positions on reset

        return self._get_observation(), {}

    def render(self):
        """Render environment with detections and depth visualization."""
        obs = self._get_observation()

        # Create visualization frame
        viz_frame = obs["rgb"].copy()

        # Draw detections and masks
        if obs["results"]:
            for result in obs["results"]:
                bbox = result["detection"]["bbox"]
                mask = result["mask"]
                object_id = result.get("object_id")
                similarity = result.get("similarity")

                # Get color based on whether we have a query
                if self.shepherd.config.default_query:
                    color = self.get_object_color(object_id, similarity)
                else:
                    color = self.get_object_color(object_id)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)

                # Draw mask
                mask_overlay = viz_frame.copy()
                mask_overlay[mask] = color
                viz_frame = cv2.addWeighted(viz_frame, 0.7, mask_overlay, 0.3, 0)

                # Add text
                text = f"ID: {object_id}"
                if similarity is not None:
                    text += f" ({similarity:.2f})"
                cv2.putText(viz_frame, text, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add query information
        query_text = f"Query: {self.shepherd.config.default_query}"
        cv2.putText(viz_frame, query_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add depth visualization
        depth_viz = cv2.normalize(obs["depth"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

        # Combine RGB and depth
        combined_viz = np.hstack((viz_frame, depth_viz))

        return combined_viz

    def save_current_point_cloud(self):
        """Save current point cloud to PLY file."""
        output_path = os.path.join(self.output_dir, "pybullet_point_cloud.ply")
        self.shepherd.database.save_point_cloud_ply(output_path)
        print(f"Saved point cloud to: {output_path}")

    def close(self):
        """Clean up resources."""
        p.disconnect()


def main():
    # Initialize Shepherd with config
    config = ShepherdConfig(camera_height=1.5, camera_pitch=0.0)

    # Update camera parameters for PyBullet
    config.camera = CameraUtils(
        width=256,
        height=256,
        fov=1.57,  # 90 degrees FOV
        camera_height=1.5,
        camera_pitch=0.0,
        camera_yaw=0.0,
        camera_roll=0.0,
        coordinate_frame="pybullet"
    )

    # Print initial configuration
    print("\nInitial Configuration:")
    print(f"Camera FOV: {np.degrees(config.camera.fov):.1f} degrees")
    print(f"Camera height: {config.camera.camera_height:.2f}m")
    print(f"Camera angles (degrees):")
    print(f"  Pitch: {np.degrees(config.camera.camera_pitch):.1f}")
    print(f"  Yaw: {np.degrees(config.camera.camera_yaw):.1f}")
    print(f"  Roll: {np.degrees(config.camera.camera_roll):.1f}")

    shepherd = Shepherd(config=config)

    try:
        env = PyBulletEnv(shepherd)
        obs, _ = env.reset()

        print("\nControls:")
        print("W - Move forward")
        print("A - Turn left")
        print("D - Turn right")
        print("Q - Enter query")
        print("S - Save point cloud")
        print("ESC - Exit")

        while True:
            # Render and display
            frame = env.render()
            cv2.imshow("PyBullet Demo", frame)

            # Handle input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("w"):
                action = 0
            elif key == ord("a"):
                action = 1
            elif key == ord("d"):
                action = 2
            elif key == ord("q"):
                query = input("\nEnter query: ")
                shepherd.update_query(query)
                continue
            elif key == ord("s"):
                env.save_current_point_cloud()
                continue
            elif key == 27:  # ESC
                break
            else:
                continue

            obs, _, _, _, _ = env.step(action)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()