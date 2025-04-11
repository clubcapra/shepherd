"""
Database wrapper.
"""

import os
import random
from concurrent.futures import Future
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
import open3d as o3d
import torch
from dbscan import DBSCAN
from sklearn.neighbors import NearestNeighbors
from chamferdist import ChamferDistance

from .utils.camera import CameraUtils

class DatabaseWrapper:
    """
    Database wrapper.
    """

    def __init__(
        self,
        collection_name: str = "detection_embeddings",
        camera_utils: Optional[CameraUtils] = None,
        distance_threshold: float = 1.5,
        similarity_threshold: float = 0.7,
        cluster_eps: float = 0.2,
        cluster_min_samples: int = 3,
    ):
        """Initialize database wrapper with ChromaDB."""
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.initialize_collection()
        # TODO: explore if this is optimal for memory efficiency - perhaps a second vector DB for point clouds would be better?
        self.point_clouds = {}  # Store point clouds in memory
        self.camera = camera_utils if camera_utils is not None else CameraUtils()

        # Thresholds for object matching and merging
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples

        # Initialize nearest neighbors
        self.nn = NearestNeighbors(n_neighbors=3, algorithm="ball_tree")
        self.chamfer = ChamferDistance()
        self.query_embedding = None  # Add query embedding storage
        self.pending_captions = {}  # Store pending caption futures    

    def _clean_point_cloud(
        self,
        point_cloud: np.ndarray,
    ) -> np.ndarray:
        """Clean point cloud using nearest neighbors and line-of-sight filtering."""
        if point_cloud is None or len(point_cloud) < self.cluster_min_samples:
            return np.array([])

        try:
            # Remove NaN and infinite values
            valid_mask = np.all(np.isfinite(point_cloud), axis=1)
            point_cloud = point_cloud[valid_mask]

            if len(point_cloud) < self.cluster_min_samples:
                return point_cloud  # Return original if too few points

            # Use nearest neighbors to find core points
            self.nn.fit(point_cloud)
            distances, _ = self.nn.kneighbors(point_cloud)
            core_points_mask = distances.ravel() < self.cluster_eps
            core_points = point_cloud[core_points_mask]

            if len(core_points) == 0:
                return point_cloud  # Return original if no core points found

            return point_cloud

        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            print(f"Error in point cloud cleaning: {e}")
            return point_cloud
        
    def _preprocess_point_clouds(self, anchor, candidate, align=True):
        """
        Preprocess point clouds for similarity comparison:
        1. Normalize to account for different scales
        2. Optionally align using ICP
        
        Parameters:
        anchor, candidate: numpy arrays of shape (n_points, 3)
        align: boolean, whether to perform ICP alignment
        
        Returns:
        Processed versions of anchor and candidate
        """
        # Step 1: Normalize both point clouds
        # Center each point cloud around its centroid
        centroid_a = np.mean(anchor, axis=0)
        centroid_b = np.mean(candidate, axis=0)
        
        centered_a = anchor - centroid_a
        centered_b = candidate - centroid_b
        
        # Scale to unit standard deviation
        scale_a = np.std(np.linalg.norm(centered_a, axis=1))
        scale_b = np.std(np.linalg.norm(centered_b, axis=1))
        
        normalized_a = centered_a / scale_a
        normalized_b = centered_b / scale_b
        
        if not align:
            return normalized_a, normalized_b
        
        # Step 2: Align using ICP if requested
        # Convert to Open3D point clouds for ICP
        pcd_a = o3d.geometry.PointCloud()
        pcd_b = o3d.geometry.PointCloud()
        
        pcd_a.points = o3d.utility.Vector3dVector(normalized_a)
        pcd_b.points = o3d.utility.Vector3dVector(normalized_b)
        
        # Perform ICP
        # Set parameters - adjust max_correspondence_distance as needed
        threshold = 0.05  # Max correspondence distance
        trans_init = np.identity(4)  # Initial transformation
        
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_b, pcd_a, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        # Apply transformation to point cloud B
        aligned_b = np.asarray(pcd_b.transform(reg_p2p.transformation).points)
        
        return normalized_a, aligned_b

    def _process_new_detection(
        self, point_cloud: np.ndarray, embedding: np.ndarray, camera_pose: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process new detection with DBSCAN clustering and pose transformation."""

        if point_cloud is None or len(point_cloud) < self.cluster_min_samples:
            return np.array([]), embedding

        try:
            # Transform to world coordinates
            point_cloud = self.camera.transform_point_cloud(point_cloud, camera_pose)

            # Run DBSCAN directly on point cloud
            labels, _ = DBSCAN(
                point_cloud,
                eps=self.cluster_eps,
                min_samples=self.cluster_min_samples,
            )

            # Get largest cluster
            unique_labels = np.unique(labels)
            if len(unique_labels) == 1 and unique_labels[0] == -1:
                return np.array([]), embedding

            # Get points from largest non-noise cluster
            cluster_sizes = np.array(
                [np.sum(labels == label) for label in unique_labels if label != -1]
            )
            if len(cluster_sizes) == 0:
                return np.array([]), embedding

            main_cluster_label = unique_labels[np.argmax(cluster_sizes)]
            cleaned_points = point_cloud[labels == main_cluster_label]

            return cleaned_points, embedding

        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            print(f"Error in detection processing: {e}")
            return np.array([]), embedding

    def _compute_geo_similarity(
        self, points1: np.ndarray, points2: np.ndarray, nn_threshold: float = 0.1
    ) -> float:
        """Compute geometric similarity using nearest neighbor ratio."""
        if len(points1) == 0 or len(points2) == 0:
            return 0.0

        try:
            # Compute centroids
            centroid1 = np.mean(points1, axis=0)
            centroid2 = np.mean(points2, axis=0)

            # Compute distance between centroids
            centroid_distance = np.linalg.norm(centroid1 - centroid2)

            # If centroids are too far apart, return 0
            if centroid_distance > 1.0:  # 1 meter threshold
                return 0.0

            # Build KD-tree for second point cloud
            tree = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(points2)

            # Find nearest neighbors for all points in first cloud
            distances, _ = tree.kneighbors(points1)

            # Count points with neighbors within threshold
            close_points = np.sum(distances < nn_threshold)

            # Compute bidirectional similarity
            tree2 = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(points1)
            distances2, _ = tree2.kneighbors(points2)
            close_points2 = np.sum(distances2 < nn_threshold)

            # Use average of both directions
            similarity = 0.5 * (
                close_points / len(points1) + close_points2 / len(points2)
            )

            return similarity

        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            print(f"Error in geometric similarity computation: {e}")
            return 0.0

    def _find_nearby_object(self, point_cloud: np.ndarray, embedding: np.ndarray) -> Optional[str]:
        """Find nearby object using combined geometric and semantic similarity (optimized)."""
        if len(self.point_clouds) == 0 or point_cloud is None or len(point_cloud) == 0:
            return None

        try:
            best_similarity = -float("inf")

            # 1. ChromaDB Filtering:
            results = self.collection.query(
                query_embeddings=[embedding.tolist()], 
                n_results=5, 
                include=["embeddings"]  
            )

            if not results or not results["embeddings"]: 
                return None
            
            ids = results["ids"][0]

            # 2. Chamfer distance between point clouds

            best_id = ids[0]
            for i in range(len(ids)):
                obj_id = ids[i]
                stored_cloud = self.point_clouds.get(obj_id)
                if stored_cloud is None or len(stored_cloud) == 0:
                    continue

                # anchor_pt, compared_pt = self._preprocess_point_clouds(anchor=point_cloud, candidate=stored_cloud, align=False)
                
                similarity = self._compute_geo_similarity(point_cloud, stored_cloud)
                if best_similarity < similarity:
                    best_similarity = similarity
                    best_id = obj_id

            if best_similarity > self.similarity_threshold:
                return best_id, best_similarity

            return None

        except Exception as e:
            print(f"Error in finding nearby object: {e}")
            return None

    def _merge_point_clouds(
        self, cloud1: np.ndarray, cloud2: np.ndarray, voxel_size: float = 0.05
    ) -> np.ndarray:
        """Merge point clouds with improved cleaning and voxelization."""
        if len(cloud1) == 0:
            return cloud2
        if len(cloud2) == 0:
            return cloud1

        # Combine clouds
        combined_cloud = np.vstack([cloud1, cloud2])

        # Remove NaN and infinite values
        valid_mask = np.all(np.isfinite(combined_cloud), axis=1)
        combined_cloud = combined_cloud[valid_mask]

        if len(combined_cloud) == 0:
            return np.array([])

        # Clean combined cloud
        cleaned_cloud = self._clean_point_cloud(combined_cloud)

        if len(cleaned_cloud) == 0:
            return np.array([])

        # Voxelize the cleaned cloud
        try:
            voxel_dict = {}
            for point in cleaned_cloud:
                voxel_key = tuple(np.round(point / voxel_size))
                if voxel_key not in voxel_dict:
                    voxel_dict[voxel_key] = []
                voxel_dict[voxel_key].append(point)

            # Compute centroids for each voxel
            voxelized_points = []
            for points in voxel_dict.values():
                if len(points) > 0:
                    centroid = np.median(points, axis=0)  # Use median for robustness
                    voxelized_points.append(centroid)

            return np.array(voxelized_points)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            print(f"Error in point cloud merging: {e}")
            return cleaned_cloud  # Return cleaned cloud if voxelization fails

    def store_object(
        self,
        embedding: np.ndarray,
        metadata: Dict,
        point_cloud: Optional[np.ndarray] = None,
        camera_pose: Optional[Dict] = None,
    ) -> Tuple[Optional[str], bool]:
        """
        Store a new object or update existing one.
        Returns (object_id, needs_caption) tuple.
        """
        if camera_pose is None:
            camera_pose = {"x": 0, "y": 0, "z": 0, "qx": 0, "qy": 0, "qz": 0, "qw": 1}

        # Process new detection
        cleaned_points, processed_embedding = self._process_new_detection(
            point_cloud, embedding, camera_pose
        )

        if len(cleaned_points) == 0:
            return None, False

        # Find nearby object
        nearby_obj = self._find_nearby_object(cleaned_points, processed_embedding)

        if nearby_obj is not None:
            # Check if object already has a caption
            nearby_id, similarity = nearby_obj
            results = self.collection.get(ids=[nearby_id], include=["metadatas"])
            existing_metadata = results["metadatas"][0]
            needs_caption = (
                not existing_metadata.get("caption")
                and nearby_id not in self.pending_captions
            )

            print(f"Merging object {nearby_id} with similarity {similarity}")
            return (
                self._merge_objects(
                    nearby_id, processed_embedding, metadata, cleaned_points
                ),
                needs_caption,
            )

        # Add new object
        new_id = str(
            self.collection.count() if self.collection.count() is not None else 0
        )

        self.point_clouds[new_id] = cleaned_points
        print(f"Adding new object {new_id}")
        self.collection.add(
            embeddings=[processed_embedding.tolist()],
            ids=[new_id],
            metadatas=[metadata],
        )

        return new_id, True

    def _merge_objects(
        self,
        existing_id: str,
        new_embedding: np.ndarray,
        new_metadata: Dict,
        new_point_cloud: np.ndarray,
    ) -> str:
        """
        Merge new object with existing one while preserving the world coordinate system.
        This function does not normalize point clouds for merging, only for alignment.
        """
        # Get existing object data
        results = self.collection.get(
            ids=[existing_id], include=["embeddings", "metadatas"]
        )

        old_embedding = np.array(results["embeddings"][0])
        old_point_cloud = self.point_clouds.get(existing_id, np.array([]))

        # If one cloud is empty, use the non-empty one
        if len(old_point_cloud) == 0:
            merged_cloud = new_point_cloud
        elif len(new_point_cloud) == 0:
            merged_cloud = old_point_cloud
        else:
            # Store original point clouds before any transformation
            orig_old = old_point_cloud.copy()
            orig_new = new_point_cloud.copy()
            
            # Create Open3D point clouds for registration
            pcd_old = o3d.geometry.PointCloud()
            pcd_new = o3d.geometry.PointCloud()
            
            pcd_old.points = o3d.utility.Vector3dVector(old_point_cloud)
            pcd_new.points = o3d.utility.Vector3dVector(new_point_cloud)
            
            # Compute normals for better registration
            pcd_old.estimate_normals()
            pcd_new.estimate_normals()
            
            # Get transformation matrix to align new_point_cloud to old_point_cloud
            # BUT do not apply it to change the actual points in the world coordinate system
            distance_threshold = 0.05
            
            # First compute centroids to roughly align before ICP
            centroid_old = np.mean(old_point_cloud, axis=0)
            centroid_new = np.mean(new_point_cloud, axis=0)
            
            # Initial transformation to align centroids
            init_translation = np.eye(4)
            init_translation[:3, 3] = centroid_old - centroid_new
            
            # Use ICP with point-to-plane for better alignment
            icp_result = o3d.pipelines.registration.registration_icp(
                pcd_new, pcd_old, distance_threshold, init_translation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            
            # DO NOT transform the points - just combine them in their original coordinates
            # This preserves the world coordinate system
            combined_points = np.vstack([orig_old, orig_new])
            
            # Clean the combined point cloud to remove outliers
            combined_pcd = o3d.geometry.PointCloud()
            combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
            
            # Use statistical outlier removal 
            pcd_cleaned, _ = combined_pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0)
            
            # Use voxel downsampling to reduce density and remove duplicates
            voxel_size = 0.02  # Adjust based on desired resolution
            pcd_downsampled = pcd_cleaned.voxel_down_sample(voxel_size)
            
            merged_cloud = np.asarray(pcd_downsampled.points)

        # Update embedding with weighted average
        weight_old = len(old_point_cloud) / max(len(old_point_cloud) + len(new_point_cloud), 1)
        weight_new = 1 - weight_old
        avg_embedding = old_embedding * weight_old + new_embedding * weight_new
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        # Update in database
        self.collection.update(
            ids=[existing_id],
            embeddings=[avg_embedding.tolist()],
            metadatas=[new_metadata],
        )

        # Update point cloud
        self.point_clouds[existing_id] = merged_cloud

        return existing_id

    def query_objects(self, query_text: str, clip_model=None) -> List[Dict]:
        """Query objects in database using text query."""
        if self.collection.count() == 0:
            return []

        # Get all objects
        results = self.collection.get(include=["metadatas", "embeddings"])
        query_results = []

        # Encode query text if CLIP model provided
        if clip_model is not None:
            query_embedding = clip_model.encode_text(query_text)

        # Get all IDs from collection
        all_ids = [str(i) for i in range(self.collection.count())]

        for i, (metadata, embedding) in enumerate(
            zip(results["metadatas"], results["embeddings"])
        ):
            if metadata is None or any(k.startswith("dummy") for k in metadata.keys()):
                continue

            # Get current ID
            current_id = all_ids[i]

            # Compute similarity if CLIP model provided
            similarity = 0.0
            if clip_model is not None:
                similarity = self._compute_semantic_similarity(
                    np.array(embedding), query_embedding
                )

            # Reconstruct nested metadata structure
            processed_metadata = {}
            depth_info = {}

            for k, v in metadata.items():
                if k.startswith("depth_info_"):
                    # Extract depth info fields
                    depth_key = k.replace("depth_info_", "")
                    try:
                        depth_info[depth_key] = float(v)
                    except ValueError:
                        depth_info[depth_key] = v
                else:
                    try:
                        processed_metadata[k] = float(v)
                    except ValueError:
                        processed_metadata[k] = v

            if depth_info:
                processed_metadata["depth_info"] = depth_info

            # Get point cloud if available
            point_cloud = self.point_clouds.get(current_id, None)

            # Only add results with valid point clouds
            if point_cloud is not None and len(point_cloud) > 0:
                query_results.append(
                    {
                        "similarity": similarity,
                        "metadata": processed_metadata,
                        "embedding": np.array(embedding),
                        "point_cloud": point_cloud,
                    }
                )

        # Sort results by similarity
        query_results.sort(key=lambda x: x["similarity"], reverse=True)
        return query_results

    def _compute_semantic_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        normalize_output: bool = True,
    ) -> float:
        """Compute cosine similarity between embeddings."""

        # Convert to numpy arrays if needed
        if isinstance(embedding1, torch.Tensor):
            embedding1 = embedding1.detach().cpu().numpy()
        if isinstance(embedding2, torch.Tensor):
            embedding2 = embedding2.detach().cpu().numpy()

        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)

        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        embedding1 = embedding1 / norm1
        embedding2 = embedding2 / norm2

        # Compute cosine similarity
        similarity = float(np.dot(embedding1, embedding2))

        # Normalize from [-1,1] to [0,1] if requested
        if normalize_output:
            similarity = (similarity + 1) / 2

        return similarity

    def initialize_collection(self):  # TODO: Fix this to not use dummy vectors
        """Initialize collection with dummy vectors."""
        if self.collection.count() == 0:
            dummy_vectors = np.zeros((3, 512))
            self.collection.add(
                embeddings=dummy_vectors.tolist(),
                ids=["dummy_1", "dummy_2", "dummy_3"],
                metadatas=[{"dummy": "1"}, {"dummy": "2"}, {"dummy": "3"}],
            )

    def update_query(self, query_embedding: Optional[np.ndarray]):
        """Update the query embedding used for similarity computations."""
        self.query_embedding = query_embedding

    def save_point_cloud_ply(self, output_path: str):
        """
        Save the current point clouds with similarity colors to a PLY file.

        Args:
            output_path (str): Path where to save the PLY file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Get all objects
            results = self.collection.get(include=["metadatas", "embeddings"])

            if not results or self.query_embedding is None:
                # If no query, use random colors for each object
                return self._save_with_random_colors(output_path, results)

            # Collect all points and their similarities
            all_points = []
            all_similarities = []

            # Get all IDs from collection
            all_ids = [str(i) for i in range(self.collection.count())]

            for i, (metadata, embedding) in enumerate(
                zip(results["metadatas"], results["embeddings"])
            ):
                if metadata is None or any(
                    k.startswith("dummy") for k in metadata.keys()
                ):
                    continue

                # Get current ID
                current_id = all_ids[i]

                # Get point cloud
                point_cloud = self.point_clouds.get(current_id)
                if point_cloud is not None and len(point_cloud) > 0:
                    # Compute similarity with query embedding
                    similarity = self._compute_semantic_similarity(
                        np.array(embedding), self.query_embedding
                    )

                    all_points.extend(point_cloud)
                    all_similarities.extend([similarity] * len(point_cloud))

            if all_points:
                # Convert to numpy arrays
                all_points = np.array(all_points)
                all_similarities = np.array(all_similarities)

                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(all_points)

                # Normalize similarities to [0, 1] range
                if (
                    len(np.unique(all_similarities)) > 1
                ):  # Only normalize if we have different values
                    min_similarity = np.min(all_similarities)
                    max_similarity = np.max(all_similarities)
                    normalized_similarities = (all_similarities - min_similarity) / (
                        max_similarity - min_similarity
                    )
                else:
                    normalized_similarities = all_similarities

                # Map similarity to black-to-red (R=normalized similarity, G=0, B=0)
                colors = np.column_stack(
                    (
                        normalized_similarities,
                        np.zeros_like(normalized_similarities),
                        np.zeros_like(normalized_similarities),
                    )
                )
                pcd.colors = o3d.utility.Vector3dVector(colors)

                # Save to PLY file
                o3d.io.write_point_cloud(output_path, pcd)

        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            print(f"Error saving point cloud to PLY: {str(e)}")

    def _save_with_random_colors(self, output_path: str, results) -> None:
        """Save point cloud with random colors when no query is present."""
        all_points = []
        all_colors = []
        object_colors = {}

        # Get all IDs from collection
        all_ids = [str(i) for i in range(self.collection.count())]

        for i, metadata in enumerate(results["metadatas"]):
            if metadata is None or any(k.startswith("dummy") for k in metadata.keys()):
                continue

            current_id = all_ids[i]
            point_cloud = self.point_clouds.get(current_id)

            if point_cloud is not None and len(point_cloud) > 0:
                # Generate random color for this object if not already assigned
                if current_id not in object_colors:
                    hue = random.random()
                    object_colors[current_id] = tuple(
                        c / 255 for c in self._hsv_to_rgb(hue, 0.8, 0.8)
                    )

                color = np.array([object_colors[current_id]] * len(point_cloud))
                all_points.extend(point_cloud)
                all_colors.extend(color)

        if all_points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
            o3d.io.write_point_cloud(output_path, pcd)

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """
        Convert HSV color to RGB color.

        Args:
            h (float): Hue (0-1)
            s (float): Saturation (0-1)
            v (float): Value (0-1)

        Returns:
            Tuple[float, float, float]: RGB color values (0-255)
        """
        if s == 0.0:
            return v, v, v

        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6

        if i == 0:
            return v * 255, t * 255, p * 255
        if i == 1:
            return q * 255, v * 255, p * 255
        if i == 2:
            return p * 255, v * 255, t * 255
        if i == 3:
            return p * 255, q * 255, v * 255
        if i == 4:
            return t * 255, p * 255, v * 255
        if i == 5:
            return v * 255, p * 255, q * 255

    def update_caption_async(self, object_id: str, caption_future: Future):
        """Store pending caption future and set up callback for when it completes."""
        self.pending_captions[object_id] = caption_future

        def _update_caption(future):
            try:
                caption = future.result()
                if caption:
                    # Get existing metadata
                    results = self.collection.get(
                        ids=[object_id], include=["metadatas"]
                    )
                    metadata = results["metadatas"][0]

                    # Update caption
                    metadata["caption"] = caption

                    # Update in database
                    self.collection.update(
                        ids=[object_id],
                        metadatas=[metadata],
                    )
            except Exception as e:
                print(f"Error updating caption for object {object_id}: {e}")
            finally:
                # Remove from pending captions
                self.pending_captions.pop(object_id, None)

        caption_future.add_done_callback(_update_caption)

    def get_object_metadata(self, object_id: str) -> Dict:
        """Get metadata for a specific object."""
        try:
            results = self.collection.get(ids=[object_id], include=["metadatas"])
            if results and results["metadatas"]:
                return results["metadatas"][0]
        except Exception as e:
            print(f"Error getting metadata for object {object_id}: {e}")
        return {}
