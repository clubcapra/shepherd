"""
Shepherd class.
"""

from asyncio.queues import Queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import asyncio 
import cv2
import logging
import numpy as np
import torch

from .database_wrapper import DatabaseWrapper
from .models.implementations import BLIP, CLIP, DAN, SAM, YOLO
from .shepherd_config import ShepherdConfig
from .data_structure import ResultsCollections, ResultsType
from .utils.visualization import VisualizationUtils as vu
from .utils.wrapper import timer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Shepherd:
    """
    Shepherd class that handles the whole vision pipeline.
    """

    def __init__(self, config: ShepherdConfig = None, database: DatabaseWrapper = None):
        """
        Initialize the Shepherd class with all required models and configurations.

        Args:
            config (ShepherdConfig): Configuration containing model paths and parameters
            database (DatabaseWrapper): Database wrapper instance for storing object data
        """
        if config is None:
            config = ShepherdConfig()

        if database is None:
            database = DatabaseWrapper(camera_utils=config.camera)

        self.config = config
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.database = database

        # Initialize models
        self.detector = YOLO(
            model_path=config.get("model_paths.yolo", "yolov8s-world.pt"),
            device=self.device,
            confidence_threshold=config.get("thresholds.detection", 0.4),
            nms_threshold=config.get("thresholds.nms", 0.45),
        )

        self.segmenter = SAM(
            model_path=config.get("model_paths.sam", "FastSAM-s.pt"),
            device=self.device,
            points_per_side=config.get("sam.points_per_side", 32),
            pred_iou_thresh=config.get("sam.pred_iou_thresh", 0.88),
        )

        self.captioner = BLIP(
            model_path=config.get(
                "model_paths.blip", "Salesforce/blip-image-captioning-base"
            ),
            device=self.device,
        )

        self.embedder = CLIP(
            model_path=config.get("model_paths.clip", "ViT-B/32"), device=self.device
        )

        if config.get("use_depth", True):
            self.depth_estimator = DAN(
                model_path=config.get("model_paths.dan"), device=self.device
            )
        else:
            self.depth_estimator = None

        self._validate_models()

        # Initialize query embedding
        self.query_embedding = None
        if self.config.default_query:
            self.update_query(self.config.default_query)

        self.frame_queue = Queue(40)
        self.results = ResultsCollections()
        self.max_concurrent_tasks = 4
        self.yolo_dan_batch_size = 16
        self.max_frame_delay = 8 

    async def _complete_pending_tasks(self, tasks: set, return_when=asyncio.FIRST_COMPLETED) -> set:
        """Helper to wait for tasks, process results, and return remaining tasks."""
        if not tasks:
            return tasks
        done, pending = await asyncio.wait(tasks, return_when=return_when, timeout=0.1) # Add timeout
        for task in done:
            try:
                result = task.result()
                if result is not None and isinstance(result, tuple) and len(result) == 3:
                    frame_id, frame, results_list = result
                    # Update ResultsCollections with the processed data
                    self.results.update(frame_id, ResultsType.ALL, [frame, results_list])
            except asyncio.CancelledError:
                 logger.info("A processing task was cancelled.")
            except Exception:
                logger.exception("Error retrieving result from completed task.")
        return pending

    async def run(self):
        while True:
            tasks = set()
            results = []
            queue_size = self.frame_queue.qsize()
            n_concurrent_tasks = min(self.max_concurrent_tasks, queue_size)

            if n_concurrent_tasks == 0:
                await asyncio.sleep(1.0)
                continue

            # if we have enough frames, run detection and depth estimation in batch
            if queue_size >= self.yolo_dan_batch_size:
                yolo_dan_batch_frames = []
                yolo_dan_batch_depths = []
                inputs = []
                for i in range(self.yolo_dan_batch_size):
                    if self.frame_queue.empty():
                        break
                    frame_id, rgb, depth, camera_pose = await self.frame_queue.get()
                    inputs.append((frame_id, rgb, depth, camera_pose))
                    yolo_dan_batch_frames.append(rgb)
                    yolo_dan_batch_depths.append(depth)

                # Run detection and depth estimation in parallel
                detections, depth_frames = await asyncio.to_thread(
                    self.step_detection_dan, yolo_dan_batch_frames, yolo_dan_batch_depths,
                    name="BatchYOLO-DAN",
                )
                i = 0
                while i < len(inputs):
                    for j in range(n_concurrent_tasks):
                        if i >= len(inputs):
                            break
                        frame_id, rgb, depth, camera_pose = inputs[i]
                        i += 1
                        task = asyncio.create_task(
                            self.process_objects(
                                detections[j], depth_frames[j], rgb, camera_pose, frame_id
                            ),
                            name=f"ProcessObjects-{frame_id}",
                        )
                        tasks.add(task)
                    await self._complete_pending_tasks(tasks)
            else:
                for i in range(n_concurrent_tasks):
                    frame_id, rgb, depth, camera_pose = await self.frame_queue.get()
                    task = asyncio.create_task(self.process_frame(frame_id, rgb, depth, camera_pose), name=f"ProcessFrame-{frame_id}")
                    tasks.add(task)
                await self._complete_pending_tasks(tasks)

    async def add_to_frame_queue(self, frame_id, rgb, depth, camera_pose):
        if self.frame_queue.full():
            dispose_frame = await self.frame_queue.get()
        self.results.add_frame(frame_id, rgb, depth, camera_pose)
        await self.frame_queue.put([frame_id, rgb, depth, camera_pose])



    async def get_latest_results(self, frame_id):
        for result_type in [ResultsType.ALL, ResultsType.DETECTIONS]:
            last_result_id = self.results.latest_results_id[result_type]
            if last_result_id != -1 and last_result_id + self.max_frame_delay >= frame_id:
                if self.results[last_result_id] is not None:
                    return last_result_id, self.results[last_result_id]
        else:
            for i in range(self.max_frame_delay, 0, -1):
                current_frame_id = frame_id - i + 1
                if current_frame_id < 0:
                    continue
                if self.results[current_frame_id] is not None:
                    result = self.results[current_frame_id]
                    return current_frame_id, result
        return -1, None
        

    def update_query(self, query_text: str):
        """Update the query and compute its embedding."""
        self.config.default_query = query_text
        if query_text:
            # Convert query embedding to numpy array
            query_embedding = self.embedder.encode_text(query_text)
            if isinstance(query_embedding, str):
                query_embedding = None
            else:
                # Ensure it's a numpy array and normalized
                query_embedding = np.array(query_embedding)
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
        else:
            query_embedding = None

        # Update query embedding in database
        self.database.update_query(query_embedding)
        self.query_embedding = query_embedding
    
    def step_detection_dan(self, frame: np.ndarray, depth_frame: np.ndarray):
        """Run detection and DAN depth estimation in parallel."""
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        # frame_tensor = torch.from_numpy(frame).to(self.device)
        """Process a single frame through the vision pipeline."""
        # Run detection
        with torch.cuda.stream(stream1):
            detections = self._process_detections(frame)

        # If no depth frame provided, estimate it using DAN
        with torch.cuda.stream(stream2):
            if depth_frame is None and self.depth_estimator is not None:
                depth_frame = self.depth_estimator.estimate_depth(frame)
                # Normalize depth values to a reasonable range (e.g. 0.1 to 10 meters)
                if depth_frame is not None:
                    depth_frame = np.clip(depth_frame, 0.1, 10.0)

        # Wait for both streams to finish
        torch.cuda.synchronize()

        return detections, depth_frame

    def _step_sam_clip(self, frame: np.ndarray, detections: List[Dict]):
        """Run SAM and CLIP in parallel."""
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        bbox_regions = []
        # Get bounding box regions
        for detection in detections:
            bbox = detection["bbox"]
            bbox_image = self._get_bbox_image(frame, bbox)
            bbox_regions.append(bbox_image)
        
        # Get segmentation masks for detections
        with torch.cuda.stream(stream1):
            masks = self._process_segments(frame, detections)

        # Get embeddings for masked regions
        with torch.cuda.stream(stream2):
            embeddings = [
                self.embedder.encode_image(
                        bbox_region
                    )
                for bbox_region in bbox_regions
            ]
        # Wait for both streams to finish
        torch.cuda.synchronize()

        return masks, embeddings, bbox_regions

    def _step_point_clouds(self, masks: List[np.ndarray], depth_frame: np.ndarray):
        point_clouds = []
        for mask in masks:
            point_cloud = self._create_point_cloud(mask, depth_frame)
            if point_cloud is not None:
                if len(point_cloud) > 10:
                    point_clouds.append(point_cloud)
                else:
                    point_clouds.append(None)
        
        return point_clouds

    def _step_add_to_db(self, obj, camera_pose, depth_frame):
        results = [None] * len(obj["detections"])
        with ThreadPoolExecutor(4) as executor:
            for i in range(len(obj["detections"])):
                detection = obj["detections"][i]
                mask = obj["masks"][i]
                embedding = obj["embeddings"][i]
                bbox_region = obj["bbox_regions"][i]
                point_cloud = obj["point_clouds"][i]
            
                if point_cloud is not None:
                    # Create metadata
                    metadata = {
                        "class_id": detection.get("class_id", 0),
                        "confidence": detection["confidence"],
                    }
            
                    # Store in database and get object ID
                    object_id, needs_caption = self.database.store_object(
                        embedding=embedding,
                        metadata=metadata,
                        point_cloud=point_cloud,
                        camera_pose=camera_pose,
                    )
            
                    if object_id is not None:
                        # Submit captioning task if needed
            
                        if needs_caption and self.config.use_caption:
                            future = executor.submit(
                                self._process_captions, bbox_region
                            )
                            self.database.update_caption_async(object_id, future)
            
                        # Get object metadata including caption
                        obj_metadata = self.database.get_object_metadata(object_id)
                        caption = obj_metadata.get("caption", "Processing caption...")
                        print(f"Object {object_id}: {caption}")
            
                        # Get similarity from metadata if it exists
                        similarity = metadata.get("query_similarity")
            
                        results[i] ={
                                "detection": detection,
                                "mask": mask,
                                "embedding": embedding,
                                "depth_frame": depth_frame,
                                "object_id": object_id,
                                "similarity": similarity,
                                "caption": caption,
                            }
        
        results = [r for r in results if r is not None]
        return results

    async def process_objects(self, detections, depth_frame, frame, camera_pose, frame_id):
        obj = {
            "frame": frame,
            "detections": detections,
            "depth_frame": depth_frame,
        }
        
        # Get segmentation masks and embeddings
        masks, embeddings, bbox_regions = self._step_sam_clip(frame, detections)
        
        obj["masks"] = masks
        obj["embeddings"] = embeddings
        obj["bbox_regions"] = bbox_regions
        
        
        # Get depth information and create point clouds
        point_clouds = await asyncio.to_thread(self._step_point_clouds, masks, depth_frame)
        obj["point_clouds"] = point_clouds
        
        # Only process if we have valid point cloud data
        
        results = self._step_add_to_db(obj, camera_pose, depth_frame)
        print(f"Processed {len(results)} objects for frame id#{frame_id}.")
        return frame_id, frame, results

    async def process_frame(
        self,
        frame_id: int,
        frame: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        camera_pose: Optional[Dict] = None,):

        # detections, depth_frame = await asyncio.to_thread(self._step_detection_dan, frame, depth_frame)
        detections, depth_frame = self.step_detection_dan(frame, depth_frame)
        self.results.update(frame_id, ResultsType.DETECTIONS, [frame, detections])
        
        return await self.process_objects(detections, depth_frame, frame, camera_pose, frame_id)

    def _validate_models(self):
        """Validate that all required models are properly initialized."""
        required_models = ["detector", "segmenter", "captioner", "embedder"]
        for model_name in required_models:
            if not hasattr(self, model_name) or getattr(self, model_name) is None:
                raise ValueError(
                    f"Required model {model_name} is not properly initialized"
                )

    def _process_detections(self, image: np.ndarray) -> List[Dict]:
        """Process image through detection model."""
        return self.detector.detect(image)

    def _process_segments(
        self, image: np.ndarray, detections: List[Dict]
    ) -> List[np.ndarray]:
        """Process image through segmentation model."""
        return self.segmenter.segment(image, detections)

    def _process_captions(self, image: np.ndarray) -> str:
        """Generate captions for detected regions."""
        return self.captioner.generate_caption(image)

    def _get_masked_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract masked region from image."""
        # Get bounding box from mask
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0 or len(x_coords) == 0:
            return image

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # Crop image to bounding box
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped

    def _get_bbox_image(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract masked region from image."""
        # Get bounding box from mask
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        
        # Cast to int   
        x, y, w, h = int(x), int(y), int(w), int(h)
        # Crop image to bounding box
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    def _create_point_cloud(
        self, mask: np.ndarray, depth_frame: np.ndarray
    ) -> np.ndarray:
        """Create point cloud from mask and depth frame."""

        # Get image dimensions
        height, width = depth_frame.shape

        # Create meshgrid of pixel coordinates
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Apply mask
        valid_points = mask > 0

        # Get valid coordinates and depths
        x = x_coords[valid_points]
        y = y_coords[valid_points]
        z = depth_frame[valid_points]

        # Filter out invalid depths
        valid_depths = np.logical_and(z > 0.1, np.isfinite(z))
        x = x[valid_depths]
        y = y[valid_depths]
        z = z[valid_depths]

        if len(z) == 0:
            return np.array([])

        # Convert to 3D coordinates using camera parameters
        x_coords = (x - self.config.camera.cx) * z / self.config.camera.fx
        y_coords = (y - self.config.camera.cy) * z / self.config.camera.fy
        z_coords = z

        # Stack coordinates
        points = np.stack([x_coords, y_coords, z_coords], axis=1)
        torch._numpy.array(points)

        # Remove outliers
        if len(points) > 0:
            median = np.median(points, axis=0)
            mad = np.median(np.abs(points - median), axis=0)
            valid_points = np.all(np.abs(points - median) <= 3 * mad, axis=1)
            points = points[valid_points]

        return points
