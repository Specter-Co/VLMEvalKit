"""Specter Alert Dataset for evaluating safety/compliance detection."""
from ..smp import *
import re
import glob
import json
import logging
import os
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image


class SpecterAlertDataset:
    """
    Specter Alert Dataset for evaluating safety/compliance detection.

    This dataset REQUIRES processor-wrapped VLMs. The dataset provides pre-extracted
    frames along with config_name and metadata. The VLM wrapper (ProcessorCloudVLM)
    is responsible for:
    1. Instantiating the EventProcessor from config_name using Hydra
    2. Loading detection metadata from local files
    3. Calling processor.process_event() with frames to generate the VLM prompt

    Expected data format:
    - TSV with columns: index, video_path, metadata_path, rule_definition_path, answer
    - video_path points to either:
      - a frames directory with frame_{timestamp}.jpg files, or
      - a video file (e.g. video.mp4)
    - metadata_path points to metadata.json with:
      - sample metadata (ground_truth, feature_type, etc.)
      - detection_metadata_paths: paths to per-frame detection JSONs
    - rule_definition_path points to rule_definition.json with:
      - prompt_config or processor: config name for EventProcessor (e.g. "fire_event_processor")
      - prompt_version: version tag for specter_prompts (e.g. "0.0.28", or null for latest)
      - processor_type: type of processor (e.g. "detectFire")
      - conditions: rule conditions for the processor
    - Optional: vlm.json in clip directory with prompt_text as fallback question

    The dataset evaluates VLM predictions by extracting answers from <answer> tags
    and comparing to ground truth (yes/no).
    """

    MODALITY = 'VIDEO'
    TYPE = 'Video-VQA'

    def __init__(self, dataset='SpecterAlert', nframe=8, fps=-1):
        self.dataset_name = dataset
        self.nframe = nframe
        self.fps = fps

        ret = self.prepare_dataset(dataset)
        assert ret is not None

        self.data_root = ret['root']
        self.data_file = ret['data_file']
        self.data = load(self.data_file)

        if 'index' not in self.data:
            self.data['index'] = np.arange(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < len(self.data)
        return dict(self.data.iloc[idx])

    @classmethod
    def supported_datasets(cls):
        return ['SpecterAlert']

    def prepare_dataset(self, dataset_name='SpecterAlert'):
        """
        Load dataset from local path.
        Expects:
        - TSV file at {LMUDataRoot()}/{dataset_name}.tsv
        - Sample clips at {LMUDataRoot()}/{dataset_name}/clips/{sample_id}/
          - frames/frame_{timestamp}.jpg
          - detections/frame_{timestamp}.json
          - metadata.json
          - video.mp4
          - rule_definition.json
          - vlm.json (optional)
        """
        lmu_root = LMUDataRoot()
        data_file = osp.join(lmu_root, f'{dataset_name}.tsv')
        dataset_root = osp.join(lmu_root, dataset_name)

        if not osp.exists(data_file):
            raise FileNotFoundError(f"Dataset TSV not found: {data_file}")
        if not osp.exists(dataset_root):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

        return dict(root=dataset_root, data_file=data_file)

    def _resolve_path(self, path_value: str) -> str:
        if not path_value:
            return ""
        if osp.isabs(path_value):
            return path_value
        dataset_prefix = f"{self.dataset_name}{osp.sep}"
        if path_value.startswith(dataset_prefix):
            lmu_root = osp.dirname(self.data_root)
            return osp.join(lmu_root, path_value)
        return osp.join(self.data_root, path_value)

    def _load_frame_paths_from_dir(self, frame_dir: str) -> List[str]:
        """Load pre-extracted frame paths from a directory."""
        if not osp.exists(frame_dir):
            raise FileNotFoundError(f"Frames directory not found: {frame_dir}")

        frame_paths = sorted(glob.glob(osp.join(frame_dir, 'frame_*.jpg')))
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in: {frame_dir}")

        return frame_paths

    def _load_detection_paths_from_metadata(self, metadata_path: str) -> List[str]:
        """Load local detection file paths from metadata.json."""
        if not metadata_path or not osp.exists(metadata_path):
            return []

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        detection_paths = metadata.get('detection_metadata_paths', []) or []
        resolved = []
        metadata_dir = osp.dirname(metadata_path)
        for path_value in detection_paths:
            if not path_value:
                resolved.append("")
                continue
            if osp.isabs(path_value):
                if osp.exists(path_value):
                    resolved.append(path_value)
                else:
                    fallback = osp.join(metadata_dir, 'detections', osp.basename(path_value))
                    resolved.append(fallback)
            else:
                resolved.append(osp.join(metadata_dir, path_value))

        return resolved

    def _fallback_detection_paths(
        self,
        frame_paths: List[str],
        metadata_path: str,
    ) -> List[str]:
        """Fallback to detections/ folder when metadata omits detection paths."""
        if not frame_paths:
            return []
        metadata_dir = osp.dirname(metadata_path) if metadata_path else ""
        detections_dir = osp.join(metadata_dir, "detections") if metadata_dir else ""
        if not detections_dir or not osp.exists(detections_dir):
            return []

        detection_paths = []
        for frame_path in frame_paths:
            frame_name = osp.basename(frame_path)
            detection_name = osp.splitext(frame_name)[0] + ".json"
            detection_path = osp.join(detections_dir, detection_name)
            detection_paths.append(detection_path)
        return detection_paths

    def _sample_video_frames(self, video_path: str) -> List[str]:
        """Sample frames from a video file and return saved frame paths."""
        try:
            from decord import VideoReader, cpu
        except Exception as exc:
            logging.warning(f"Failed to import decord for video sampling: {exc}")
            return []

        if self.fps > 0 and self.nframe > 0:
            raise ValueError('fps and nframe should not be set at the same time')

        try:
            vid = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        except Exception as exc:
            logging.warning(f"Failed to open video {video_path}: {exc}")
            return []

        total_frames = len(vid)
        if total_frames == 0:
            logging.warning(f"Video has zero frames: {video_path}")
            return []

        if self.fps > 0:
            video_fps = vid.get_avg_fps()
            total_duration = total_frames / video_fps if video_fps else 0
            required_frames = int(total_duration * self.fps) if total_duration > 0 else 0
            if required_frames <= 0:
                logging.warning(f"Computed zero frames for video: {video_path}")
                return []
            step_size = video_fps / self.fps if self.fps else 1
            indices = [int(i * step_size) for i in range(required_frames)]
        elif self.nframe > 0:
            step_size = total_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
        else:
            indices = list(range(total_frames))

        indices = [i for i in indices if 0 <= i < total_frames]
        if not indices:
            logging.warning(f"No valid frame indices for video: {video_path}")
            return []

        frames_dir = osp.join(osp.dirname(video_path), 'frames_sampled')
        os.makedirs(frames_dir, exist_ok=True)
        frame_paths = [osp.join(frames_dir, f'frame_{idx}.jpg') for idx in indices]

        # Skip if already sampled
        if np.all([osp.exists(p) for p in frame_paths]):
            return frame_paths

        try:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)
        except Exception as exc:
            logging.warning(f"Failed to sample frames from video {video_path}: {exc}")
            return []

        return frame_paths

    def build_prompt(self, line, video_llm=False):
        """Build prompt for processor-wrapped inference.

        This method builds a message containing frames or video plus metadata
        for ProcessorCloudVLM wrapper. The wrapper is responsible for:
        1. Instantiating EventProcessor from config_name using Hydra
        2. Loading detections from local files
        3. Calling processor.process_event() with frames to generate VLM prompt

        Args:
            line: Row from TSV or index
            video_llm: Not used (frames are pre-extracted)

        Returns:
            List of message dicts with types:
            - 'image' or 'video': Frame file paths or a video file
            - 'config_name': EventProcessor config name for Hydra instantiation
            - 'original_question': Fallback prompt text if config_name is empty
            - 'detection_paths': Local paths to detection metadata JSONs
            - 'rule_definition_path': Path to rule_definition.json (optional)
            - 'prompt_version': Version of specter_prompts to use (optional)
        """
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video_path = self._resolve_path(line.get('video_path', ''))
        metadata_path = self._resolve_path(line.get('metadata_path', ''))
        rule_definition_path = self._resolve_path(line.get('rule_definition_path', ''))

        # Load rule_definition.json for prompt_config and prompt_version
        rule_definition = {}
        if rule_definition_path and osp.exists(rule_definition_path):
            with open(rule_definition_path, 'r') as f:
                rule_definition = json.load(f)

        # Handle both old "processor" field and new "prompt_config" field
        config_name = rule_definition.get('prompt_config') or rule_definition.get('processor', '')
        # prompt_version can be null (use latest) or a version string
        # Convert None to empty string for consistency
        prompt_version = rule_definition.get('prompt_version') or ''
        
        # Load original_question from vlm.json if available, otherwise from rule_definition
        original_question = rule_definition.get('prompt_text', '')
        if video_path:
            # Try to find vlm.json in the same directory as video_path
            if osp.isdir(video_path):
                vlm_json_path = osp.join(osp.dirname(video_path), 'vlm.json')
            else:
                vlm_json_path = osp.join(osp.dirname(video_path), 'vlm.json')
            
            if osp.exists(vlm_json_path):
                try:
                    with open(vlm_json_path, 'r') as f:
                        vlm_data = json.load(f)
                    # Use prompt_text from vlm.json as original_question
                    if 'prompt_text' in vlm_data and vlm_data['prompt_text']:
                        original_question = vlm_data['prompt_text']
                except (OSError, json.JSONDecodeError):
                    pass

        # Load detection paths from metadata.json (fallback to detections/ folder)
        detection_paths = self._load_detection_paths_from_metadata(metadata_path)

        # Build message for ProcessorCloudVLM wrapper
        message = []
        if video_path and osp.isdir(video_path):
            frame_paths = self._load_frame_paths_from_dir(video_path)
            message.extend([dict(type='image', value=frame) for frame in frame_paths])
            logging.info(f"SpecterAlert: using {len(frame_paths)} frames from dir {video_path}")
        elif video_path:
            frame_paths = self._sample_video_frames(video_path)
            message.extend([dict(type='image', value=frame) for frame in frame_paths])
            logging.info(f"SpecterAlert: sampled {len(frame_paths)} frames from video {video_path}")
        else:
            frame_paths = []

        if not detection_paths and frame_paths:
            detection_paths = self._fallback_detection_paths(frame_paths, metadata_path)
            if detection_paths:
                logging.info(
                    f"SpecterAlert: using {len(detection_paths)} detections from folder "
                    f"for {video_path or metadata_path}"
                )

        message.append(dict(type='config_name', value=config_name))
        message.append(dict(type='original_question', value=original_question))
        message.append(dict(type='detection_paths', value=detection_paths))
        if rule_definition_path:
            message.append(dict(type='rule_definition_path', value=rule_definition_path))
        if prompt_version:
            message.append(dict(type='prompt_version', value=prompt_version))

        return message

    @staticmethod
    def extract_answer(text):
        """Extract yes/no/unknown from model response."""
        if not text:
            return 'unknown'

        text_lower = text.lower().strip()

        # First try: look for <answer> tags
        match = re.search(r'<answer>\s*(yes|no|unknown)\s*</answer>', text_lower)
        if match:
            return match.group(1)

        # Second try: strip punctuation and check if response is just yes/no
        text_clean = re.sub(r'[^\w\s]', '', text_lower).strip()
        if text_clean in ['yes', 'no', 'unknown']:
            return text_clean

        # Third try: look for yes/no anywhere in a short response (< 20 chars)
        if len(text_clean) < 20:
            if 'yes' in text_clean:
                return 'yes'
            elif 'no' in text_clean:
                return 'no'

        # Fourth try: check last word after stripping punctuation
        words = text_clean.split()
        if words:
            last_word = words[-1]
            if last_word in ['yes', 'no', 'unknown']:
                return last_word

        return 'unknown'

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        Evaluate predictions against ground truth using simple yes/no matching.

        Extracts answers from <answer> tags in predictions and compares to
        the ground truth answer column. Unknown predictions are treated as "no".
        """
        from ..smp import load, dump
        from ..smp.file import get_intermediate_file_path

        data = load(eval_file)

        # Confusion matrix counts
        tp = 0  # predicted yes, actual yes
        fp = 0  # predicted yes, actual no
        tn = 0  # predicted no, actual no
        fn = 0  # predicted no, actual yes

        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            gt = str(row.get('answer', '')).lower().strip()

            pred = cls.extract_answer(pred_text)

            # Treat "unknown" as "no"
            if pred == 'unknown':
                pred = 'no'

            # Only count rows with valid ground truth (yes/no)
            if gt in ['yes', 'no']:
                if pred == 'yes' and gt == 'yes':
                    tp += 1
                elif pred == 'yes' and gt == 'no':
                    fp += 1
                elif pred == 'no' and gt == 'no':
                    tn += 1
                elif pred == 'no' and gt == 'yes':
                    fn += 1

            # Store extracted prediction back for analysis
            data.at[idx, 'extracted_pred'] = pred
            data.at[idx, 'hit'] = int(pred == gt) if gt in ['yes', 'no'] else -1

        total = tp + fp + tn + fn
        correct = tp + tn
        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        score_dict = {
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2),
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'total': total
        }

        # Save detailed results
        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(score_dict, score_file)

        # Also save the detailed data with extracted predictions
        detail_file = get_intermediate_file_path(eval_file, '_detail')
        dump(data, detail_file)

        return score_dict
