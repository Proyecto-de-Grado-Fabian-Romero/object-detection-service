from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    def __init__(self):
        # max_age=5: tracks live 5 frames max (single-frame use, so low value is fine)
        # n_init=1: confirm track immediately on first detection
        # max_cosine_distance=0.4: tighter appearance matching reduces false ID merges
        # nms_max_overlap=0.7: suppress redundant detections fed to tracker
        self.tracker = DeepSort(
            max_age=5,
            n_init=1,
            max_cosine_distance=0.4,
            nms_max_overlap=0.7,
        )

    def update(self, img, detections):
        """
        img: complete image (np.ndarray)
        detections: dicts list with keys:
            'bbox' (xmin, ymin, xmax, ymax),
            'confidence',
            'class_id'
        Returns list with detection dicts with added ID:
            'bbox', 'confidence', 'class_id', 'track_id'
        """

        # Prepare detections for DeepSort [xmin, ymin, width, height, confidence]
        ds_detections = []
        for det in detections:
            bbox, confidence, class_id = det
            xmin, ymin, xmax, ymax = bbox
            width = xmax - xmin
            height = ymax - ymin
            bbox_xywh = [xmin, ymin, width, height]
            ds_detections.append([bbox_xywh, confidence, class_id])

        tracks = self.tracker.update_tracks(ds_detections, frame=img)

        results = []
        for track in tracks:
            # if not track.is_confirmed():
            # continue
            bbox = track.to_tlbr()  # top-left bottom-right

            results.append(
                {
                    "bbox": bbox,
                    "confidence": track.det_conf,
                    "class_id": track.det_class,
                    "track_id": track.track_id,
                }
            )
        return results
