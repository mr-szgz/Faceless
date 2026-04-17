## class `ultralytics.engine.results.Boxes`

```
Boxes(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None
```

**Bases:** `BaseTensor`

A class for managing and manipulating detection boxes.

This class provides comprehensive functionality for handling detection boxes, including their coordinates, confidence scores, class labels, and optional tracking IDs. It supports various box formats and offers methods for easy manipulation and conversion between different coordinate systems.

This class manages detection boxes, providing easy access and manipulation of box coordinates, confidence scores, class identifiers, and optional tracking IDs. It supports multiple formats for box coordinates, including both absolute and normalized forms.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `boxes` | `torch.Tensor \\| np.ndarray` | A tensor or numpy array with detection boxes of shape (num\\_boxes, 6) or (num\\_boxes, 7). Columns should contain \\[x1, y1, x2, y2, (optional) track\\_id, confidence, class\\]. | _required_ |
| `orig_shape` | `tuple[int, int]` | The original image shape as (height, width). Used for normalization. | _required_ |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `data` | `torch.Tensor \\| np.ndarray` | The raw tensor containing detection boxes and associated data. |
| `orig_shape` | `tuple[int, int]` | The original image dimensions (height, width). |
| `is_track` | `bool` | Indicates whether tracking IDs are included in the box data. |
| `xyxy` | `torch.Tensor \\| np.ndarray` | Boxes in \\[x1, y1, x2, y2\\] format. |
| `conf` | `torch.Tensor \\| np.ndarray` | Confidence scores for each box. |
| `cls` | `torch.Tensor \\| np.ndarray` | Class labels for each box. |
| `id` | `torch.Tensor \\| None` | Tracking IDs for each box (if available). |
| `xywh` | `torch.Tensor \\| np.ndarray` | Boxes in \\[x, y, width, height\\] format. |
| `xyxyn` | `torch.Tensor \\| np.ndarray` | Normalized \\[x1, y1, x2, y2\\] boxes relative to orig\\_shape. |
| `xywhn` | `torch.Tensor \\| np.ndarray` | Normalized \\[x, y, width, height\\] boxes relative to orig\\_shape. |

**Methods**

| Name | Description |
| --- | --- |
| [`xyxy`](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes.xyxy) | Return bounding boxes in \\[x1, y1, x2, y2\\] format. |
| [`conf`](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes.conf) | Return the confidence scores for each detection box. |
| [`cls`](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes.cls) | Return the class ID tensor representing category predictions for each bounding box. |
| [`id`](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes.id) | Return the tracking IDs for each detection box if available. |
| [`xywh`](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes.xywh) | Convert bounding boxes from \\[x1, y1, x2, y2\\] format to \\[x, y, width, height\\] format. |
| [`xyxyn`](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes.xyxyn) | Return normalized bounding box coordinates relative to the original image size. |
| [`xywhn`](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes.xywhn) | Return normalized bounding boxes in \\[x, y, width, height\\] format. |

**Examples**

```
>>> import torch
>>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
>>> orig_shape = (480, 640)  # height, width
>>> boxes = Boxes(boxes_data, orig_shape)
>>> print(boxes.xyxy)
>>> print(boxes.conf)
>>> print(boxes.cls)
>>> print(boxes.xywhn)
```

Source code in `ultralytics/engine/results.py`[View on GitHub](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/results.py#L825-L1019)

```
class Boxes(BaseTensor):
    """A class for managing and manipulating detection boxes.

    This class provides comprehensive functionality for handling detection boxes, including their coordinates,
    confidence scores, class labels, and optional tracking IDs. It supports various box formats and offers methods for
    easy manipulation and conversion between different coordinate systems.

    Attributes:
        data (torch.Tensor | np.ndarray): The raw tensor containing detection boxes and associated data.
        orig_shape (tuple[int, int]): The original image dimensions (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xyxy (torch.Tensor | np.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | np.ndarray): Confidence scores for each box.
        cls (torch.Tensor | np.ndarray): Class labels for each box.
        id (torch.Tensor | None): Tracking IDs for each box (if available).
        xywh (torch.Tensor | np.ndarray): Boxes in [x, y, width, height] format.
        xyxyn (torch.Tensor | np.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
        xywhn (torch.Tensor | np.ndarray): Normalized [x, y, width, height] boxes relative to orig_shape.

    Methods:
        cpu: Return a copy of the object with all tensors on CPU memory.
        numpy: Return a copy of the object with all tensors as numpy arrays.
        cuda: Return a copy of the object with all tensors on GPU memory.
        to: Return a copy of the object with tensors on specified device and dtype.

    Examples:
        >>> import torch
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # height, width
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
    """

    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """Initialize the Boxes class with detection box data and the original image shape.

        This class manages detection boxes, providing easy access and manipulation of box coordinates, confidence
        scores, class identifiers, and optional tracking IDs. It supports multiple formats for box coordinates,
        including both absolute and normalized forms.

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape (num_boxes, 6) or
                (num_boxes, 7). Columns should contain [x1, y1, x2, y2, (optional) track_id, confidence, class].
            orig_shape (tuple[int, int]): The original image shape as (height, width). Used for normalization.
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape
```
