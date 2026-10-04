********************************************************
How to run streamlit run stock_vision_app_fixed_final.py
********************************************************
# Stock Vision System (SVS)

**Retail shelf monitoring with YOLO11n, edge inference, and a live dashboard.**

Stock Vision System (SVS) is a computer vision project that detects empty and occupied slots on retail shelves. It connects camera-based monitoring with an edge-to-cloud pipeline to help staff identify shelf gaps and prioritize replenishment.

## Overview

Manual shelf inspection can be time-consuming, and empty slots may go unnoticed between checks. SVS supports this process by capturing shelf images every **5 seconds**, running object detection on an edge device, and displaying the latest results on a web dashboard.

The system uses a **custom-trained YOLO11n model**, saved as `best.pt`, to detect two classes:

- `Occupied_Slot`: a shelf slot containing a product.
- `Empty_Slot`: an empty shelf slot.

The project focuses on **visible shelf availability**. It does not directly measure total inventory in storage.

## Features

- **Shelf-slot detection** — Detect empty and occupied areas on retail shelves.
- **Visual results** — Show bounding boxes for detected shelf slots.
- **Periodic image capture** — Process camera images at 5-second intervals.
- **Edge inference** — Run the detection model locally.
- **Edge-to-cloud pipeline** — Send detection results to a cloud/database layer.
- **Live dashboard** — Display the latest shelf status through a Streamlit interface.
- **Cached model loading** — Use `st.cache_resource` to reuse the loaded model across Streamlit reruns.

## System Workflow

1. **Capture:** The camera captures an image of the shelf every 5 seconds.
2. **Detect:** The edge device runs YOLO11n inference using `best.pt`.
3. **Visualize:** Detection results are displayed with bounding boxes.
4. **Transmit:** Results are sent to the cloud/database layer.
5. **Monitor:** The dashboard displays the latest shelf availability.

## Technology

| Component | Technology |
| --- | --- |
| Programming language | Python |
| Object detection model | YOLO11n |
| Model loading and inference | Ultralytics |
| Dataset annotation | Roboflow |
| Web dashboard | Streamlit |
| Inference approach | Edge processing |
| Data pipeline | Edge-to-cloud IoT |

## Dataset

Shelf images are annotated in **Roboflow** using bounding boxes for the two detection classes.

| Class | Description |
| --- | --- |
| `Occupied_Slot` | A shelf slot containing a product |
| `Empty_Slot` | An empty shelf slot |

The dataset is divided into the following subsets:

| Subset | Proportion | Purpose |
| --- | --- | --- |
| Training | 70% | Train the model |
| Validation | 20% | Evaluate the model during development |
| Testing | 10% | Evaluate the trained model on held-out images |

## Model and Weights

The primary detection model is **YOLO11n**, fine-tuned for shelf-slot detection. Its trained weights are stored in `best.pt`.

| File | Role |
| --- | --- |
| `best.pt` | Custom-trained YOLO11n weights used for shelf-slot detection |
| `yolov8n.pt` | Backup model referenced by the current loading code |

### Model Loading Behavior

The current application tries to load model files in this order:

1. `best.pt`
2. `best_s.pt`
3. `yolov8n.pt`

The first existing file that loads successfully is used. The repository currently contains `best.pt` and `yolov8n.pt`; `best_s.pt` is a remaining reference in the loading code.

When `best.pt` loads successfully, the application uses the custom-trained **YOLO11n** model.

**Fallback limitation:** Standard YOLOv8n weights are not trained on the project's `Empty_Slot` and `Occupied_Slot` classes. Loading `yolov8n.pt` may allow the application to continue running, but it does not provide equivalent shelf-slot detection. Correct SVS operation requires the custom-trained weights.

## Model Selection Background

The initial project requirements specified **YOLOv8n**. During implementation, the model was changed to **YOLO11n** after difficulties with the original setup.

The original requirements document was not updated to reflect this change. This README documents the implemented model: **YOLO11n with `best.pt`**.

## Running the Project

The application requires:

- Python and the project's dependencies.
- The custom-trained `best.pt` model.
- A configured image or camera input.
- Cloud/database configuration for the connected monitoring pipeline.

Place `best.pt` where the application's model loader can find it. The current loader uses relative filenames, so model discovery depends on the working directory from which the application is launched.

Repository-specific installation commands, configuration steps, and the Streamlit entry-point filename still need to be documented.

## Limitations

- Detection quality depends on lighting, camera angle, occlusion, and similarity between training images and deployment conditions.
- An occupied shelf slot does not reveal the number of products behind the visible item.
- An empty shelf slot does not necessarily mean the product is unavailable in storage.
- The 5-second capture interval introduces a delay between a shelf change and its next detection.
- Falling back to a general-purpose model does not preserve the custom shelf-detection capability.

## Evaluation

The project uses a held-out test subset for model evaluation. Numerical results are not included in this README because the evaluation outputs have not yet been verified.

Reported results should identify the metric used, such as precision, recall, mAP@50, or mAP@50–95, alongside the evaluated model and dataset version.

## Future Improvements

- Stop inference with a clear error when custom shelf-detection weights cannot be loaded.
- Remove unused model-path references.
- Add verified model evaluation results.
- Document reproducible installation and deployment steps.
- Add dashboard screenshots and example detection outputs.
