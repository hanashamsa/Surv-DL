from pathlib import Path

import yaml
from clearml import Dataset, Task
from ultralytics import YOLO

from config.paths import YOLO_ROOT


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_NAME = "SGTS/hanas"
TASK_NAME = "warehouse_yolo11_baseline"

QUEUE_NAME = "gpu-48gb"

MODEL_NAME = "yolo11n.pt"

EPOCHS = 20
IMAGE_SIZE = 640
BATCH_SIZE = 16
WORKERS = 4
DEVICE = 0

RUN_NAME = "warehouse_yolo11"

DATASET_PROJECT = "SGTS/hanas"
DATASET_NAME = "warehouse_yolo_dataset"


# ============================================================
# CLEARML TASK
# ============================================================

task = Task.init(
    project_name=PROJECT_NAME,
    task_name=TASK_NAME,
    task_type="training",
)


# ============================================================
# PARAMETERS
# ============================================================

task.connect(
    {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "workers": WORKERS,
        "device": DEVICE,
        "dataset_name": DATASET_NAME,
    }
)


# ============================================================
# LOCAL DATASET VALIDATION
# ============================================================

def validate_local_dataset():
    """
    Check that the prepared YOLO dataset exists locally.
    """

    dataset_yaml = YOLO_ROOT / "dataset.yaml"

    if not YOLO_ROOT.exists():
        raise FileNotFoundError(
            f"YOLO dataset directory does not exist:\n{YOLO_ROOT}"
        )

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"dataset.yaml was not found:\n{dataset_yaml}"
        )

    train_images = YOLO_ROOT / "images" / "train"
    val_images = YOLO_ROOT / "images" / "val"

    train_labels = YOLO_ROOT / "labels" / "train"
    val_labels = YOLO_ROOT / "labels" / "val"

    required_directories = [
        train_images,
        val_images,
        train_labels,
        val_labels,
    ]

    for directory in required_directories:

        if not directory.exists():

            raise FileNotFoundError(
                f"Required dataset directory missing:\n{directory}"
            )

    print("=" * 60)
    print("LOCAL DATASET VALIDATION")
    print("=" * 60)

    print(f"Dataset: {YOLO_ROOT}")
    print(f"Train images: {train_images}")
    print(f"Validation images: {val_images}")

    print("[OK] Dataset structure looks valid.")


# ============================================================
# CREATE / UPLOAD CLEARML DATASET
# ============================================================

def upload_dataset():
    """
    Create a ClearML Dataset and upload the local YOLO dataset.
    """

    validate_local_dataset()

    print("=" * 60)
    print("CREATING CLEARML DATASET")
    print("=" * 60)

    dataset = Dataset.create(
        dataset_project=DATASET_PROJECT,
        dataset_name=DATASET_NAME,
        dataset_tags=["warehouse", "yolo11"],
        description="YOLO11 warehouse object detection dataset",
    )

    print(f"ClearML Dataset ID: {dataset.id}")

    print("=" * 60)
    print("ADDING LOCAL DATASET FILES")
    print("=" * 60)

    dataset.add_files(
        path=str(YOLO_ROOT),
        recursive=True,
        verbose=True,
    )

    print("=" * 60)
    print("UPLOADING DATASET")
    print("=" * 60)

    dataset.upload(
        show_progress=True,
        verbose=True,
    )

    print("=" * 60)
    print("FINALIZING DATASET")
    print("=" * 60)

    dataset.finalize(
        raise_on_error=True,
    )

    print("[OK] ClearML dataset finalized.")
    print(f"Dataset ID: {dataset.id}")

    return dataset.id


# ============================================================
# GET DATASET ID
# ============================================================

def get_dataset_id():
    """
    Get the ClearML Dataset ID.

    Locally we create/upload the dataset.
    Remotely we retrieve the dataset ID stored
    in the ClearML task parameters.
    """

    if task.running_locally():

        dataset_id = upload_dataset()

        # Store the dataset ID in ClearML so the
        # remote worker knows exactly which dataset
        # to download.
        task.set_parameter(
            name="dataset_id",
            value=dataset_id,
        )

        return dataset_id

    # --------------------------------------------------------
    # Remote execution
    # --------------------------------------------------------

    dataset_id = task.get_parameter(
        name="dataset_id",
        default=None,
    )

    if not dataset_id:

        raise RuntimeError(
            "ClearML Dataset ID was not found in task parameters."
        )

    print("=" * 60)
    print("REMOTE DATASET")
    print("=" * 60)

    print(f"Dataset ID: {dataset_id}")

    return dataset_id


# ============================================================
# DOWNLOAD DATASET ON GPU MACHINE
# ============================================================

def download_dataset(dataset_id):
    """
    Download the ClearML Dataset to the GPU machine.
    """

    print("=" * 60)
    print("DOWNLOADING CLEARML DATASET")
    print("=" * 60)

    dataset = Dataset.get(
        dataset_id=dataset_id,
        only_completed=True,
    )

    print(f"Dataset name: {dataset.name}")
    print(f"Dataset ID: {dataset.id}")

    dataset_path = dataset.get_local_copy(
        max_workers=8,
    )

    dataset_path = Path(dataset_path)

    print(f"Dataset downloaded to:")
    print(dataset_path)

    return dataset_path


# ============================================================
# FIND DATASET YAML
# ============================================================

def find_dataset_yaml(dataset_root):
    """
    Find dataset.yaml inside the downloaded dataset.
    """

    yaml_files = list(
        dataset_root.rglob("dataset.yaml")
    )

    if not yaml_files:

        raise FileNotFoundError(
            "dataset.yaml was not found inside the "
            "ClearML dataset."
        )

    if len(yaml_files) > 1:

        print("Warning: multiple dataset.yaml files found.")

    dataset_yaml = yaml_files[0]

    print(f"[OK] dataset.yaml found:")
    print(dataset_yaml)

    return dataset_yaml


# ============================================================
# PREPARE REMOTE YAML
# ============================================================

def prepare_yaml(dataset_yaml):
    """
    Rewrite dataset.yaml so that YOLO uses the
    actual path on the GPU machine.
    """

    print("=" * 60)
    print("PREPARING REMOTE DATASET YAML")
    print("=" * 60)

    with open(dataset_yaml, "r", encoding="utf-8") as file:

        data = yaml.safe_load(file)

    if not isinstance(data, dict):

        raise ValueError(
            "dataset.yaml does not contain a valid YAML dictionary."
        )

    dataset_root = dataset_yaml.parent

    train_path = dataset_root / "images" / "train"
    val_path = dataset_root / "images" / "val"
    test_path = dataset_root / "images" / "test"

    if not train_path.exists():

        raise FileNotFoundError(
            f"Training images not found:\n{train_path}"
        )

    if not val_path.exists():

        raise FileNotFoundError(
            f"Validation images not found:\n{val_path}"
        )

    # --------------------------------------------------------
    # Convert paths to strings understood by Ultralytics
    # --------------------------------------------------------

    data["path"] = str(dataset_root)

    data["train"] = str(train_path)

    data["val"] = str(val_path)

    if test_path.exists():

        data["test"] = str(test_path)

    remote_yaml = dataset_root / "remote_dataset.yaml"

    with open(
        remote_yaml,
        "w",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )

    print("[OK] Remote YAML created.")

    print(f"Dataset root : {dataset_root}")
    print(f"Train        : {train_path}")
    print(f"Validation   : {val_path}")

    if test_path.exists():

        print(f"Test         : {test_path}")

    return remote_yaml


# ============================================================
# TRAIN YOLO
# ============================================================

def train(dataset_yaml):
    """
    Train YOLO11 on the ClearML GPU machine.
    """

    print("=" * 60)
    print("STARTING YOLO11 TRAINING")
    print("=" * 60)

    print(f"Model      : {MODEL_NAME}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Image size : {IMAGE_SIZE}")
    print(f"Batch      : {BATCH_SIZE}")
    print(f"Workers    : {WORKERS}")
    print(f"Device     : CUDA:{DEVICE}")
    print(f"Dataset    : {dataset_yaml}")

    # --------------------------------------------------------
    # Load pretrained YOLO11 nano model
    # --------------------------------------------------------

    model = YOLO(MODEL_NAME)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    results = model.train(

        data=str(dataset_yaml),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        workers=WORKERS,

        device=DEVICE,

        project="runs",

        name=RUN_NAME,

        exist_ok=True,

        pretrained=True,

        save=True,

        plots=True,

        verbose=True,
    )

    return model, results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("WAREHOUSE YOLO11 + CLEARML")
    print("=" * 60)

    # ========================================================
    # LOCAL WINDOWS EXECUTION
    # ========================================================

    if task.running_locally():

        print("Execution mode: LOCAL WINDOWS")

        # Upload dataset and store dataset ID
        dataset_id = get_dataset_id()

        print("=" * 60)
        print("DATASET READY")
        print("=" * 60)

        print(f"ClearML Dataset ID: {dataset_id}")

        print("=" * 60)
        print("QUEUEING GPU TRAINING")
        print("=" * 60)

        task.execute_remotely(
            queue_name=QUEUE_NAME,
            clone=False,
            exit_process=True,
        )

        return

    # ========================================================
    # REMOTE GPU EXECUTION
    # ========================================================

    print("Execution mode: REMOTE GPU")

    # --------------------------------------------------------
    # Get dataset ID
    # --------------------------------------------------------

    dataset_id = get_dataset_id()

    # --------------------------------------------------------
    # Download dataset
    # --------------------------------------------------------

    dataset_root = download_dataset(
        dataset_id
    )

    # --------------------------------------------------------
    # Find YAML
    # --------------------------------------------------------

    dataset_yaml = find_dataset_yaml(
        dataset_root
    )

    # --------------------------------------------------------
    # Prepare YAML
    # --------------------------------------------------------

    remote_yaml = prepare_yaml(
        dataset_yaml
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model, results = train(
        remote_yaml
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print("YOLO training completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
