import os


class Path:
    """Minimal dataset-root resolver.

    Reconstructed to satisfy `from mypath import Path` in the ResNet entry point.
    The original DGMS repo shipped this; it is missing from SQS_private.
    Point SQS_DATA_DIR at wherever CIFAR should live (defaults to repo Data/).
    """

    @staticmethod
    def db_root_dir(dataset):
        root = os.environ.get(
            "SQS_DATA_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Data"),
        )
        os.makedirs(root, exist_ok=True)
        return root
