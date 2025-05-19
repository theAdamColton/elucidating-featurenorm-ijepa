from datetime import datetime
from pathlib import Path


def get_viz_output_path():
    viz_output_path = (
        Path(".") / "viz-outputs" / datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    viz_output_path.mkdir(parents=True)
    return viz_output_path
