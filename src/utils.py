from datetime import datetime
from pathlib import Path
import uuid


def get_viz_output_path():
    folder_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder_str += str(uuid.uuid4())
    viz_output_path = Path(".") / "viz-outputs" / folder_str
    viz_output_path.mkdir(parents=True)
    return viz_output_path
