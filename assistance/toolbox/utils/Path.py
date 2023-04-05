import os
from pathlib import Path

# Change directory
# Modify this cell to insure that the output shows the correct path.
# Define all paths relative to the project root shown in the cell output
def project_root_path() -> Path:
    project_root = Path.cwd()
    while not Path('LICENSE').is_file():
        os.chdir(Path(Path.cwd(), '../'))
        if Path.cwd() == project_root:
            raise Exception('Cannot find project root')
    return Path.cwd()
