"""Root entry point: `streamlit run draft_app.py`.

Deliberately at the repo root rather than inside src/: `streamlit run` puts the
SCRIPT'S OWN directory on sys.path, so an app living inside src/ could not
resolve `from src import ...` (src/ is a package, not the path Streamlit would
add). All real UI logic lives in src/app_ui.py; this file is a three-line shim.
"""

from src.app_ui import main

main()
