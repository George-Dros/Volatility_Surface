"""Ensure the project root is importable and silence Streamlit's
"no runtime" cache warnings so the test output stays readable."""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
logging.getLogger("streamlit").setLevel(logging.ERROR)
