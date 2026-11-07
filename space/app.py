"""Hugging Face Space entry point for TLEye (https://github.com/Lakshaykhatri231/TLEye).
Downloads the verdict adapter at its pinned revision (checksums verified), loads it once, and serves the Gradio demo.
Certificates are off here: they need a Docker sandbox. Submitted code is only read as text."""
import os

os.environ["TLEYE_HOSTED"] = "1"

from tleye.models.fetch import fetch  # noqa: E402

fetch()

import tleye.app as demo  # noqa: E402

demo.MODEL_CHOICE = "lora"
demo.run_check(open("tleye/examples/p02779_fast.cpp").read(), "C++", "CPython", "", demo.EXAMPLE_STATEMENT, 2.0, False)   # load the model before the first visitor
demo.build_ui().launch(server_name="0.0.0.0", server_port=7860)
