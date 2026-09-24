"""eyeclean-client: drive the EyeClean engine from any Python >= 3.9 (Blender, Nuke, pipelines, farm workers).

No dependency on torch or CUDA: the engine runs in its own environment as a subprocess.  numpy is needed only to read EXRs.

    from eyeclean_client import run, RunOutput

    res = run({"input": "shot/fg/%06d.exr", "output": "shot/eyeclean", "input_color": "linear-rec709", "fps": "24"},
              on_event=lambda e: print(e["event"], e.get("done"), e.get("total")))
    out = RunOutput(res.output)
    print(out.report["eye_status_counts"])
"""
from .exr import Exr, ExrReadError, read_exr
from .outputs import EyeInfo, FrameInfo, RunOutput
from .runner import (EyeCleanCancelled, EyeCleanConfigError, EyeCleanError, EyeCleanGpuError, EyeCleanInputError, RunResult,
                     find_engine, run)

__version__ = "0.2.0"
__all__ = ["run", "find_engine", "RunResult", "RunOutput", "FrameInfo", "EyeInfo", "read_exr", "Exr", "ExrReadError",
           "EyeCleanError", "EyeCleanConfigError", "EyeCleanGpuError", "EyeCleanInputError", "EyeCleanCancelled"]
