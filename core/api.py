from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import multiprocessing
from typing import Optional
import logging
from pathlib import Path

from core.config import DEFAULT_ITERATION_BITS, HostSetting
from core.opencl.manager import get_all_gpu_devices, get_chosen_devices
from core.searcher import multi_gpu_init, save_result
from core.utils.helpers import check_character, load_kernel_source

app = FastAPI()

class GenerateRequest(BaseModel):
    starts_with: str = ""
    ends_with: str = ""
    count: int = 1
    output_dir: str = "./"
    select_device: bool = False
    iteration_bits: int = DEFAULT_ITERATION_BITS
    is_case_sensitive: bool = True

@app.post("/generate")
async def generate_keys(request: GenerateRequest):
    if not request.starts_with and not request.ends_with:
        raise HTTPException(status_code=400, detail="Please provide at least one of starts_with or ends_with")
    
    try:
        check_character("starts_with", request.starts_with)
        check_character("ends_with", request.ends_with)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Ensure output directory exists
    output_path = Path(request.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chosen_devices: Optional[tuple[int, list[int]]] = None
    if request.select_device:
        chosen_devices = get_chosen_devices()
        gpu_counts = len(chosen_devices[1])
    else:
        gpu_counts = len(get_all_gpu_devices())

    logging.info(
        f"Searching Solana pubkey with starts_with='{request.starts_with}', ends_with='{request.ends_with}', case_sensitive={'on' if request.is_case_sensitive else 'off'}"
    )
    logging.info(f"Using {gpu_counts} OpenCL device(s)")

    result_count = 0
    with multiprocessing.Manager() as manager:
        with multiprocessing.Pool(processes=gpu_counts) as pool:
            kernel_source = load_kernel_source(
                request.starts_with, request.ends_with, request.is_case_sensitive
            )
            lock = manager.Lock()
            while result_count < request.count:
                stop_flag = manager.Value("i", 0)
                results = pool.starmap(
                    multi_gpu_init,
                    [
                        (
                            x,
                            HostSetting(kernel_source, request.iteration_bits),
                            gpu_counts,
                            stop_flag,
                            lock,
                            chosen_devices,
                        )
                        for x in range(gpu_counts)
                    ],
                )
                result_count += save_result(results, request.output_dir)

    return {"message": f"Generated {result_count} keys", "count": result_count} 