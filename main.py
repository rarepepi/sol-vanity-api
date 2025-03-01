import multiprocessing
import uvicorn
from core.api import app

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    uvicorn.run(app, host="0.0.0.0", port=8000)
