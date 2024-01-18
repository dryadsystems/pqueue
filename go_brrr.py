from typing import Any, Iterator, Generic, TypeVar
import os
import time
import pickle
import sys
import logging
import traceback
from PIL.Image import Image

ModelType = TypeVar("ModelType")

# 8 random bytes
RANDOM_DELIMITER = b"\xae\xdc\t\xd1\xffr\xbe\xd1"


class BrrrGoer(Generic[ModelType]):
    cold = True
    # get model and version here somehow

    def __init__(self) -> None:
        self.output = os.fdopen(int(os.environ["PICKLE_FD"]), "wb")

    def write_output(self, obj: Any) -> None:
        self.output.write(pickle.dumps(obj, protocol=5))
        self.output.write(RANDOM_DELIMITER)
        self.output.flush()

    def admin(self, msg: str) -> None:
        self.write_output({"admin": msg})

    def create_generator(self) -> ModelType:
        raise NotImplementedError

    def run(self) -> None:
        start_warmup = time.time()
        generator = self.create_generator()
        elapsed_warmup = time.time() - start_warmup
        if elapsed_warmup > 60:
            self.admin(f"warmup took {elapsed_warmup}!")
        while 1:
            params = pickle.load(sys.stdin.buffer)
            try:
                for result in self.handle_item(generator, params):
                    if isinstance(result, dict):
                        result["cold"] = self.cold
                        if self.cold:
                            result["elapsed_warmup"] = elapsed_warmup
                    self.write_output(result)
            except Exception as e:  # pylint: disable=broad-except
                logging.info("caught exception")
                # otherwise handle this or recover from error...?
                error_message = traceback.format_exc()
                logging.error(error_message)
                if "out of memory" in str(e).lower():
                    sys.exit(137)
                self.write_output({"error": error_message})
            self.cold = False

    def handle_item(self, generator: ModelType, params: dict) -> Iterator[dict | Image]:
        raise NotImplementedError
