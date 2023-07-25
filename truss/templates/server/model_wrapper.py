import asyncio
import importlib
import inspect
import logging
import os
import sys
import time
import traceback
from collections.abc import Generator
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import Any, AsyncGenerator, Dict, Optional, Union

from anyio import Semaphore, to_thread
from common.patches import apply_patches
from common.retry import retry
from shared.secrets_resolver import SecretsResolver

MODEL_BASENAME = "model"

NUM_LOAD_RETRIES = int(os.environ.get("NUM_LOAD_RETRIES_TRUSS", "3"))
STREAMING_RESPONSE_QUEUE_READ_TIMEOUT_SECS = 60
DEFAULT_PREDICT_CONCURRENCY = 1


class ModelWrapper:
    class Status(Enum):
        NOT_READY = 0
        LOADING = 1
        READY = 2
        FAILED = 3

    def __init__(self, config: Dict):
        self._config = config
        self._logger = logging.getLogger()
        self.name = MODEL_BASENAME
        self.ready = False
        self._load_lock = Lock()
        self._status = ModelWrapper.Status.NOT_READY
        self._predict_semaphore = Semaphore(
            self._config.get("runtime", {}).get(
                "predict_concurrency", DEFAULT_PREDICT_CONCURRENCY
            )
        )

    def load(self) -> bool:
        if self.ready:
            return self.ready

        # if we are already loading, just pass; our container will return 503 while we're loading
        if not self._load_lock.acquire(blocking=False):
            return False

        self._status = ModelWrapper.Status.LOADING

        self._logger.info("Executing model.load()...")

        try:
            start_time = time.perf_counter()
            self.try_load()
            self.ready = True
            self._status = ModelWrapper.Status.READY
            self._logger.info(
                f"Completed model.load() execution in {_elapsed_ms(start_time)} ms"
            )

            return self.ready
        except Exception:
            self._logger.exception("Exception while loading model")
            self._status = ModelWrapper.Status.FAILED
        finally:
            self._load_lock.release()

        return self.ready

    def start_load(self):
        if self.should_load():
            thread = Thread(target=self.load)
            thread.start()

    def load_failed(self) -> bool:
        return self._status == ModelWrapper.Status.FAILED

    def should_load(self) -> bool:
        # don't retry failed loads
        return (
            not self._load_lock.locked()
            and not self._status == ModelWrapper.Status.FAILED
            and not self.ready
        )

    def try_load(self):
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        if "bundled_packages_dir" in self._config:
            bundled_packages_path = Path("/packages")
            if bundled_packages_path.exists():
                sys.path.append(str(bundled_packages_path))
        model_module_name = str(
            Path(self._config["model_class_filename"]).with_suffix("")
        )
        module = importlib.import_module(
            f"{self._config['model_module_dir']}.{model_module_name}"
        )
        model_class = getattr(module, self._config["model_class_name"])
        model_class_signature = inspect.signature(model_class)
        model_init_params = {}
        if _signature_accepts_keyword_arg(model_class_signature, "config"):
            model_init_params["config"] = self._config
        if _signature_accepts_keyword_arg(model_class_signature, "data_dir"):
            model_init_params["data_dir"] = data_dir
        if _signature_accepts_keyword_arg(model_class_signature, "secrets"):
            model_init_params["secrets"] = SecretsResolver.get_secrets(self._config)
        apply_patches(
            self._config.get("apply_library_patches", True),
            self._config["requirements"],
        )
        self._model = model_class(**model_init_params)

        if hasattr(self._model, "load"):
            retry(
                self._model.load,
                NUM_LOAD_RETRIES,
                self._logger.warn,
                "Failed to load model.",
                gap_seconds=1.0,
            )

    async def preprocess(
        self,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        if not hasattr(self._model, "preprocess"):
            return payload

        if inspect.iscoroutinefunction(self._model.preprocess):
            return await self._model.preprocess(payload)
        else:
            return await to_thread.run_sync(self._model.preprocess, payload)

    def _execute_sync_predict(self, payload):
        try:
            return self._model.predict(payload)
        except Exception:
            logging.exception("Exception while running predict")
            return {"error": {"traceback": traceback.format_exc()}}

    async def _execute_async_predict(self, payload):
        try:
            return await self._model.predict(payload)
        except Exception:
            logging.exception("Exception while running predict")
            return {"error": {"traceback": traceback.format_exc()}}

    async def predict(
        self,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        # It's possible for the user's predict function to be a:
        #   1. Generator function (function that returns a generator)
        #   2. Async generator (function that returns async generator)
        # In these cases, just return the generator or async generator,
        # as we will be propagating these up. No need for await at this point.
        #   3. Coroutine -- in this case, await the predict function as it is async
        #   4. Normal function -- in this case, offload to a separate thread to prevent
        #      blocking the main event loop
        if inspect.isasyncgenfunction(
            self._model.predict
        ) or inspect.isgeneratorfunction(self._model.predict):
            return self._model.predict(payload)

        if inspect.iscoroutinefunction(self._model.predict):
            return await self._execute_async_predict(payload)

        return await to_thread.run_sync(self._execute_sync_predict, payload)

    async def postprocess(
        self,
        response: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        # Similar to the predict function, it is possible for postprocess
        # to return either a generator or async generator, in which case
        # just return the generator.
        #
        # It can also return a coroutine or just be a function, in which
        # case either await, or offload to a thread respectively.
        if not hasattr(self._model, "postprocess"):
            return response

        if inspect.isasyncgenfunction(
            self._model.postprocess
        ) or inspect.isgeneratorfunction(self._model.postprocess):
            return self._model.predict(response, headers)

        if inspect.iscoroutinefunction(self._model.postprocess):
            return await self._model.postprocess(response)

        return await to_thread.run_sync(self._model.postprocess, response)

    async def __call__(
        self, body: Any, headers: Optional[Dict[str, str]] = None
    ) -> Union[Dict, Generator]:
        """Method to call predictor or explainer with the given input.

        Args:
            body (Any): Request payload body.
            headers (Dict): Request headers.

        Returns:
            Dict: Response output from preprocess -> predictor -> postprocess
            Generator: In case of streaming response
        """

        payload = await self.preprocess(body, headers)

        async with self._predict_semaphore:
            response = await self.predict(payload, headers)

            processed_response = await self.postprocess(response)

            # Streaming cases
            if inspect.isgenerator(response) or inspect.isasyncgen(response):
                async_generator = _force_async_generator(response)

                if headers and headers.get("accept") == "application/json":
                    # In the case of a streaming response, consume stream
                    # if the http accept header is set, and json is requested.
                    return await _convert_streamed_response_to_string(async_generator)

                # To ensure that a partial read from a client does not cause the semaphore
                # to stay claimed, we immediately write all of the data from the stream to a
                # queue. We then return a new generator that reads from the queue, and then
                # exit the semaphore block.
                response_queue: asyncio.Queue = asyncio.Queue()
                async for chunk in async_generator:
                    await response_queue.put(ResponseChunk(chunk))

                await response_queue.put(None)

                async def _response_generator():
                    while True:
                        chunk = await response_queue.get()
                        if chunk is None:
                            return
                        yield chunk.value

                return _response_generator()

            return processed_response


class ResponseChunk:
    def __init__(self, value):
        self.value = value


async def _convert_streamed_response_to_string(response: AsyncGenerator):
    return "".join([str(chunk) async for chunk in response])


def _force_async_generator(gen: Union[Generator, AsyncGenerator]) -> AsyncGenerator:
    """
    Takes a generator, and converts it into an async generator if it is not already.
    """
    if inspect.isasyncgen(gen):
        return gen

    async def _convert_generator_to_async():
        """
        Runs each iteration of the generator in an offloaded thread, to ensure
        the main loop is not blocked, and yield to create an async generator.
        """
        FINAL_GENERATOR_VALUE = object()
        while True:
            chunk = await to_thread.run_sync(next, gen, FINAL_GENERATOR_VALUE)
            if chunk == FINAL_GENERATOR_VALUE:
                break
            yield chunk

    return _convert_generator_to_async()


def _signature_accepts_keyword_arg(signature: inspect.Signature, kwarg: str) -> bool:
    return kwarg in signature.parameters or _signature_accepts_kwargs(signature)


def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _elapsed_ms(since_micro_seconds: float) -> int:
    return int((time.perf_counter() - since_micro_seconds) * 1000)
