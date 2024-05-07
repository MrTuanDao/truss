import asyncio
import logging
import time

import bcp47
import data_types
import helpers
import httpx
import numpy as np
import truss_chains as chains
from truss_chains import stub

_IMAGE = docker_image = chains.DockerImage(
    apt_requirements=["ffmpeg"], pip_requirements=["pandas", "bcp47"]
)

# Whisper is deployed as a normal truss model from examples/library.
_WHISPER_URL = "https://model-5woz91z3.api.baseten.co/production/predict"
# See `InternalWebhook`-chainlet below.
_INTERNAL_WEBHOOK_URL = "https://model-4q99pkxq.api.baseten.co/development/predict"


class DeployedWhisper(chains.StubBase):
    """Transcribes b64_encoded wave snippets.

    Treat the whisper model like an external third party tool."""

    async def run(self, audio_b64: str) -> data_types.WhisperOutput:
        resp = await self._remote.predict_async(json_payload={"audio": audio_b64})
        # TODO: Get Whisper model with langauge, ideally also timestamps.
        language = "dummy language"
        bcp47_key = bcp47.languages.get(language.capitalize(), "default")
        return data_types.WhisperOutput(
            text=resp["text"], language=language, bcp47_key=bcp47_key
        )


class MacroChunkWorker(chains.ChainletBase):
    """Downloads and transcribes larger chunks of the total file (~300s)."""

    remote_config = chains.RemoteConfig(
        docker_image=_IMAGE, compute=chains.Compute(cpu_count=16, memory="32G")
    )
    _whisper: DeployedWhisper

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
    ) -> None:
        super().__init__(context)
        self._whisper = DeployedWhisper.from_url(
            _WHISPER_URL,
            context,
            options=chains.RPCOptions(retries=3),
        )

    async def run(
        self,
        media_url: str,
        params: data_types.TranscribeParams,
        macro_chunk_index: int,
        start_time: str,
        duration_sec: int,
    ) -> data_types.TranscribeOutput:
        t0 = time.time()
        tasks = []
        seg_infos = []
        logging.debug(f"Macro-chunk [{macro_chunk_index:03}]: Starting.")
        async with helpers.DownloadSubprocess(
            media_url, start_time, duration_sec, params.wav_sampling_rate_hz
        ) as wav_stream:
            chunk_stream = helpers.wav_chunker(params, wav_stream, macro_chunk_index)
            async for seg_info, audio_b64 in chunk_stream:
                tasks.append(asyncio.ensure_future(self._whisper.run(audio_b64)))
                seg_infos.append(seg_info)

        results: list[data_types.WhisperOutput] = await asyncio.gather(*tasks)
        segments = []
        for transcription, seg_info in zip(results, seg_infos):
            segments.append(
                data_types.Segment(transcription=transcription, segment_info=seg_info)
            )
        logging.debug(f"Chunk [{macro_chunk_index:03}]: Complete.")
        t1 = time.time()
        processing_duration_sec = t1 - t0
        return data_types.TranscribeOutput(
            segments=segments,
            input_duration_sec=duration_sec,
            processing_duration_sec=processing_duration_sec,
            speedup=duration_sec / processing_duration_sec,
        )


class Transcribe(chains.ChainletBase):
    """Transcribes one file end-to-end and sends results to webhook."""

    remote_config = chains.RemoteConfig(
        docker_image=_IMAGE,
        compute=chains.Compute(cpu_count=16, memory="32G"),
        assets=chains.Assets(secret_keys=["dummy_webhook_key"]),
    )
    _video_worker: MacroChunkWorker
    _async_http: httpx.AsyncClient

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
        video_worker: MacroChunkWorker = chains.provide(MacroChunkWorker, retries=3),
    ) -> None:
        super().__init__(context)
        self._video_worker = video_worker
        self._async_http = httpx.AsyncClient()
        logging.getLogger("httpx").setLevel(logging.WARNING)

    async def _assert_media_supports_range_downloads(self, media_url: str) -> None:
        ok = False
        try:
            head_response = await self._async_http.head(media_url)
            if "bytes" in head_response.headers.get("Accept-Ranges", ""):
                ok = True
            # Check by making a test range request to see if '206' is returned.
            range_header = {"Range": "bytes=0-0"}
            range_response = await self._async_http.get(media_url, headers=range_header)
            ok = range_response.status_code == 206
        except httpx.HTTPError as e:
            logging.error(f"Error checking URL: {e}")

        if not ok:
            raise NotImplementedError(f"Range downloads unsupported for `{media_url}`.")

    async def run(
        self,
        media_url: str,
        params: data_types.TranscribeParams,
        job_descr: data_types.JobDescriptor,
    ) -> data_types.TranscriptionExternal:
        t0 = time.time()
        await self._assert_media_supports_range_downloads(media_url)
        duration_secs = await helpers.query_video_length_secs(media_url)
        video_chunks = helpers.generate_time_chunks(
            int(np.ceil(duration_secs)), params.macro_chunk_size_sec
        )
        tasks = []
        for i, chunk_limits in enumerate(video_chunks):
            logging.info(f"Starting macro-chunk [{i+1:03}/{len(video_chunks):03}].")
            tasks.append(
                asyncio.ensure_future(
                    self._video_worker.run(media_url, params, i, *chunk_limits)
                )
            )

        results: list[data_types.TranscribeOutput] = await asyncio.gather(*tasks)
        t1 = time.time()
        processing_time = t1 - t0
        result = data_types.TranscribeOutput(
            segments=[],
            input_duration_sec=duration_secs,
            processing_duration_sec=processing_time,
            speedup=duration_secs / processing_time,
        )
        logging.info(result)

        # Type issue seems to be a mypy bug.
        external_result = data_types.TranscriptionExternal(
            media_url=job_descr.media_url,
            media_id=job_descr.media_id,
            job_uuid=job_descr.job_uuid,
            status=data_types.JobStatus.SUCCEEDED,  # type: ignore[arg-type]
            text=[
                data_types.TranscriptionSegmentExternal(
                    start=seg.segment_info.start_time_sec,
                    end=seg.segment_info.end_time_sec,
                    text=seg.transcription.text,
                    language=seg.transcription.language,
                    bcp47_key=seg.transcription.bcp47_key,
                )
                for part in results
                for seg in part.segments
            ],
        )
        return external_result


# Shims for external APIs ##############################################################


class InternalWebhook(chains.ChainletBase):
    """Receives results for debugging."""

    remote_config = chains.RemoteConfig(
        docker_image=_IMAGE, compute=chains.Compute(cpu_count=16, memory="32G")
    )

    async def run(self, transcription: data_types.AsyncTranscriptionExternal) -> None:
        logging.info(transcription.json(indent=4))
        # This would call external webhook next.
        # result_json = transcription.json()
        # payload_signature = hmac.new(
        #     self._context.secrets["dummy_webhook_key"].encode("utf-8"),
        #     result_json.encode("utf-8"),
        #     hashlib.sha1,
        # ).hexdigest()
        # headers = {
        #     "X-Baseten-Signature": payload_signature,
        #     "Authorization": f"Api-Key {self._context.get_baseten_api_key()}",
        # }
        # # Extra key `transcription` is needed for test webhook.
        # resp = await self._async_http.post(
        #     webhook_url, json={"transcription": result.dict()}, headers=headers
        # )
        # if resp.status_code == 200:
        #     return
        # else:
        #     raise Exception(f"Could not call results webhook: {resp.content.decode()}.")


class BatchTranscribe(chains.ChainletBase):
    """Accepts a request with multiple transcription jobs and starts the sub-jobs."""

    remote_config = chains.RemoteConfig(
        docker_image=_IMAGE, compute=chains.Compute(cpu_count=16, memory="32G")
    )

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
        transcribe: Transcribe = chains.provide(Transcribe),
        internal_webhook: InternalWebhook = chains.provide(InternalWebhook),
    ):
        super().__init__(context)
        transcribe_service = context.get_service_descriptor(Transcribe.__name__)
        predict_url = transcribe_service.predict_url.replace("predict", "async_predict")
        async_transcribe = transcribe_service.copy(update={"predict_url": predict_url})

        self._async_transcribe = stub.BasetenSession(
            async_transcribe, context.get_baseten_api_key()
        )
        del transcribe, internal_webhook

    def _enqueue(
        self, job: data_types.JobDescriptor, params: data_types.TranscribeParams
    ):
        # TODO: use pydantic model.
        json_payload = {
            "model_input": {
                "media_url": job.media_url,
                "params": params.dict(),
                "job": job.dict(),
            },
            "webhook_endpoint": _INTERNAL_WEBHOOK_URL,
            "inference_retry_config": {
                "max_attempts": 3,
                # "initial_delay_ms": 1000,
                # "max_delay_ms": 5000,
            },
        }
        return self._async_transcribe.predict_async(json_payload)

    async def run(self, batch_input: data_types.BatchInput) -> data_types.BatchOutput:
        logging.info(batch_input)
        logging.info(f"Got `{len(batch_input.media_for_transcription)}` tasks.")
        params = data_types.TranscribeParams()
        tasks = []
        for job in batch_input.media_for_transcription:
            tasks.append(asyncio.ensure_future(self._enqueue(job, params)))

        jobs = []
        for i, val in enumerate(await asyncio.gather(*tasks, return_exceptions=True)):
            job = batch_input.media_for_transcription[i]
            if isinstance(val, Exception):
                logging.exception(f"Could not enqueue `{job.json()}`. {val}")
                job = job.copy(update={"status": data_types.JobStatus.PERMAFAILED})
            else:
                logging.info(val)
                job = job.copy(update={"status": data_types.JobStatus.QUEUED})

            jobs.append(job)
        output = data_types.BatchOutput(success=True, jobs=jobs)
        return output
