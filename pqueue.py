#!/usr/bin/python3.10
# Copyright (c) 2022 Dryad Systems
# Copyright (c) 2021 Sylvie Liberman
# pylint: disable=consider-using-f-string
import asyncio
import dataclasses
import io
import json
import logging
import os
import pickle
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Callable
import aiohttp
import asyncpg
from PIL.Image import Image
from PIL.Image import open as open_image

# import image_combiner
import vector_logging
import blur
from forest import pghelp, utils

if not utils.get_secret("CHECK_SAFETY"):
    check_nsfw: Optional[Callable] = None
else:
    from safety import check_nsfw


# 8 random bytes
RANDOM_DELIMITER = b"\xae\xdc\t\xd1\xffr\xbe\xd1"
R2_URL = "https://image-gen-worker.drysys.workers.dev"
hostname = socket.gethostname()
scale_id = utils.get_secret("SCALE_ID")
admin_signal_url = utils.get_secret("SIGNAL_URL") or "https://imogen-dryad.fly.dev"
log_url = f"https://ui.honeycomb.io/dryadsystems/environments/test/datasets/sparkl?query=%7B%22time_range%22%3A1209600%2C%22calculations%22%3A%5B%5D%2C%22filters%22%3A%5B%7B%22column%22%3A%22host%22%2C%22op%22%3A%22%3D%22%2C%22value%22%3A%22{hostname}%22%7D%5D%2C%22limit%22%3A100%2C%22end_time%22%3ATIME%7D"
delay_upload = float(utils.get_secret("DELAY_UPLOAD") or 0)
load_threshold = float(utils.get_secret("LOAD_THRESHOLD") or 2)


@dataclasses.dataclass
class Prompt:
    prompt_id: int
    callbacks: str
    params: str


@dataclasses.dataclass
class Result:
    generation_info: dict
    images: list[Image]
    prompt: Prompt


QueueExpressions = pghelp.PGExpressions(
    table="prompt_queue",
    clear_assigned="SELECT clear_assigned();",
    _get_prompt="SELECT get_prompt($1, $2)",
    mark_errored="""
        UPDATE prompt_queue SET status=CASE WHEN errors>=2 THEN 'failed' ELSE 'pending' END,
        errors=errors+1 WHERE id=$1
    """,
    set_uploading="""
        UPDATE prompt_queue SET status='uploading', uploading_ts=now(),
        generation_info=$2::jsonb WHERE id=$1;
    """,
    append_image_url="""
        UPDATE prompt_queue SET
        outputs=jsonb_set(outputs, '{image_urls}', outputs->'image_urls' || $2::jsonb) WHERE id=$1
    """,
    set_outputs="""
        UPDATE prompt_queue SET outputs=
        jsonb_set(coalesce(outputs, jsonb '{{}}'), $2::text[], $3::jsonb) WHERE id=$1
    """,
    set_done="UPDATE prompt_queue SET status='done', done_ts=now() WHERE id=$1",
    queue_length="SELECT queue_length($1)",
    append="""
        UPDATE prompt_queue SET output=jsonb_set(
            coalesce(outputs, jsonb '{{}}'),
            $2::text[],
            coalesce(coalesce(outputs, jsonb '{{}}') #> $2::text[], jsonb '[]') || $3
        ) WHERE id=$1
    """,
)


class QueueManager(pghelp.PGInterface):
    def __init__(self, database: str, database_name: str = "postgres") -> None:
        super().__init__(QueueExpressions, database, database_name=database_name)

    async def get_prompt(self, model: str) -> Optional[Prompt]:
        res = dict((await self._get_prompt(model, hostname))[0].get("get_prompt", {}))
        if res.get("prompt_id"):
            return Prompt(**res)
        return None


WorkerExpressions = pghelp.PGExpressions(
    table="workers",
    heartbeat="UPDATE workers SET last_online=now() WHERE hostname=$1",
    set_status="UPDATE workers SET status=$2 WHERE hostname=$1",
    exit="UPDATE workers SET status='exited' WHERE scale_id=$1",
    load_info="SELECT worker_counts($1), queue_length($1)",
    my_cloud_id="select cloud_identifier from workers where scale_id=$1",
    # select should_exit from workers where scale_id=$1
    # should_exit="select should_exit from flags where model=$1",
    # update state_changes set exiting=true, hostname=$2 where model=$1 returning should_exit; # ?
)


class WorkerManager(pghelp.PGInterface):
    def __init__(self, database: str, database_name: str = "postgres") -> None:
        super().__init__(WorkerExpressions, database, database_name=database_name)

    async def mark_running(self, model: str) -> None:
        # FIXME
        # in a scale_id conflict, check if hostname is set - if so, do something else
        args = [scale_id, hostname, utils.ENV.lower(), model]
        raw_result = await self.execute(
            """
            INSERT INTO workers (scale_id, hostname, env, model, first_online, last_online, status)
            VALUES ($1, $2, $3, $4, now(), now(), 'running')
            ON CONFLICT (scale_id) DO UPDATE SET
            hostname=$2, env=$3, model=$4, status='running',
            first_online=coalesce(workers.first_online, now()), last_online=now();
            """,
            *args,
        )
        logging.info(raw_result)


async def async_fdopen(fd: int, mode: str = "rb") -> asyncio.StreamReader:
    "open a file descriptor as an async stream reader"
    # pylint: disable=unnecessary-lambda-assignment
    loop = asyncio.get_running_loop()
    # 32 MB covers upscaled images
    reader = asyncio.StreamReader(limit=2**25)
    # pulled from https://gist.github.com/oconnor663/08c081904264043e55bf
    # tbh i don't understand how this works so here's the docs
    # "Helper class to adapt between Protocol and StreamReader"
    protocol = lambda: asyncio.StreamReaderProtocol(reader)
    # "Register the read end of pipe in the event loop."
    await loop.connect_read_pipe(protocol, os.fdopen(fd, mode))
    return reader


def img_bytes(image: Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="png")
    buf.seek(0)
    return buf.read()


last_logged: dict[tuple, int] = {}


class PQueue:
    model: str
    version = "unknown"
    exiting = False

    def __init__(self, *_: Any) -> None:
        self.version = utils.get_secret("MODEL_VERSION") or self.version
        self.model = utils.get_secret("MODEL", fail_if_none=True)
        self.queue = QueueManager(database=utils.get_secret("DATABASE_URL"))
        self.workers = WorkerManager(database=utils.get_secret("DATABASE_URL"))
        self.vector = vector_logging.Vector()

    client_session: aiohttp.ClientSession

    async def startup(self) -> None:
        await self.vector.init_vector()
        # client session needs to be started with a running loop
        self.client_session = aiohttp.ClientSession()
        Path("./input").mkdir(exist_ok=True)
        moji = "\N{artist palette}\N{construction worker}\N{hiking boot}"
        await self.admin(f"{moji} {hostname} / {scale_id}")
        logging.info("starting pqueue on %s", hostname)
        # FIXME: after an error we might restart with the same scale id but a different hostname
        await self.workers.mark_running(self.model)

        def heartbeat_restarter(_: Optional[asyncio.Task] = None) -> None:
            if self.exiting:
                return
            self.heartbeat_task = asyncio.create_task(self.heartbeat())
            self.heartbeat_task.add_done_callback(utils.log_task_result)
            self.heartbeat_task.add_done_callback(heartbeat_restarter)

        heartbeat_restarter()

    async def shutdown(self) -> None:
        logging.warning("shutting down")
        self.exiting = True
        await self.workers.pool.close()

    async def admin(self, msg: str) -> None:
        "send a message to admin"
        logging.info(msg)
        for i in range(3):
            try:
                await self.client_session.post(
                    f"{admin_signal_url}/admin", params={"message": str(msg)}
                )
                break
            except aiohttp.ClientError:
                logging.error("couldn't send admin message")

    async def should_exit(self) -> str:
        is_k8s = utils.get_secret("KUBERNETES_PORT")
        _worker_counts, queue_length = (await self.workers.load_info(self.model))[0]
        worker_counts = json.loads(_worker_counts)
        workers = sum(worker_counts.values())
        runpod_workers = worker_counts.get("runpod", 0)
        load = queue_length / workers if workers else float("inf")
        if queue_length == 0:
            if is_k8s:
                return "\N{Octagonal Sign}%s queue empty and we're k8s"
            if utils.ENV.lower() not in ("prod", "staging"):
                return "\N{Octagonal Sign}%s queue empty and not in prod"
        if workers <= 1:
            # we should count as online
            logging.info("only %s workers healthy, not exiting", workers)
            return ""
        # FIXME: check here how many workers we should have for each model in prod
        if load < load_threshold and (is_k8s or runpod_workers > 1):
            return f"\N{chart with downwards trend}%s queue {queue_length} workers {workers} load {load} runpod {runpod_workers}"
        # don't log the same message more than once per two minutes
        message = (
            "not exiting: queue %s workers %s runpod %s",
            queue_length,
            workers,
            runpod_workers,
        )
        if time.time() - last_logged.get(message, 0) > 120:
            logging.debug(*message)
            last_logged[message] = round(time.time())
        return ""

    async def exit(self, reason: str) -> None:
        "decide how to exit based on envvars"
        if utils.get_secret("NEVER_EXIT"):
            return
        await self.workers.exit(scale_id)
        logging.debug("exiting")
        prefix = "\N{cross mark}\N{construction worker}"
        if utils.get_secret("POWEROFF"):
            method = "\N{Mobile Phone Off}"
            suffix = f" h_{hostname}/s_{scale_id}/{self.model}"
            await self.admin(prefix + reason % method + suffix)
            await self.shutdown()
            await asyncio.create_subprocess_shell("sudo poweroff")
        elif utils.get_secret("RUNPOD_API_KEY"):
            method = "\N{Skull and Crossbones}\N{Variation Selector-16}"
            _pod_id = os.getenv("RUNPOD_POD_ID")
            pod_id = _pod_id or (await self.workers.my_cloud_id(scale_id))[0][0]
            suffix = f" h_{hostname}/s_{scale_id}/p_{pod_id}/{self.model}"
            await self.admin(prefix + reason % method + suffix)
            await self.shutdown()
            resp = await self.client_session.post(
                "https://api.runpod.io/graphql",
                params={"api_key": utils.get_secret("RUNPOD_API_KEY")},
                json={
                    "query": 'mutation {podTerminate(input: {podId: "%s"})}' % pod_id
                },
                headers={"Content-Type": "application/json"},
            )
            logging.info(resp)
        elif utils.get_secret("EXIT"):
            method = "\N{sleeping symbol}"
            suffix = f" h_{hostname}/s_{scale_id}/{self.model}"
            await self.admin(prefix + reason % method + suffix)
            await self.shutdown()
            sys.exit(0)

    async def maybe_exit(self) -> bool:
        reason = await self.should_exit()
        if reason:
            logging.info("should exit, waiting for uploads to finish")
            # FIXME: if uploads were slower than generations
            # and there's an uploads pending after the one in progress
            # there could be a race condition where a pending upload might be dropped
            async with self.upload_lock:
                logging.info("uploads done, exiting")
                # check again that other workers haven't exited while we were uploading
                if await self.should_exit():
                    await self.exit(reason)
                    return True
        return False

    async def heartbeat(self) -> None:
        while not self.exiting:
            try:
                code = self.brrr_proc.returncode
                if not code:
                    await self.workers.heartbeat(hostname)
                else:
                    await self.workers.set_status(hostname, f"unhealthy ({code})")
                    logging.error("brrr proc return code is %s, unhealthy", code)
            except AttributeError:
                logging.error("no self.brrr_proc, not beating heart")
            except asyncpg.PostgresConnectionError:
                logging.error("pg connection error, couldn't beat heart")
            await asyncio.sleep(2)

    async def start_brrr_proc(self) -> None:
        brrr_read, brrr_write = os.pipe()
        self.brrr_reader = await async_fdopen(brrr_read)
        self.brrr_proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "./brrr.py",
            stdin=-1,
            limit=2**20,
            close_fds=True,
            pass_fds=[brrr_write],
            env=dict(os.environ) | {"PICKLE_FD": str(brrr_write)},
        )  # limit 1MB buffer

    async def generate_prompt(self, prompt: Prompt) -> AsyncIterator[Image | dict]:
        # check params[init_image, target_image...], download that
        # and replace the attribute with bytes, or just filenames
        assert self.brrr_proc.stdin and self.brrr_reader
        self.brrr_proc.stdin.write(pickle.dumps(json.loads(prompt.params), protocol=5))
        await self.brrr_proc.stdin.drain()
        images: list[Image] = []
        while 1:
            try:
                # FIXME: with a manually created pipe, it can stay open despite the process exiting
                data = await asyncio.wait_for(
                    self.brrr_reader.readuntil(RANDOM_DELIMITER), 10
                )
            except asyncio.TimeoutError as exc:
                if self.brrr_proc.returncode:
                    msg = f"brrr_proc exited, check {log_url.replace('TIME', str(round(time.time())))}"
                    raise Exception(msg) from exc
                continue
            except asyncio.IncompleteReadError:
                # EOF, restart proc
                logging.error("IncompleteReadError")
                raise
            except asyncio.LimitOverrunError:
                # chunk is bigger than the pipe limit (1MB), collate?
                logging.error("LimitOverrunError")
                raise
            try:
                blob = pickle.loads(data)
            except pickle.UnpicklingError:
                logging.error("couldn't unpickle, maybe this is a print?")
                raise
            if isinstance(blob, bytes):
                # handle itermediate blob
                yield open_image(io.BytesIO(blob))
                # images.append()
            elif isinstance(blob, Image):
                yield blob
                # handle_partial_prompt...
            elif isinstance(blob, dict):
                # it's generation info at the end
                # assume generation info is always last
                if "error" in blob:
                    raise Exception(blob["error"])
                if "admin" in blob:
                    await self.admin(blob["admin"])
                    continue
                yield blob
                break
            else:
                logging.info("pickle blob neither bytes nor dict: %s", blob)

    upload_lock = asyncio.Lock()

    async def main(self, *_: Any) -> None:
        "setup, get prompts, handle them, mark as uploading, upload, mark done"
        await self.startup()
        backoff = 60.0
        gpu_model = (
            await (
                await asyncio.create_subprocess_shell("nvidia-smi -L", stdout=-1)
            ).communicate()
        )[0].decode()
        await self.start_brrr_proc()
        try:
            # is there a prompt?
            # if we should be exiting, are we uploading?
            # spend a minute checking
            while not self.exiting:
                if await self.maybe_exit():
                    return
                # try to claim
                prompt = await self.queue.get_prompt(self.model)
                if not prompt:
                    try:
                        await self.queue.pool.listen_once("prompt", 60)
                        prompt = await self.queue.get_prompt(self.model)
                    except asyncio.TimeoutError:
                        pass
                if not prompt:
                    continue
                logging.info("got prompt: %s", prompt)
                self.vector.prompt_context.set(prompt.prompt_id)
                t0 = time.time()
                generation_info: Optional[dict] = None
                # FIXME:
                # it would be extremely cool and sexy of us to download init_image
                # and target_images here + rewrite those params as filepaths
                try:
                    images = []
                    async for blob in self.generate_prompt(prompt):
                        if isinstance(blob, Image):
                            images.append(blob)
                            # await self.handle_partial_result(prompt, images)
                        else:
                            generation_info = blob
                    assert generation_info
                    backoff = 60
                except Exception as e:  # pylint: disable=broad-except
                    logging.info("caught exception")
                    # clear existing outputs and start over
                    error_message = traceback.format_exc()
                    if prompt:
                        await self.admin(repr(prompt))
                    logging.error(error_message)
                    await self.admin(error_message)
                    await self.queue.mark_errored(prompt.prompt_id)
                    await asyncio.sleep(backoff)
                    if self.brrr_proc.returncode:
                        await self.start_brrr_proc()
                    backoff *= 1.5
                    continue
                info_from_pqueue = {
                    "model_version": self.version,
                    "gpu": gpu_model,
                    "scale_id": utils.get_secret("SCALE_ID"),
                    "outer_elapsed": round(time.time() - t0, 4),
                }
                result = Result(generation_info | info_from_pqueue, images, prompt)
                upload_task = asyncio.create_task(self.upload_images(result))
                upload_task.add_done_callback(utils.log_task_result)
        finally:
            logging.info("hit main loop finally, shutting down(redundant?)")
            await self.shutdown()

    async def put_image(self, image: Image, url: str, fmt: str = "png") -> str:
        img = img_bytes(image)
        auth = utils.get_secret("R2_UPLOAD_KEY")
        headers = {"X-Custom-Auth-Key": auth, "Content-Type": f"image/{fmt}"}
        for i in range(3):
            try:
                resp = await self.client_session.put(url, headers=headers, data=img)
                resp.raise_for_status()
                return url
            except aiohttp.ClientError:
                logging.error("couldn't upload")
        raise aiohttp.ClientError

    # async def handle_partial_result(self, prompt: Prompt, image: Image) -> str:
    #     # FIXME: this requies keeping track of these partial uploads
    #     # cancelling them, making sure they're not going, and then
    #     return ""

    async def upload_images(self, result: Result) -> None:
        async with self.upload_lock:
            start_upload = time.time()
            prompt_id = result.prompt.prompt_id
            if check_nsfw:
                is_nsfw = check_nsfw(result.images)
                result.generation_info["has_nsfw"] = any(is_nsfw)
            await self.queue.set_uploading(
                prompt_id, json.dumps(result.generation_info)
            )
            blurs = await blur.blur_images(result.images)
            await self.queue.set_outputs(prompt_id, ["blurhashes"], json.dumps(blurs))
            env = utils.ENV.lower()
            tasks: list[asyncio.Task[str]] = []
            for i, image in enumerate(result.images):
                url = f"{R2_URL}/{env}/{prompt_id}/{i}.png"
                task = asyncio.create_task(self.put_image(image, url))
                task.add_done_callback(utils.log_task_result)
                tasks.append(task)
            # tasks.append(asyncio.create_task(self.combined(result)))
            urls = await asyncio.gather(*tasks)
            if delay_upload:
                await asyncio.sleep(delay_upload)
            await self.queue.set_outputs(prompt_id, ["image_urls"], json.dumps(urls))
            await self.callbacks(result, urls)
            await self.queue.set_done(prompt_id)
            logging.info("set done, poasting time: %s", time.time() - start_upload)
            await self.signal(result)

    async def callbacks(self, result: Result, urls: list[str]) -> None:
        for callback in result.prompt.callbacks:
            if callback["type"] != "webhook":
                continue
            url = callback["url"]
            for i in range(3):
                try:
                    resp = await self.client_session.post(
                        url,
                        params={"id": str(result.prompt.prompt_id), "env": utils.ENV},
                        data={"image": io.BytesIO(img)},
                    )
                    logging.info(resp)
                    break
                except aiohttp.ClientError:
                    logging.info("pausing before retry")
                    time.sleep(i)
        params = json.loads(result.prompt.params)
        maybe_index = params.get("index")
        maybe_id = params.get("upscale_id")
        logging.info("in callbacks, params are %s", params)
        if maybe_index is not None and maybe_id is not None:
            logging.info("setting to %s", json.dumps(urls[0]))
            await self.queue.set_outputs(
                maybe_id, ["upscaled_image_urls", str(maybe_index)], json.dumps(urls[0])
            )

    # FIXME: come up with something clean for callbacks and partial callbacks
    # async def combined(self, result: Result) -> str:
    #     prompt_text = ", ".join(
    #         [prompt["text"] for prompt in result.prompt.params["prompts"]]  # type: ignore
    #     )
    #     image = image_combiner.make_image(
    #         prompt_text, result.images, image_combiner.RawConfigWithHeader
    #     )
    #     path = f"{utils.ENV.lower()}/{result.prompt.prompt_id}/combined_image.png"
    #     url = f"{R2_URL}/{path}"
    #     await self.put_image(image, url)
    #     return url

    async def signal(self, result: Result) -> None:
        img = img_bytes(result.images[0])  # webhook only one file for now
        # url = result.prompt.callbacks["signal"]["url"] or admin_signal_url
        url = admin_signal_url
        for i in range(3):
            try:
                resp = await self.client_session.post(
                    f"{url}/attachment",
                    params={"id": str(result.prompt.prompt_id), "env": utils.ENV},
                    data={"image": io.BytesIO(img)},
                )
                logging.info(resp)
                break
            except aiohttp.ClientError:
                logging.info("pausing before retry")
                time.sleep(i)


if __name__ == "__main__":
    with blur.pool:
        asyncio.run(PQueue().main())
