import asyncio
import concurrent
import logging
import time
from itertools import chain
import blurhash
from PIL.Image import Image


def compute_blurhash(image: Image, x_components: int = 4, y_components: int = 4) -> str:
    "like blurhash.encode, but takes an Image instead of a file"
    red_band = image.getdata(band=0)
    green_band = image.getdata(band=1)
    blue_band = image.getdata(band=2)
    rgb_data = list(chain.from_iterable(zip(red_band, green_band, blue_band)))
    width, height = image.size

    rgb = blurhash._ffi.new("uint8_t[]", rgb_data)
    bytes_per_row = blurhash._ffi.cast("size_t", width * 3)
    width = blurhash._ffi.cast("int", width)
    height = blurhash._ffi.cast("int", height)
    x_components = blurhash._ffi.cast("int", x_components)
    y_components = blurhash._ffi.cast("int", y_components)

    result = blurhash._lib.create_hash_from_pixels(
        x_components, y_components, width, height, rgb, bytes_per_row
    )

    if result == blurhash._ffi.NULL:
        logging.error("Invalid x_components or y_components")
        return ""

    return blurhash._ffi.string(result).decode()


pool = concurrent.futures.ProcessPoolExecutor()


async def blur_images(images: list[Image]) -> list[str]:
    start = time.time()
    if len(images) == 1:
        blurs = [compute_blurhash(images[0])]
    else:
        # we don't want to close the pool as we exit
        # hopefully shutdown gets called properly!
        # with pool as executor:
        loop = asyncio.get_running_loop()
        blurs = await asyncio.gather(
            *[loop.run_in_executor(pool, compute_blurhash, img) for img in images]
        )
    logging.info("blured %s images in %.3f", len(images), time.time() - start)
    return blurs
