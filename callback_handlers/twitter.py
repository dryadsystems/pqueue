import os
import logging


def post_tweet(self, result: Result, prompt: Prompt) -> None:
    "post tweet, either all at once for images or in chunks for videos"
    if prompt.param_dict.get("nopost") or not config.get_secret("TWITTER"):
        return
    import TwitterAPI as t

    logging.info("uploading to twitter")

    twitter_api = t.TwitterAPI(
        *config.get_secret("TWITTER_CREDS").split(","),
        api_version="1.1",
    )
    username = "@dreambs3"
    if not result.filepath.endswith("mp4"):
        media_resp = twitter_api.request(
            "media/upload", None, {"media": open(result.filepath, mode="rb").read()}
        )
    else:
        bytes_sent = 0
        total_bytes = os.path.getsize(result.filepath)
        file = open(result.filepath, "rb")
        init_req = twitter_api.request(
            "media/upload",
            {
                "command": "INIT",
                "media_type": "video/mp4",
                "total_bytes": total_bytes,
            },
        )

        media_id = init_req.json()["media_id"]
        segment_id = 0

        while bytes_sent < total_bytes:
            chunk = file.read(4 * 1024 * 1024)
            twitter_api.request(
                "media/upload",
                {
                    "command": "APPEND",
                    "media_id": media_id,
                    "segment_index": segment_id,
                },
                {"media": chunk},
            )
            segment_id = segment_id + 1
            bytes_sent = file.tell()
        media_resp = twitter_api.request(
            "media/upload", {"command": "FINALIZE", "media_id": media_id}
        )
    try:
        media = media_resp.json()
        media_id = media["media_id"]
        twitter_post = {
            "status": prompt.prompt,
            "media_ids": media_id,
        }
        twitter_api.request("statuses/update", twitter_post)
    except KeyError:
        try:
            logging.error(media_resp.text)
            admin(media_resp.text)
        except:  # pylint: disable=bare-except
            logging.error("couldn't send to admin")
