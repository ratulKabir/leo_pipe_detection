#!/usr/bin/env python3
"""Extract and undistort frames from a ROS2 mcap bag file."""

import argparse, json
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def _build_undistort_maps(info: dict, alpha: float):
    K = np.array(info["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(info["D"], dtype=np.float64)
    w, h = info["width"], info["height"]
    model = info.get("distortion_model", "plumb_bob")

    if "fisheye" in model or "equidistant" in model:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=alpha
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (w, h), cv2.CV_16SC2
        )
    return map1, map2, new_K


def extract(bag_path, output_dir, every_n, topic, alpha=0.0):
    # Find the .mcap file — accept either the file directly or the bag directory
    bag = Path(bag_path)
    if bag.is_dir():
        mcap_files = list(bag.glob("*.mcap"))
        if not mcap_files:
            raise FileNotFoundError(f"No .mcap file found in {bag}")
        mcap_file = mcap_files[0]
    else:
        mcap_file = bag

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    map1, map2, new_K = None, None, None
    saved_info = False
    frame_idx, saved = 0, 0

    with open(mcap_file, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=[topic, "/camera/camera_info"]
        ):
            if channel.topic == "/camera/camera_info" and not saved_info:
                m = decoded
                info = {
                    "width": m.width, "height": m.height,
                    "distortion_model": m.distortion_model,
                    "D": list(m.d), "K": list(m.k), "P": list(m.p),
                }
                map1, map2, new_K = _build_undistort_maps(info, alpha)
                (out / "camera_info.json").write_text(json.dumps({
                    "width": m.width, "height": m.height,
                    "distortion_model": "none",
                    "D": [0.0] * 5,
                    "K": new_K.flatten().tolist(),
                }, indent=2))
                saved_info = True

            elif channel.topic == topic and frame_idx % every_n == 0:
                m = decoded
                img = cv2.imdecode(
                    np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR
                )
                if map1 is not None:
                    img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
                cv2.imwrite(str(out / f"frame_{message.log_time:020d}.jpg"), img)
                saved += 1
                frame_idx += 1

            elif channel.topic == topic:
                frame_idx += 1

    print(f"Saved {saved} frames → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bag", help="Path to .mcap file or bag directory")
    p.add_argument("-o", "--output", default="frames")
    p.add_argument("-n", "--every-n", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.0,
                   help="0.0=crop black borders (default), 1.0=keep all pixels")
    args = p.parse_args()
    extract(args.bag, args.output, args.every_n, "/camera/image_color/compressed", args.alpha)
