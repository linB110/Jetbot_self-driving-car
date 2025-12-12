import pyrealsense2 as rs

def get_realsense_intrinsics():
    # create pipeline & config
    pipeline = rs.pipeline()
    config = rs.config()

    # enable depth / color stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # start streaming
    profile = pipeline.start(config)

    try:
        # acquire depth stream profile and cmaera intrinsics
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intr = depth_profile.get_intrinsics()

        # acquire color stream profile and cmaera intrinsics
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intr = color_profile.get_intrinsics()

        depth_intrinsics = {
            "width": depth_intr.width,
            "height": depth_intr.height,
            "ppx": depth_intr.ppx,   # principal point x
            "ppy": depth_intr.ppy,   # principal point y
            "fx": depth_intr.fx,     # focal length x
            "fy": depth_intr.fy,     # focal length y
            "model": str(depth_intr.model),   # distortion model
            "coeffs": list(depth_intr.coeffs) # distortion coefficients
        }

        color_intrinsics = {
            "width": color_intr.width,
            "height": color_intr.height,
            "ppx": color_intr.ppx,
            "ppy": color_intr.ppy,
            "fx": color_intr.fx,
            "fy": color_intr.fy,
            "model": str(color_intr.model),
            "coeffs": list(color_intr.coeffs)
        }

        return depth_intrinsics, color_intrinsics

    finally:
        # stop pipeline
        pipeline.stop()


if __name__ == "__main__":
    depth_intr, color_intr = get_realsense_intrinsics()

    print("Depth intrinsics:")
    for k, v in depth_intr.items():
        print(f"  {k}: {v}")

    print("\nColor intrinsics:")
    for k, v in color_intr.items():
        print(f"  {k}: {v}")
