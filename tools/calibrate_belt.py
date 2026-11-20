import argparse
import json
from pathlib import Path
import numpy as np
from applications.common import read_csv
from applications.conveyor import project


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Estimate an image-to-belt homography from measured planar points")
    cli.add_argument("--points", required=True, help="CSV columns: image_x,image_y,belt_x_m,belt_y_m")
    cli.add_argument("--geometry", required=True, help="JSON containing belt_polygon and belt_direction")
    cli.add_argument("--output", required=True)
    args = cli.parse_args()
    import cv2
    rows = read_csv(args.points)
    image = np.array([[float(r[k]) for k in ("image_x", "image_y")] for r in rows])
    physical = np.array([[float(r[k]) for k in ("belt_x_m", "belt_y_m")] for r in rows])
    if len(rows) < 4:
        raise ValueError("At least four non-collinear points are required; paper protocol uses ten.")
    homography, _ = cv2.findHomography(image, physical, method=0)
    if homography is None or not np.isfinite(homography).all():
        raise ValueError("Degenerate calibration point arrangement.")
    residual = np.linalg.norm(project(physical, np.linalg.inv(homography)) - image, axis=1)
    geometry = json.loads(Path(args.geometry).read_text(encoding="utf-8"))
    geometry.update(homography=homography.tolist(), points=rows, coordinate_system="crop_pixels_to_belt_meters",
                    in_sample_reprojection_mean_px=float(residual.mean()),
                    in_sample_reprojection_max_px=float(residual.max()))
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    print(f"Mean / max in-sample reprojection residual: {residual.mean():.4f} / {residual.max():.4f} px")
