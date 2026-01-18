"""
Exercise 4 Bonus: Forward Panoramas using X-Slit Sampling

Creates the illusion of forward/backward motion through a scene
by varying the strip sampling position across frames.

Supports two blending modes:
- 'pyramid': Barcode blending with averaging (default, smooth but can ghost)
- 'mincut': DP seam finding (sharper, preserves object boundaries)
"""
import numpy as np
import cv2
import imageio.v3 as iio
import time

from ex4 import (
    load_video_frames,
    accumulate_homographies,
    compute_bounding_box,
    warp_image,
    pad_to_macro_block,
    detect_and_fix_direction,
    estimate_horizon_homography,
    PyramidBlender
)


def find_mincut_seam_dp(strip_left, strip_right, overlap_width):
    """
    Find optimal seam between two overlapping strips using min-cut.
    
    Args:
        strip_left: Left strip image (H x W1 x 3)
        strip_right: Right strip image (H x W2 x 3)
        overlap_width: Width of overlap region
    
    Returns:
        seam: Array of x-coordinates (length H) in overlap region [0, overlap_width)
    """
    height = strip_left.shape[0]
    
    if overlap_width <= 0:
        return np.zeros(height, dtype=np.int32)
    
    # Extract overlap regions
    left_overlap = strip_left[:, -overlap_width:].astype(np.float32)
    right_overlap = strip_right[:, :overlap_width].astype(np.float32)
    
    # Cost: squared difference between strips (high = bad place to cut)
    cost = np.sum((left_overlap - right_overlap) ** 2, axis=2)
    
    # DP to find minimum cost path from top to bottom
    dp = np.full((height, overlap_width), np.inf, dtype=np.float32)
    parent = np.zeros((height, overlap_width), dtype=np.int32)
    
    # Initialize first row
    dp[0, :] = cost[0, :]
    
    # Fill DP table (can move ±1 horizontally per row)
    for y in range(1, height):
        for x in range(overlap_width):
            best_cost = np.inf
            best_parent = x
            
            for dx in [-1, 0, 1]:
                px = x + dx
                if 0 <= px < overlap_width:
                    if dp[y-1, px] < best_cost:
                        best_cost = dp[y-1, px]
                        best_parent = px
            
            dp[y, x] = best_cost + cost[y, x]
            parent[y, x] = best_parent
    
    # Backtrack to find seam
    seam = np.zeros(height, dtype=np.int32)
    seam[-1] = np.argmin(dp[-1, :])
    for y in range(height - 2, -1, -1):
        seam[y] = parent[y + 1, seam[y + 1]]
    return seam


def generate_xslit_panorama(warped_frames, bounding_boxes, slope, reference_idx, 
                            canvas_width, canvas_height, tilt=0.0, blend_mode='pyramid'):
    """
    Generate X-Slit panorama with selectable blending mode.
    
    Args:
        warped_frames: List of warped frame images
        bounding_boxes: Bounding boxes for each frame
        slope: X-Slit slope parameter
        reference_idx: Reference frame index
        canvas_width, canvas_height: Output canvas size
        tilt: Horizontal bias for sampling
        blend_mode: 'pyramid' for averaging, 'mincut' for min-cut seam finding
    
    Returns:
        Panorama image (canvas_height x canvas_width x 3)
    """
    if blend_mode == 'mincut':
        return _generate_xslit_panorama_mincut(
            warped_frames, bounding_boxes, slope, reference_idx,
            canvas_width, canvas_height, tilt
        )
    else:
        return _generate_xslit_panorama_pyramid(
            warped_frames, bounding_boxes, slope, reference_idx,
            canvas_width, canvas_height, tilt
        )


def _generate_xslit_panorama_pyramid(warped_frames, bounding_boxes, slope, reference_idx, 
                                     canvas_width, canvas_height, tilt=0.0):
    """
    Generate X-Slit panorama using logic ported strictly from ex4.py.
    """
    n_frames = len(warped_frames)
    
    # 1. Initialize Accumulators (Float32)
    acc_odd = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    w_odd = np.zeros((canvas_height, canvas_width, 1), dtype=np.float32)
    
    acc_even = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    w_even = np.zeros((canvas_height, canvas_width, 1), dtype=np.float32)

    # Mask: 1.0 = Odd, 0.0 = Even. We fill this on the fly.
    barcode_mask = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    for i in range(n_frames):
        warped_image = warped_frames[i]
        _, warped_w = warped_image.shape[:2]
        
        # --- 1. Strip Geometry (Dynamic) ---
        curr_x = int(bounding_boxes[i][0, 0])
        curr_y = int(bounding_boxes[i][0, 1])
        
        next_idx = min(i + 2, n_frames - 1)
        if i == n_frames - 1:
            gap_width = 20 
        else:
            next_x = int(bounding_boxes[next_idx][0, 0])
            gap_width = next_x - curr_x
        required_width = max(1, gap_width + 2, int(abs(slope)) + 2)
        
        frame_offset = i - reference_idx
        center_factor = 0.5 + tilt
        local_center = int(warped_w * center_factor)
        sample_center = local_center + int(slope * frame_offset)
        
        src_start = sample_center - (required_width // 2)
        src_end = src_start + required_width
        
        if src_start < 0 or src_end > warped_w: continue
        strip = warped_image[:, src_start:src_end].astype(np.float32)
        if strip.shape[1] == 0: continue

        # --- 2. Canvas Mapping ---
        dst_x_start = curr_x
        dst_x_end = dst_x_start + strip.shape[1]
        dst_y_start = curr_y
        dst_y_end = dst_y_start + strip.shape[0]

        canvas_x1 = max(0, dst_x_start)
        canvas_x2 = min(canvas_width, dst_x_end)
        canvas_y1 = max(0, dst_y_start)
        canvas_y2 = min(canvas_height, dst_y_end)
        
        if canvas_x1 >= canvas_x2 or canvas_y1 >= canvas_y2: continue
        
        strip_x1 = canvas_x1 - dst_x_start
        strip_x2 = strip_x1 + (canvas_x2 - canvas_x1)
        strip_y1 = canvas_y1 - dst_y_start
        strip_y2 = strip_y1 + (canvas_y2 - canvas_y1)
        
        clipped_strip = strip[strip_y1:strip_y2, strip_x1:strip_x2]
        
        target_w = canvas_x2 - canvas_x1
        target_h = canvas_y2 - canvas_y1
        if clipped_strip.shape[1] != target_w or clipped_strip.shape[0] != target_h:
            clipped_strip = cv2.resize(clipped_strip, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # --- 3. Accumulate & Mask ---
        if i % 2 != 0: # Odd
            acc_odd[canvas_y1:canvas_y2, canvas_x1:canvas_x2] += clipped_strip
            w_odd[canvas_y1:canvas_y2, canvas_x1:canvas_x2] += 1.0
            # Paint Mask: Odd "wins" this region
            barcode_mask[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = 1.0
        else: # Even
            acc_even[canvas_y1:canvas_y2, canvas_x1:canvas_x2] += clipped_strip
            w_even[canvas_y1:canvas_y2, canvas_x1:canvas_x2] += 1.0
            # Paint Mask: Even "wins" this region
            barcode_mask[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = 0.0

    
    # Odd Mosaic
    valid_odd = w_odd > 0
    mosaic_odd = np.zeros_like(acc_odd)
    mosaic_odd[np.repeat(valid_odd, 3, axis=2)] = (acc_odd / np.maximum(w_odd, 1))[np.repeat(valid_odd, 3, axis=2)]
    
    # Even Mosaic
    valid_even = w_even > 0
    mosaic_even = np.zeros_like(acc_even)
    mosaic_even[np.repeat(valid_even, 3, axis=2)] = (acc_even / np.maximum(w_even, 1))[np.repeat(valid_even, 3, axis=2)]

    # --- 5. Hole Filling (EXACTLY AS IN EX4.PY) ---
    only_odd = valid_odd & ~valid_even
    only_even = valid_even & ~valid_odd
    barcode_mask[only_odd[:, :, 0]] = 1.0
    barcode_mask[only_even[:, :, 0]] = 0.0
    
    # --- 6. Blend ---
    blender = PyramidBlender(levels=3)
    result = blender.blend(mosaic_odd, mosaic_even, barcode_mask)

    return result


def _generate_xslit_panorama_mincut(warped_frames, bounding_boxes, slope, reference_idx,
                                    canvas_width, canvas_height, tilt=0.0):
    """
    Generate X-Slit panorama with min-cut seam finding.
    Same strip sampling as pyramid, but uses optimal seams instead of blending.
    """
    n_frames = len(warped_frames)
    
    # Collect all strips first
    strips = []
    
    for i in range(n_frames):
        warped_image = warped_frames[i]
        _, warped_w = warped_image.shape[:2]
        
        curr_x = int(bounding_boxes[i][0, 0])
        curr_y = int(bounding_boxes[i][0, 1])
        
        # Double width lookahead
        next_idx = min(i + 2, n_frames - 1)
        
        if i == n_frames - 1:
            gap_width = 20
        else:
            next_x = int(bounding_boxes[next_idx][0, 0])
            gap_width = next_x - curr_x
        
        required_width = max(1, gap_width + 2, int(abs(slope)) + 2)
        
        # X-Slit sampling position
        frame_offset = i - reference_idx
        center_factor = 0.5 + tilt
        local_center = int(warped_w * center_factor)
        sample_center = local_center + int(slope * frame_offset)
        
        src_start = sample_center - (required_width // 2)
        src_end = src_start + required_width
        
        if src_start < 0 or src_end > warped_w:
            continue
        
        strip = warped_image[:, src_start:src_end].copy()
        if strip.shape[1] == 0:
            continue
        
        # Canvas coordinates
        dst_x_start = curr_x
        dst_x_end = dst_x_start + strip.shape[1]
        dst_y_start = curr_y
        dst_y_end = dst_y_start + strip.shape[0]
        
        # Clip to canvas
        canvas_x1 = max(0, dst_x_start)
        canvas_x2 = min(canvas_width, dst_x_end)
        canvas_y1 = max(0, dst_y_start)
        canvas_y2 = min(canvas_height, dst_y_end)
        
        if canvas_x1 >= canvas_x2 or canvas_y1 >= canvas_y2:
            continue
        
        # Clip strip
        strip_x1 = canvas_x1 - dst_x_start
        strip_x2 = strip_x1 + (canvas_x2 - canvas_x1)
        strip_y1 = canvas_y1 - dst_y_start
        strip_y2 = strip_y1 + (canvas_y2 - canvas_y1)
        
        clipped_strip = strip[strip_y1:strip_y2, strip_x1:strip_x2]
        
        strips.append({
            'image': clipped_strip,
            'x1': canvas_x1,
            'x2': canvas_x2,
            'y1': canvas_y1,
            'y2': canvas_y2,
            'frame_idx': i
        })
    
    if len(strips) == 0:
        return np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Composite with min-cut seams
    result = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Place first strip directly
    s = strips[0]
    result[s['y1']:s['y2'], s['x1']:s['x2']] = s['image']
    
    for i in range(1, len(strips)):
        curr = strips[i]
        prev = strips[i - 1]
        
        # Find overlap region
        overlap_x1 = max(curr['x1'], prev['x1'])
        overlap_x2 = min(curr['x2'], prev['x2'])
        overlap_y1 = max(curr['y1'], prev['y1'])
        overlap_y2 = min(curr['y2'], prev['y2'])
        
        overlap_width = overlap_x2 - overlap_x1
        overlap_height = overlap_y2 - overlap_y1
        
        if overlap_width <= 2 or overlap_height <= 10:
            # No meaningful overlap, just paste
            result[curr['y1']:curr['y2'], curr['x1']:curr['x2']] = curr['image']
            continue
        
        # Extract overlapping portions
        prev_in_overlap_x1 = overlap_x1 - prev['x1']
        prev_in_overlap_x2 = overlap_x2 - prev['x1']
        prev_in_overlap_y1 = overlap_y1 - prev['y1']
        prev_in_overlap_y2 = overlap_y2 - prev['y1']
        
        curr_in_overlap_x1 = overlap_x1 - curr['x1']
        curr_in_overlap_x2 = overlap_x2 - curr['x1']
        curr_in_overlap_y1 = overlap_y1 - curr['y1']
        curr_in_overlap_y2 = overlap_y2 - curr['y1']
        
        prev_overlap = prev['image'][prev_in_overlap_y1:prev_in_overlap_y2,
                                     prev_in_overlap_x1:prev_in_overlap_x2]
        curr_overlap = curr['image'][curr_in_overlap_y1:curr_in_overlap_y2,
                                     curr_in_overlap_x1:curr_in_overlap_x2]
        
        # Find optimal seam using min-cut
        seam = find_mincut_seam_dp(prev_overlap, curr_overlap, overlap_width)
        
        # Paste non-overlap part of current strip (right side)
        if curr['x2'] > overlap_x2:
            non_overlap_x1 = overlap_x2 - curr['x1']
            result[curr['y1']:curr['y2'], overlap_x2:curr['x2']] = curr['image'][:, non_overlap_x1:]
        
        # Handle overlap region with seam
        for y_local in range(overlap_height):
            y_canvas = overlap_y1 + y_local
            seam_x_local = seam[y_local]
            seam_x_canvas = overlap_x1 + seam_x_local
            
            # Right of seam: use current strip
            if seam_x_canvas < curr['x2']:
                curr_local_x = seam_x_canvas - curr['x1']
                curr_local_y = y_canvas - curr['y1']
                
                if 0 <= curr_local_y < curr['image'].shape[0]:
                    end_x = curr['x2']
                    curr_end_local = end_x - curr['x1']
                    
                    if 0 <= curr_local_x < curr_end_local:
                        result[y_canvas, seam_x_canvas:end_x] = \
                            curr['image'][curr_local_y, curr_local_x:curr_end_local]
    
    return result


def find_content_bounds(panorama):
    """Find the bounding box of non-black content."""
    mask = np.any(panorama > 5, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]
    
    if len(row_idx) == 0 or len(col_idx) == 0:
        return 0, panorama.shape[1], 0, panorama.shape[0]
    
    return col_idx[0], col_idx[-1]+1, row_idx[0], row_idx[-1]+1


def generate_forward_walkthrough(frames, homographies, num_output_frames=60, 
                                 slope_start=0.0, slope_end=2.0, 
                                 zoom_start=1.0, zoom_end=1.5, tilt=0.0,
                                 blend_mode='pyramid', 
                                 vertical_stretch_start=1.0, vertical_stretch_end=1.0):
    """
    Generate a sequence of panoramas with varying X-Slit slopes and zoom.
    
    Args:
        frames: List of input video frames
        homographies: List of accumulated homographies
        num_output_frames: Number of output frames to generate
        slope_start, slope_end: X-Slit slope range
        zoom_start, zoom_end: Zoom factor range
        tilt: Horizontal bias for sampling
        blend_mode: 'pyramid' for averaging, 'mincut' for DP seam finding
    
    Returns:
        List of output frames
    """
    n_frames = len(frames)
    h, w = frames[0].shape[:2]
    reference_idx = n_frames // 2
    
    print(f"\n{'='*60}")
    print("X-SLIT FORWARD WALKTHROUGH GENERATION")
    print(f"{'='*60}")
    print(f"Input frames: {n_frames}, Output frames: {num_output_frames}")
    print(f"Slope range: {slope_start} -> {slope_end}")
    print(f"Zoom range: {zoom_start} -> {zoom_end}")
    print(f"Tilt: {tilt}")
    print(f"Blend mode: {blend_mode}")
    
    # Compute bounding boxes
    print("\nComputing bounding boxes...")
    bounding_boxes = np.zeros((n_frames, 2, 2))
    for i in range(n_frames):
        bounding_boxes[i] = compute_bounding_box(homographies[i], w, h)
    
    global_offset = np.min(bounding_boxes, axis=(0, 1))
    bounding_boxes -= global_offset
    
    canvas_size = np.max(bounding_boxes, axis=(0, 1)).astype(int) + 1
    canvas_width, canvas_height = canvas_size[0], canvas_size[1]
    
    print(f"Canvas size: {canvas_width} x {canvas_height}")
    
    # Pre-warp all frames
    print("Warping frames...")
    warped_frames = []
    for i, frame in enumerate(frames):
        warped_image, _ = warp_image(frame, homographies[i])
        warped_frames.append(warped_image)
        if (i + 1) % 50 == 0:
            print(f"  Warped {i+1}/{n_frames} frames")
    
    print(f"\nGenerating {num_output_frames} X-Slit panoramas...")
    slopes = np.linspace(slope_start, slope_end, num_output_frames)
    zooms = np.linspace(zoom_start, zoom_end, num_output_frames)
    
    # --- NEW: Interpolate vertical stretch ---
    v_stretches = np.linspace(vertical_stretch_start, vertical_stretch_end, num_output_frames)

    raw_panoramas = []
    for k, slope in enumerate(slopes):
        panorama = generate_xslit_panorama(
            warped_frames, bounding_boxes, slope,
            reference_idx, canvas_width, canvas_height,
            tilt=tilt, blend_mode=blend_mode
        )
        raw_panoramas.append(panorama)
        if (k + 1) % 10 == 0:
            print(f"  Generated {k+1}/{num_output_frames} (slope={slope:.2f})")

    # Lock size to first frame (Same as your original logic)
    first_pano = raw_panoramas[0]
    crop_left, crop_right, crop_top, crop_bottom = find_content_bounds(first_pano)
    content_w = crop_right - crop_left
    content_h = crop_bottom - crop_top
    content_center_x = (crop_left + crop_right) // 2
    content_center_y = (crop_top + crop_bottom) // 2
    output_w, output_h = content_w, content_h

    print("\nApplying zoom and aspect ratio correction...")
    output_frames = []
    
    for k, (pano, zoom, v_scale) in enumerate(zip(raw_panoramas, zooms, v_stretches)):
        
        # 1. Apply Aspect Ratio Correction (Vertical Scaling)
        # The paper says to scale vertically to fix distortion [cite: 428]
        if abs(v_scale - 1.0) > 0.01:
            h_curr, w_curr = pano.shape[:2]
            new_h = int(h_curr * v_scale)
            # Resize strictly the height, keeping width the same
            pano = cv2.resize(pano, (w_curr, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Re-calculate center Y because image height changed
            current_center_y = int(content_center_y * v_scale)
        else:
            current_center_y = content_center_y

        # 2. Apply Zoom (Cropping)
        crop_w_zoomed = int(content_w / zoom)
        crop_h_zoomed = int(content_h / zoom) # Note: Zoom is uniform, v_scale was already applied to the image
        
        z_left = max(0, content_center_x - crop_w_zoomed // 2)
        z_right = min(pano.shape[1], z_left + crop_w_zoomed)
        
        z_top = max(0, current_center_y - crop_h_zoomed // 2)
        z_bottom = min(pano.shape[0], z_top + crop_h_zoomed)
        
        cropped = pano[z_top:z_bottom, z_left:z_right]
        
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            # Resize back to fixed output container
            zoomed = cv2.resize(cropped, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
            output_frames.append(zoomed)
        else:
            output_frames.append(np.zeros((output_h, output_w, 3), dtype=np.uint8))
            
    print("Done!")
    return output_frames


def generate_forward_panorama_video(video_path, num_output_frames=60,
                                    slope_start=0.0, slope_end=2.0,
                                    zoom_start=1.0, zoom_end=1.5,
                                    reverse=False, tilt=0.0, blend_mode='pyramid',
                                    vertical_stretch_start=1.0, vertical_stretch_end=1.0):
    """
    Main entry point for bonus forward panorama effect.
    
    Args:
        video_path: Path to input video
        num_output_frames: Number of output frames
        slope_start, slope_end: X-Slit slope range
        zoom_start, zoom_end: Zoom factor range
        reverse: If True, reverse output for backward motion
        tilt: Horizontal bias (-0.2 to 0.2 typical)
        blend_mode: 'pyramid' for averaging, 'mincut' for DP seam finding
    
    Returns:
        List of output frames
    """
    print("="*60)
    print("FORWARD PANORAMA (X-SLIT) GENERATION")
    print("="*60)
    
    # Load and prepare frames
    frames = load_video_frames(video_path)
    print(f"Loaded {len(frames)} frames")
    
    # Detect direction and fix if needed
    frames = detect_and_fix_direction(frames)
    
    # Compute homographies
    print("Computing transforms...")
    H_successive = []
    for i in range(len(frames) - 1):
        H = estimate_horizon_homography(frames[i], frames[i + 1])
        H_successive.append(H)
        if (i + 1) % 50 == 0:
            print(f"  Computed {i+1}/{len(frames)-1} transforms")
    
    reference_idx = len(frames) // 2
    print(f"Accumulating to reference frame {reference_idx}...")
    homographies = accumulate_homographies(H_successive, reference_idx)
    
    # Generate walkthrough
    output_frames = generate_forward_walkthrough(
        frames, homographies, num_output_frames,
        slope_start, slope_end,
        zoom_start, zoom_end,
        tilt=tilt, blend_mode=blend_mode,
        vertical_stretch_start=vertical_stretch_start, 
        vertical_stretch_end=vertical_stretch_end
    )
    
    # Reverse if requested (backward motion)
    if reverse:
        output_frames = output_frames[::-1]
    
    # Pad for video encoding
    output_frames = [pad_to_macro_block(f) for f in output_frames]
    
    return output_frames


# =============================================================================
# Demo / Testing
# =============================================================================

if __name__ == "__main__":
    video_name = "boat"
    video_path = f"Ex4/inputs/{video_name}.mp4" 
    
    params = dict(
    num_output_frames=120,
    slope_start=0.0,
    slope_end=1.5,
    zoom_start=1.0, 
    zoom_end=1.5,
    
    # NEW PARAMS
    # Start at 1.0 (no distortion at slope 0)
    # End at 0.85 (shrink height by 15% as we move forward)
    vertical_stretch_start=1.0,
    vertical_stretch_end=0.85 
)
    
    # Test pyramid blending
    print("\n" + "="*70)
    print("GENERATING WITH PYRAMID BLENDING")
    print("="*70)
    start_time = time.time()
    output_pyramid = generate_forward_panorama_video(video_path, **params, blend_mode='pyramid')
    output_pyramid = np.concatenate([output_pyramid, output_pyramid[::-1]], axis=0)
    pyramid_time = time.time() - start_time
    
    output_path = f"Ex4/{video_name}_xslit_barcode.mp4"
    iio.imwrite(output_path, output_pyramid, fps=60)
    print(f"\nSaved to {output_path}")
    print(f"Pyramid blending time: {pyramid_time:.1f}s")
    
    # Test mincut blending
    print("\n" + "="*70)
    print("GENERATING WITH MINCUT SEAMS")
    print("="*70)
    start_time = time.time()
    output_mincut = generate_forward_panorama_video(video_path, **params, blend_mode='mincut')
    output_mincut = np.concatenate([output_mincut, output_mincut[::-1]], axis=0)
    mincut_time = time.time() - start_time
    
    output_path = f"Ex4/{video_name}_xslit_mincut.mp4"
    iio.imwrite(output_path, output_mincut, fps=60)
    print(f"\nSaved to {output_path}")
    print(f"Mincut seams time: {mincut_time:.1f}s")
    
    # Summary
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(f"Pyramid blending: {pyramid_time:.1f}s")
    print(f"Mincut seams:     {mincut_time:.1f}s")
    print(f"Difference:       {mincut_time - pyramid_time:+.1f}s")