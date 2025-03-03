import numpy as np
from numba import njit, prange
import cv2 

@njit(parallel=True, fastmath=True)
def rescale_image_numba(image, factor):
    """
    Custom rescale function using bilinear interpolation optimized with numba.
    """
    height, width = image.shape[:2]
    new_width = int(width * factor)
    new_height = int(height * factor)

    # Create an empty output array
    output = np.zeros((new_height, new_width, image.shape[2]), dtype=np.float32)

    for y in prange(new_height):
        for x in prange(new_width):
            src_x = x / factor
            src_y = y / factor
            x0, y0 = int(src_x), int(src_y)
            x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)

            # Bilinear interpolation
            dx, dy = src_x - x0, src_y - y0
            for c in range(image.shape[2]):  # Loop over channels
                output[y, x, c] = (
                    image[y0, x0, c] * (1 - dx) * (1 - dy)
                    + image[y0, x1, c] * dx * (1 - dy)
                    + image[y1, x0, c] * (1 - dx) * dy
                    + image[y1, x1, c] * dx * dy
                )

    return output

@njit(parallel=True, fastmath=True)
def flip_horizontal_numba(image):
    """
    Flips an image horizontally using Numba JIT.
    """
    height, width, channels = image.shape
    output = np.empty_like(image)
    for y in prange(height):
        for x in prange(width):
            output[y, x] = image[y, width - x - 1]
    return output

@njit(parallel=True, fastmath=True)
def flip_vertical_numba(image):
    """
    Flips an image vertically using Numba JIT.
    """
    height, width, channels = image.shape
    output = np.empty_like(image)
    for y in prange(height):
        output[y] = image[height - y - 1]
    return output

@njit(parallel=True, fastmath=True)
def rotate_90_clockwise_numba(image):
    """
    Rotates the image 90 degrees clockwise.
    """
    height, width, channels = image.shape
    output = np.empty((width, height, channels), dtype=image.dtype)
    for y in prange(height):
        for x in prange(width):
            output[x, height - 1 - y] = image[y, x]
    return output

@njit(parallel=True, fastmath=True)
def rotate_90_counterclockwise_numba(image):
    """
    Rotates the image 90 degrees counterclockwise.
    """
    height, width, channels = image.shape
    output = np.empty((width, height, channels), dtype=image.dtype)
    for y in prange(height):
        for x in prange(width):
            output[width - 1 - x, y] = image[y, x]
    return output

@njit(parallel=True, fastmath=True)
def invert_image_numba(image):
    """
    Inverts an image (1 - pixel value).
    """
    output = np.empty_like(image)
    for y in prange(image.shape[0]):
        for x in prange(image.shape[1]):
            for c in prange(image.shape[2]):
                output[y, x, c] = 1.0 - image[y, x, c]
    return output


@njit(parallel=True, fastmath=True)
def apply_flat_division_numba_2d(image, master_flat, master_bias=None):
    """
    Mono version: image.shape == (H,W)
    """
    if master_bias is not None:
        master_flat = master_flat - master_bias
        image = image - master_bias

    median_flat = np.median(master_flat)
    height, width = image.shape

    for y in prange(height):
        for x in range(width):
            image[y, x] /= (master_flat[y, x] / median_flat)

    return image


@njit(parallel=True, fastmath=True)
def apply_flat_division_numba_3d(image, master_flat, master_bias=None):
    """
    Color version: image.shape == (H,W,C)
    """
    if master_bias is not None:
        master_flat = master_flat - master_bias
        image = image - master_bias

    median_flat = np.median(master_flat)
    height, width, channels = image.shape

    for y in prange(height):
        for x in range(width):
            for c in range(channels):
                image[y, x, c] /= (master_flat[y, x, c] / median_flat)

    return image

def apply_flat_division_numba(image, master_flat, master_bias=None):
    """
    Dispatcher that calls the correct Numba function
    depending on whether 'image' is 2D or 3D.
    """
    if image.ndim == 2:
        # Mono
        return apply_flat_division_numba_2d(image, master_flat, master_bias)
    elif image.ndim == 3:
        # Color
        return apply_flat_division_numba_3d(image, master_flat, master_bias)
    else:
        raise ValueError(f"apply_flat_division_numba: expected 2D or 3D, got shape {image.shape}")


@njit(parallel=True)
def subtract_dark_3d(frames, dark_frame):
    """
    For mono stack:
      frames.shape == (F,H,W)
      dark_frame.shape == (H,W)
    Returns the same shape (F,H,W).
    """
    num_frames, height, width = frames.shape
    result = np.empty_like(frames, dtype=np.float32)

    for i in prange(num_frames):
        # Subtract the dark frame from each 2D slice
        result[i] = frames[i] - dark_frame

    return result


@njit(parallel=True)
def subtract_dark_4d(frames, dark_frame):
    """
    For color stack:
      frames.shape == (F,H,W,C)
      dark_frame.shape == (H,W,C)
    Returns the same shape (F,H,W,C).
    """
    num_frames, height, width, channels = frames.shape
    result = np.empty_like(frames, dtype=np.float32)

    for i in prange(num_frames):
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    result[i, y, x, c] = frames[i, y, x, c] - dark_frame[y, x, c]

    return result

def subtract_dark(frames, dark_frame):
    """
    Dispatcher function that calls the correct Numba function
    depending on whether 'frames' is 3D or 4D.
    """
    if frames.ndim == 3:
        # frames: (F,H,W), dark_frame: (H,W)
        return subtract_dark_3d(frames, dark_frame)
    elif frames.ndim == 4:
        # frames: (F,H,W,C), dark_frame: (H,W,C)
        return subtract_dark_4d(frames, dark_frame)
    else:
        raise ValueError(f"subtract_dark: frames must be 3D or 4D, got {frames.shape}")


@njit(parallel=True, fastmath=True)
def windsorized_sigma_clip_weighted_3d(stack, weights, lower=2.5, upper=2.5):
    """
    Weighted Windsorized Sigma Clipping for a 3D mono stack:
      stack.shape == (F,H,W)
      weights.shape can be (F,) or (F,H,W).
    Returns a 2D clipped image (H,W).
    """
    num_frames, height, width = stack.shape
    clipped = np.zeros((height, width), dtype=np.float32)

    # Check shape of weights
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 3 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("windsorized_sigma_clip_weighted_3d: mismatch in shapes for 3D stack & weights")

    for i in prange(height):
        for j in range(width):
            pixel_values = stack[:, i, j]  # shape=(F,)

            # Figure out corresponding weights
            if weights.ndim == 1:
                pixel_weights = weights[:]
            else:
                pixel_weights = weights[:, i, j]

            median_val = np.median(pixel_values)
            std_dev = np.std(pixel_values)
            lower_bound = median_val - lower * std_dev
            upper_bound = median_val + upper * std_dev

            valid_mask = (pixel_values != 0) & (pixel_values >= lower_bound) & (pixel_values <= upper_bound)
            valid_vals = pixel_values[valid_mask]
            valid_w = pixel_weights[valid_mask]

            wsum = np.sum(valid_w)
            if wsum > 0:
                clipped[i, j] = np.sum(valid_vals * valid_w) / wsum
            else:
                clipped[i, j] = median_val

    return clipped

@njit(parallel=True, fastmath=True)
def windsorized_sigma_clip_weighted_4d(stack, weights, lower=2.5, upper=2.5):
    """
    Weighted Windsorized Sigma Clipping for a 4D color stack:
      stack.shape == (F,H,W,C)
      weights.shape can be (F,) or (F,H,W,C).
    Returns a 3D clipped image (H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.zeros((height, width, channels), dtype=np.float32)

    # Check shape of weights
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 4 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("windsorized_sigma_clip_weighted_4d: mismatch in shapes for 4D stack & weights")

    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                pixel_values = stack[:, i, j, c]  # shape=(F,)

                if weights.ndim == 1:
                    pixel_weights = weights[:]
                else:
                    pixel_weights = weights[:, i, j, c]

                median_val = np.median(pixel_values)
                std_dev = np.std(pixel_values)
                lower_bound = median_val - lower * std_dev
                upper_bound = median_val + upper * std_dev

                valid_mask = (pixel_values != 0) & (pixel_values >= lower_bound) & (pixel_values <= upper_bound)
                valid_vals = pixel_values[valid_mask]
                valid_w = pixel_weights[valid_mask]

                wsum = np.sum(valid_w)
                if wsum > 0:
                    clipped[i, j, c] = np.sum(valid_vals * valid_w) / wsum
                else:
                    clipped[i, j, c] = median_val

    return clipped

def windsorized_sigma_clip_weighted(stack, weights, lower=2.5, upper=2.5):
    """
    Dispatcher that calls either the 3D or 4D Numba function
    depending on 'stack.ndim'.
    """
    if stack.ndim == 3:
        return windsorized_sigma_clip_weighted_3d(stack, weights, lower, upper)
    elif stack.ndim == 4:
        return windsorized_sigma_clip_weighted_4d(stack, weights, lower, upper)
    else:
        raise ValueError(f"windsorized_sigma_clip_weighted: stack must be 3D or 4D, got {stack.shape}")


@njit(parallel=True, fastmath=True)
def kappa_sigma_clip_weighted_3d(stack, weights, kappa=2.5, iterations=3):
    """
    Kappa-Sigma Clipping for a 3D mono stack:
      stack.shape == (F,H,W)
      weights can be (F,) or (F,H,W)
    Returns a 2D clipped result (H,W).
    """
    num_frames, height, width = stack.shape
    clipped = np.empty((height, width), dtype=np.float32)

    # Validate weights shape
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 3 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("kappa_sigma_clip_weighted_3d: mismatch in shapes for 3D stack & weights")

    for i in prange(height):
        for j in range(width):
            pixel_values = stack[:, i, j].copy()
            if weights.ndim == 1:
                pixel_weights = weights[:]
            else:
                pixel_weights = weights[:, i, j].copy()

            current_vals = pixel_values
            current_w = pixel_weights
            med = 0.0  # keep track of last median

            for _ in range(iterations):
                if current_vals.size == 0:
                    break
                med = np.median(current_vals)
                std = np.std(current_vals)
                lower_bound = med - kappa * std
                upper_bound = med + kappa * std
                valid = (current_vals != 0) & (current_vals >= lower_bound) & (current_vals <= upper_bound)
                current_vals = current_vals[valid]
                current_w = current_w[valid]

            if current_w.size > 0 and current_w.sum() > 0:
                clipped[i, j] = np.sum(current_vals * current_w) / current_w.sum()
            else:
                clipped[i, j] = med  # fallback to last median

    return clipped

@njit(parallel=True, fastmath=True)
def kappa_sigma_clip_weighted_4d(stack, weights, kappa=2.5, iterations=3):
    """
    Kappa-Sigma Clipping for a 4D color stack:
      stack.shape == (F,H,W,C)
      weights can be (F,) or (F,H,W,C)
    Returns a 3D clipped result (H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.empty((height, width, channels), dtype=np.float32)

    # Validate weights shape
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 4 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("kappa_sigma_clip_weighted_4d: mismatch in shapes for 4D stack & weights")

    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                pixel_values = stack[:, i, j, c].copy()
                if weights.ndim == 1:
                    pixel_weights = weights[:]
                else:
                    pixel_weights = weights[:, i, j, c].copy()

                current_vals = pixel_values
                current_w = pixel_weights
                med = 0.0

                for _ in range(iterations):
                    if current_vals.size == 0:
                        break
                    med = np.median(current_vals)
                    std = np.std(current_vals)
                    lower_bound = med - kappa * std
                    upper_bound = med + kappa * std
                    valid = (current_vals != 0) & (current_vals >= lower_bound) & (current_vals <= upper_bound)
                    current_vals = current_vals[valid]
                    current_w = current_w[valid]

                if current_w.size > 0 and current_w.sum() > 0:
                    clipped[i, j, c] = np.sum(current_vals * current_w) / current_w.sum()
                else:
                    clipped[i, j, c] = med

    return clipped

def kappa_sigma_clip_weighted(stack, weights, kappa=2.5, iterations=3):
    """
    Dispatcher function that calls either the 3D or 4D specialized Numba function.
    """
    if stack.ndim == 3:
        return kappa_sigma_clip_weighted_3d(stack, weights, kappa, iterations)
    elif stack.ndim == 4:
        return kappa_sigma_clip_weighted_4d(stack, weights, kappa, iterations)
    else:
        raise ValueError(f"kappa_sigma_clip_weighted: stack must be 3D or 4D, got {stack.shape}")


@njit(parallel=True, fastmath=True)
def trimmed_mean_weighted_3d(stack, weights, trim_fraction=0.1):
    """
    Trimmed Mean for a 3D mono stack:
      stack.shape == (F,H,W)
      weights can be (F,) or (F,H,W)
    Returns a 2D result (H,W).

    'trim_fraction' is the fraction of lowest and highest values to remove.
    """
    num_frames, height, width = stack.shape
    clipped = np.empty((height, width), dtype=np.float32)

    # Validate weights shape
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 3 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("trimmed_mean_weighted_3d: mismatch in shapes for 3D stack & weights")

    for i in prange(height):
        for j in range(width):
            pix = stack[:, i, j]
            # Determine per-pixel weights
            if weights.ndim == 1:
                w = weights[:]
            else:
                w = weights[:, i, j]

            # Exclude zeros
            valid_mask = pix != 0
            pix = pix[valid_mask]
            w = w[valid_mask]
            n = pix.size

            if n == 0:
                clipped[i, j] = 0.0
            else:
                trim = int(trim_fraction * n)
                idx = np.argsort(pix)
                if n > 2 * trim:
                    trimmed_values = pix[idx][trim : n - trim]
                    trimmed_weights = w[idx][trim : n - trim]
                else:
                    trimmed_values = pix[idx]
                    trimmed_weights = w[idx]

                wsum = trimmed_weights.sum()
                if wsum > 0:
                    clipped[i, j] = np.sum(trimmed_values * trimmed_weights) / wsum
                else:
                    clipped[i, j] = np.median(trimmed_values)

    return clipped

@njit(parallel=True, fastmath=True)
def trimmed_mean_weighted_4d(stack, weights, trim_fraction=0.1):
    """
    Trimmed Mean for a 4D color stack:
      stack.shape == (F,H,W,C)
      weights can be (F,) or (F,H,W,C)
    Returns a 3D result (H,W,C).

    'trim_fraction' is the fraction of lowest and highest values to remove.
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.empty((height, width, channels), dtype=np.float32)

    # Validate weights shape
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 4 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("trimmed_mean_weighted_4d: mismatch in shapes for 4D stack & weights")

    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                pix = stack[:, i, j, c]
                # Determine per-pixel weights
                if weights.ndim == 1:
                    w = weights[:]
                else:
                    w = weights[:, i, j, c]

                # Exclude zeros
                valid_mask = pix != 0
                pix = pix[valid_mask]
                w = w[valid_mask]
                n = pix.size

                if n == 0:
                    clipped[i, j, c] = 0.0
                else:
                    trim = int(trim_fraction * n)
                    idx = np.argsort(pix)
                    if n > 2 * trim:
                        trimmed_values = pix[idx][trim : n - trim]
                        trimmed_weights = w[idx][trim : n - trim]
                    else:
                        trimmed_values = pix[idx]
                        trimmed_weights = w[idx]

                    wsum = trimmed_weights.sum()
                    if wsum > 0:
                        clipped[i, j, c] = np.sum(trimmed_values * trimmed_weights) / wsum
                    else:
                        clipped[i, j, c] = np.median(trimmed_values)

    return clipped

def trimmed_mean_weighted(stack, weights, trim_fraction=0.1):
    """
    Dispatcher that calls either the 3D or 4D specialized Numba function
    depending on 'stack.ndim'.
    """
    if stack.ndim == 3:
        return trimmed_mean_weighted_3d(stack, weights, trim_fraction)
    elif stack.ndim == 4:
        return trimmed_mean_weighted_4d(stack, weights, trim_fraction)
    else:
        raise ValueError(f"trimmed_mean_weighted: stack must be 3D or 4D, got {stack.shape}")


@njit(parallel=True, fastmath=True)
def esd_clip_weighted_3d(stack, weights, threshold=3.0):
    """
    Extreme Studentized Deviate (ESD) Clipping for a 3D mono stack:
      stack.shape == (F,H,W)
      weights can be (F,) or (F,H,W)
    Returns a 2D result (H,W).

    threshold is the z-score cutoff.
    """
    num_frames, height, width = stack.shape
    clipped = np.empty((height, width), dtype=np.float32)

    # Validate weights shape
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 3 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("esd_clip_weighted_3d: mismatch in shapes for 3D stack & weights")

    for i in prange(height):
        for j in range(width):
            pix = stack[:, i, j]
            if weights.ndim == 1:
                w = weights[:]
            else:
                w = weights[:, i, j]

            # Exclude zeros
            valid_mask = pix != 0
            values = pix[valid_mask]
            wvals = w[valid_mask]

            if values.size == 0:
                clipped[i, j] = 0.0
                continue

            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                clipped[i, j] = mean_val
                continue

            z_scores = np.abs((values - mean_val) / std_val)
            valid2 = z_scores < threshold
            values = values[valid2]
            wvals = wvals[valid2]

            wsum = wvals.sum()
            if wsum > 0:
                clipped[i, j] = np.sum(values * wvals) / wsum
            else:
                clipped[i, j] = mean_val

    return clipped

@njit(parallel=True, fastmath=True)
def esd_clip_weighted_4d(stack, weights, threshold=3.0):
    """
    Extreme Studentized Deviate (ESD) Clipping for a 4D color stack:
      stack.shape == (F,H,W,C)
      weights can be (F,) or (F,H,W,C)
    Returns a 3D result (H,W,C).

    threshold is the z-score cutoff.
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.empty((height, width, channels), dtype=np.float32)

    # Validate weights shape
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 4 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("esd_clip_weighted_4d: mismatch in shapes for 4D stack & weights")

    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                pix = stack[:, i, j, c]
                if weights.ndim == 1:
                    w = weights[:]
                else:
                    w = weights[:, i, j, c]

                # Exclude zeros
                valid_mask = pix != 0
                values = pix[valid_mask]
                wvals = w[valid_mask]

                if values.size == 0:
                    clipped[i, j, c] = 0.0
                    continue

                mean_val = np.mean(values)
                std_val = np.std(values)
                if std_val == 0:
                    clipped[i, j, c] = mean_val
                    continue

                z_scores = np.abs((values - mean_val) / std_val)
                valid2 = z_scores < threshold
                values = values[valid2]
                wvals = wvals[valid2]

                wsum = wvals.sum()
                if wsum > 0:
                    clipped[i, j, c] = np.sum(values * wvals) / wsum
                else:
                    clipped[i, j, c] = mean_val

    return clipped

def esd_clip_weighted(stack, weights, threshold=3.0):
    """
    Dispatcher that calls either the 3D or 4D specialized Numba function
    depending on 'stack.ndim'.
    """
    if stack.ndim == 3:
        return esd_clip_weighted_3d(stack, weights, threshold)
    elif stack.ndim == 4:
        return esd_clip_weighted_4d(stack, weights, threshold)
    else:
        raise ValueError(f"esd_clip_weighted: stack must be 3D or 4D, got {stack.shape}")


@njit(parallel=True, fastmath=True)
def biweight_location_weighted_3d(stack, weights, tuning_constant=6.0):
    """
    Biweight Location for a 3D mono stack:
      stack.shape == (F,H,W)
      weights can be (F,) or (F,H,W)
    Returns a 2D result (H,W).

    'tuning_constant' is the usual 6.0 for astro usage, can be tweaked.
    """
    num_frames, height, width = stack.shape
    clipped = np.empty((height, width), dtype=np.float32)

    # Validate weights shape
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 3 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("biweight_location_weighted_3d: mismatch in shapes for 3D stack & weights")

    for i in prange(height):
        for j in range(width):
            x = stack[:, i, j]
            # Extract weights
            if weights.ndim == 1:
                w = weights[:]
            else:
                w = weights[:, i, j]

            # Exclude zeros
            valid_mask = x != 0
            x = x[valid_mask]
            w = w[valid_mask]
            n = x.size

            if n == 0:
                clipped[i, j] = 0.0
                continue

            M = np.median(x)
            mad = np.median(np.abs(x - M))
            if mad == 0:
                clipped[i, j] = M
                continue

            u = (x - M) / (tuning_constant * mad)
            mask = np.abs(u) < 1
            x_masked = x[mask]
            w_masked = w[mask]
            u = u[mask]

            numerator = ((x_masked - M) * (1 - u**2)**2 * w_masked).sum()
            denominator = ((1 - u**2)**2 * w_masked).sum()
            if denominator != 0:
                biweight = M + numerator / denominator
            else:
                biweight = M

            clipped[i, j] = biweight

    return clipped

@njit(parallel=True, fastmath=True)
def biweight_location_weighted_4d(stack, weights, tuning_constant=6.0):
    """
    Biweight Location for a 4D color stack:
      stack.shape == (F,H,W,C)
      weights can be (F,) or (F,H,W,C)
    Returns a 3D result (H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.empty((height, width, channels), dtype=np.float32)

    # Validate weights shape
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 4 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("biweight_location_weighted_4d: mismatch in shapes for 4D stack & weights")

    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                x = stack[:, i, j, c]
                # Extract weights
                if weights.ndim == 1:
                    w = weights[:]
                else:
                    w = weights[:, i, j, c]

                # Exclude zeros
                valid_mask = x != 0
                x = x[valid_mask]
                w = w[valid_mask]
                n = x.size

                if n == 0:
                    clipped[i, j, c] = 0.0
                    continue

                M = np.median(x)
                mad = np.median(np.abs(x - M))
                if mad == 0:
                    clipped[i, j, c] = M
                    continue

                u = (x - M) / (tuning_constant * mad)
                mask = np.abs(u) < 1
                x_masked = x[mask]
                w_masked = w[mask]
                u = u[mask]

                numerator = ((x_masked - M) * (1 - u**2)**2 * w_masked).sum()
                denominator = ((1 - u**2)**2 * w_masked).sum()
                if denominator != 0:
                    biweight = M + numerator / denominator
                else:
                    biweight = M

                clipped[i, j, c] = biweight

    return clipped

def biweight_location_weighted(stack, weights, tuning_constant=6.0):
    """
    Dispatcher function that calls either the 3D or 4D specialized Numba function,
    depending on 'stack.ndim'.
    """
    if stack.ndim == 3:
        return biweight_location_weighted_3d(stack, weights, tuning_constant)
    elif stack.ndim == 4:
        return biweight_location_weighted_4d(stack, weights, tuning_constant)
    else:
        raise ValueError(f"biweight_location_weighted: stack must be 3D or 4D, got {stack.shape}")


@njit(parallel=True, fastmath=True)
def modified_zscore_clip_weighted_3d(stack, weights, threshold=3.5):
    """
    Modified Z-Score Clipping for a 3D mono stack:
      stack.shape == (F,H,W)
      weights can be (F,) or (F,H,W)
    Returns a 2D result (H,W).

    threshold is the z-score cutoff.
    """
    num_frames, height, width = stack.shape
    clipped = np.empty((height, width), dtype=np.float32)

    # Validate weights shape
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 3 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("modified_zscore_clip_weighted_3d: mismatch in shapes for 3D stack & weights")

    for i in prange(height):
        for j in range(width):
            x = stack[:, i, j]
            # Extract corresponding weights
            if weights.ndim == 1:
                w = weights[:]
            else:
                w = weights[:, i, j]

            # Exclude zeros
            valid_mask = x != 0
            x = x[valid_mask]
            w = w[valid_mask]
            n = x.size

            if n == 0:
                clipped[i, j] = 0.0
                continue

            median_val = np.median(x)
            mad = np.median(np.abs(x - median_val))
            if mad == 0:
                clipped[i, j] = median_val
                continue

            # Compute modified z-scores
            modified_z = 0.6745 * (x - median_val) / mad
            valid2 = np.abs(modified_z) < threshold
            x = x[valid2]
            w = w[valid2]

            wsum = w.sum()
            if wsum > 0:
                clipped[i, j] = np.sum(x * w) / wsum
            else:
                clipped[i, j] = median_val

    return clipped

@njit(parallel=True, fastmath=True)
def modified_zscore_clip_weighted_4d(stack, weights, threshold=3.5):
    """
    Modified Z-Score Clipping for a 4D color stack:
      stack.shape == (F,H,W,C)
      weights can be (F,) or (F,H,W,C)
    Returns a 3D result (H,W,C).

    threshold is the z-score cutoff.
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.empty((height, width, channels), dtype=np.float32)

    # Validate weights shape
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 4 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("modified_zscore_clip_weighted_4d: mismatch in shapes for 4D stack & weights")

    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                x = stack[:, i, j, c]
                # Extract corresponding weights
                if weights.ndim == 1:
                    w = weights[:]
                else:
                    w = weights[:, i, j, c]

                # Exclude zeros
                valid_mask = x != 0
                x = x[valid_mask]
                w = w[valid_mask]
                n = x.size

                if n == 0:
                    clipped[i, j, c] = 0.0
                    continue

                median_val = np.median(x)
                mad = np.median(np.abs(x - median_val))
                if mad == 0:
                    clipped[i, j, c] = median_val
                    continue

                # Compute modified z-scores
                modified_z = 0.6745 * (x - median_val) / mad
                valid2 = np.abs(modified_z) < threshold
                x = x[valid2]
                w = w[valid2]

                wsum = w.sum()
                if wsum > 0:
                    clipped[i, j, c] = np.sum(x * w) / wsum
                else:
                    clipped[i, j, c] = median_val

    return clipped

def modified_zscore_clip_weighted(stack, weights, threshold=3.5):
    """
    Dispatcher function that calls either the 3D or 4D specialized Numba function,
    depending on 'stack.ndim'.
    """
    if stack.ndim == 3:
        return modified_zscore_clip_weighted_3d(stack, weights, threshold)
    elif stack.ndim == 4:
        return modified_zscore_clip_weighted_4d(stack, weights, threshold)
    else:
        raise ValueError(f"modified_zscore_clip_weighted: stack must be 3D or 4D, got {stack.shape}")


@njit(parallel=True, fastmath=True)
def windsorized_sigma_clip_3d(stack, lower=2.5, upper=2.5):
    """
    Windsorized Sigma Clipping for a 3D mono stack:
      stack.shape == (F,H,W)
    Returns a 2D result (H,W).
    """
    num_frames, height, width = stack.shape
    clipped = np.zeros((height, width), dtype=np.float32)

    for i in prange(height):
        for j in range(width):
            pixel_values = stack[:, i, j]
            if pixel_values.size == 0:
                continue
            median_val = np.median(pixel_values)
            std_dev = np.std(pixel_values)

            lower_bound = median_val - lower * std_dev
            upper_bound = median_val + upper * std_dev

            valid = (pixel_values >= lower_bound) & (pixel_values <= upper_bound)
            valid_vals = pixel_values[valid]
            if valid_vals.size > 0:
                clipped[i, j] = np.mean(valid_vals)
            else:
                clipped[i, j] = median_val

    return clipped

@njit(parallel=True, fastmath=True)
def windsorized_sigma_clip_4d(stack, lower=2.5, upper=2.5):
    """
    Windsorized Sigma Clipping for a 4D color stack:
      stack.shape == (F,H,W,C)
    Returns a 3D result (H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.zeros((height, width, channels), dtype=np.float32)

    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                pixel_values = stack[:, i, j, c]
                if pixel_values.size == 0:
                    continue
                median_val = np.median(pixel_values)
                std_dev = np.std(pixel_values)

                lower_bound = median_val - lower * std_dev
                upper_bound = median_val + upper * std_dev

                valid = (pixel_values >= lower_bound) & (pixel_values <= upper_bound)
                valid_vals = pixel_values[valid]
                if valid_vals.size > 0:
                    clipped[i, j, c] = np.mean(valid_vals)
                else:
                    clipped[i, j, c] = median_val

    return clipped

def windsorized_sigma_clip(stack, lower=2.5, upper=2.5):
    """
    Dispatcher function that calls either the 3D or 4D specialized Numba function,
    depending on 'stack.ndim'.
    """
    if stack.ndim == 3:
        return windsorized_sigma_clip_3d(stack, lower, upper)
    elif stack.ndim == 4:
        return windsorized_sigma_clip_4d(stack, lower, upper)
    else:
        raise ValueError(f"windsorized_sigma_clip: stack must be 3D or 4D, got {stack.shape}")


@njit(parallel=True)
def subtract_dark_with_pedestal_3d(frames, dark_frame, pedestal):
    """
    For mono stack:
      frames.shape == (F,H,W)
      dark_frame.shape == (H,W)
    Adds 'pedestal' after subtracting dark_frame from each frame.
    Returns the same shape (F,H,W).
    """
    num_frames, height, width = frames.shape
    result = np.empty_like(frames, dtype=np.float32)

    # Validate dark_frame shape
    if dark_frame.ndim != 2 or dark_frame.shape != (height, width):
        raise ValueError(
            "subtract_dark_with_pedestal_3d: for 3D frames, dark_frame must be 2D (H,W)"
        )

    for i in prange(num_frames):
        for y in range(height):
            for x in range(width):
                result[i, y, x] = frames[i, y, x] - dark_frame[y, x] + pedestal

    return result

@njit(parallel=True)
def subtract_dark_with_pedestal_4d(frames, dark_frame, pedestal):
    """
    For color stack:
      frames.shape == (F,H,W,C)
      dark_frame.shape == (H,W,C)
    Adds 'pedestal' after subtracting dark_frame from each frame.
    Returns the same shape (F,H,W,C).
    """
    num_frames, height, width, channels = frames.shape
    result = np.empty_like(frames, dtype=np.float32)

    # Validate dark_frame shape
    if dark_frame.ndim != 3 or dark_frame.shape != (height, width, channels):
        raise ValueError(
            "subtract_dark_with_pedestal_4d: for 4D frames, dark_frame must be 3D (H,W,C)"
        )

    for i in prange(num_frames):
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    result[i, y, x, c] = frames[i, y, x, c] - dark_frame[y, x, c] + pedestal

    return result

def subtract_dark_with_pedestal(frames, dark_frame, pedestal):
    """
    Dispatcher function that calls either the 3D or 4D specialized Numba function
    depending on 'frames.ndim'.
    """
    if frames.ndim == 3:
        return subtract_dark_with_pedestal_3d(frames, dark_frame, pedestal)
    elif frames.ndim == 4:
        return subtract_dark_with_pedestal_4d(frames, dark_frame, pedestal)
    else:
        raise ValueError(
            f"subtract_dark_with_pedestal: frames must be 3D or 4D, got {frames.shape}"
        )


@njit(parallel=True, fastmath=True)
def parallel_measure_frames(images):
    """
    Parallel processing for measuring simple stats (mean only).
    'images' is a list (or array) of N images, each of which can be:
      - 2D (H,W) for a single mono image
      - 3D (H,W,C) for a single color image
      - Possibly 3D or 4D if you're storing multi-frame stacks in 'images'
    We just compute np.mean(...) of each image, no matter how many dims.
    """
    n = len(images)
    means = np.zeros(n, dtype=np.float32)

    for i in prange(n):
        arr = images[i]
        # arr could have shape (H,W) or (H,W,C) or (F,H,W) etc.
        # np.mean works for any dimension, so no special logic needed.
        means[i] = np.float32(np.mean(arr))

    return means


@njit(fastmath=True)
def fast_mad(image):
    """ Computes the Median Absolute Deviation (MAD) as a robust noise estimator. """
    flat_image = image.ravel()  # ✅ Flatten the 2D array into 1D
    median_val = np.median(flat_image)  # Compute median
    mad = np.median(np.abs(flat_image - median_val))  # Compute MAD
    return mad * 1.4826  # ✅ Scale MAD to match standard deviation (for Gaussian noise)



@njit(fastmath=True)
def compute_snr(image):
    """ Computes the Signal-to-Noise Ratio (SNR) using fast Numba std. """
    mean_signal = np.mean(image)
    noise = compute_noise(image)
    return mean_signal / noise if noise > 0 else 0




@njit(fastmath=True)
def compute_noise(image):
    """ Estimates noise using Median Absolute Deviation (MAD). """
    return fast_mad(image)




def compute_star_count(image):
    """ Uses fast star detection instead of DAOStarFinder. """
    return fast_star_count(image)


def fast_star_count(image, blur_size=5, threshold_factor=2.5):
    """ Fast star detection using local contrast and Otsu thresholding. """
    
    # ✅ Step 1: Convert to 8-bit (scale intensity to 0-255)
    norm_img = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
    norm_img = norm_img.astype(np.uint8)

    # ✅ Step 2: Apply Gaussian Blur to create background model
    blurred = cv2.GaussianBlur(norm_img, (blur_size, blur_size), 0)

    # ✅ Step 3: Subtract to enhance stars
    enhanced = cv2.absdiff(norm_img, blurred)

    # ✅ Step 4: Otsu’s thresholding for adaptive star detection
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ✅ Step 5: Count connected components (stars)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    return num_labels - 1  # Subtract 1 because the first label is the background

@njit(parallel=True, fastmath=True)
def normalize_images_3d(stack, ref_median):
    """
    Normalizes each frame in a 3D mono stack (F,H,W)
    so that its median equals ref_median.

    Returns a 3D result (F,H,W).
    """
    num_frames, height, width = stack.shape
    normalized_stack = np.zeros_like(stack, dtype=np.float32)

    for i in prange(num_frames):
        # shape of one frame: (H,W)
        img = stack[i]
        img_median = np.median(img)

        # Prevent division by zero
        scale_factor = ref_median / max(img_median, 1e-6)
        # Scale the entire 2D frame
        normalized_stack[i] = img * scale_factor

    return normalized_stack

@njit(parallel=True, fastmath=True)
def normalize_images_4d(stack, ref_median):
    """
    Normalizes each frame in a 4D color stack (F,H,W,C)
    so that its median equals ref_median.

    Returns a 4D result (F,H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    normalized_stack = np.zeros_like(stack, dtype=np.float32)

    for i in prange(num_frames):
        # shape of one frame: (H,W,C)
        img = stack[i]  # (H,W,C)
        # Flatten to 1D to compute median across all channels/pixels
        img_median = np.median(img.ravel())

        # Prevent division by zero
        scale_factor = ref_median / max(img_median, 1e-6)

        # Scale the entire 3D frame
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    normalized_stack[i, y, x, c] = img[y, x, c] * scale_factor

    return normalized_stack

def normalize_images(stack, ref_median):
    """
    Dispatcher that calls either the 3D or 4D specialized Numba function
    depending on 'stack.ndim'.

    - If stack.ndim == 3, we assume shape (F,H,W).
    - If stack.ndim == 4, we assume shape (F,H,W,C).
    """
    if stack.ndim == 3:
        return normalize_images_3d(stack, ref_median)
    elif stack.ndim == 4:
        return normalize_images_4d(stack, ref_median)
    else:
        raise ValueError(f"normalize_images: stack must be 3D or 4D, got shape {stack.shape}")


@njit(fastmath=True, parallel=True)
def debayer_RGGB_fast(image):
    h, w = image.shape
    new_h = h // 2
    new_w = w // 2
    out = np.empty((new_h, new_w, 3), dtype=image.dtype)
    for i in prange(new_h):
        for j in range(new_w):
            i2 = i * 2
            j2 = j * 2
            # RGGB: top-left red, top-right green, bottom-left green, bottom-right blue.
            r = image[i2, j2]
            g1 = image[i2, j2 + 1]
            g2 = image[i2 + 1, j2]
            b = image[i2 + 1, j2 + 1]
            out[i, j, 0] = r
            out[i, j, 1] = (g1 + g2) / 2.0
            out[i, j, 2] = b
    return out

@njit(fastmath=True, parallel=True)
def debayer_BGGR_fast(image):
    h, w = image.shape
    new_h = h // 2
    new_w = w // 2
    out = np.empty((new_h, new_w, 3), dtype=image.dtype)
    for i in prange(new_h):
        for j in range(new_w):
            i2 = i * 2
            j2 = j * 2
            # BGGR: top-left blue, top-right green, bottom-left green, bottom-right red.
            b = image[i2, j2]
            g1 = image[i2, j2 + 1]
            g2 = image[i2 + 1, j2]
            r = image[i2 + 1, j2 + 1]
            out[i, j, 2] = b
            out[i, j, 1] = (g1 + g2) / 2.0
            out[i, j, 0] = r
    return out

@njit(fastmath=True, parallel=True)
def debayer_GRBG_fast(image):
    h, w = image.shape
    new_h = h // 2
    new_w = w // 2
    out = np.empty((new_h, new_w, 3), dtype=image.dtype)
    for i in prange(new_h):
        for j in range(new_w):
            i2 = i * 2
            j2 = j * 2
            # GRBG: top-left green, top-right red, bottom-left blue, bottom-right green.
            g1 = image[i2, j2]
            r = image[i2, j2 + 1]
            b = image[i2 + 1, j2]
            g2 = image[i2 + 1, j2 + 1]
            out[i, j, 0] = r
            out[i, j, 1] = (g1 + g2) / 2.0
            out[i, j, 2] = b
    return out

@njit(fastmath=True, parallel=True)
def debayer_GBRG_fast(image):
    h, w = image.shape
    new_h = h // 2
    new_w = w // 2
    out = np.empty((new_h, new_w, 3), dtype=image.dtype)
    for i in prange(new_h):
        for j in range(new_w):
            i2 = i * 2
            j2 = j * 2
            # GBRG: top-left green, top-right blue, bottom-left red, bottom-right green.
            g1 = image[i2, j2]
            b = image[i2, j2 + 1]
            r = image[i2 + 1, j2]
            g2 = image[i2 + 1, j2 + 1]
            out[i, j, 0] = r
            out[i, j, 1] = (g1 + g2) / 2.0
            out[i, j, 2] = b
    return out

def debayer_fits_fast(image_data, bayer_pattern):
    bp = bayer_pattern.upper()
    if bp == 'RGGB':
        return debayer_RGGB_fast(image_data)
    elif bp == 'BGGR':
        return debayer_BGGR_fast(image_data)
    elif bp == 'GRBG':
        return debayer_GRBG_fast(image_data)
    elif bp == 'GBRG':
        return debayer_GBRG_fast(image_data)
    else:
        raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")

def debayer_raw_fast(raw_image_data, bayer_pattern="RGGB"):
    # For RAW images, use the same debayering logic.
    return debayer_fits_fast(raw_image_data, bayer_pattern)


@njit(parallel=True, fastmath=True)
def applyPixelMath_numba(image_array, amount):
    factor = 3 ** amount
    denom_factor = 3 ** amount - 1
    height, width, channels = image_array.shape
    output = np.empty_like(image_array, dtype=np.float32)

    for y in prange(height):
        for x in prange(width):
            for c in prange(channels):
                val = (factor * image_array[y, x, c]) / (denom_factor * image_array[y, x, c] + 1)
                output[y, x, c] = min(max(val, 0.0), 1.0)  # Equivalent to np.clip()
    
    return output

@njit(parallel=True, fastmath=True)
def adjust_saturation_numba(image_array, saturation_factor):
    height, width, channels = image_array.shape
    output = np.empty_like(image_array, dtype=np.float32)

    for y in prange(int(height)):  # Ensure y is an integer
        for x in prange(int(width)):  # Ensure x is an integer
            r, g, b = image_array[int(y), int(x)]  # Force integer indexing

            # Convert RGB to HSV manually
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            delta = max_val - min_val

            # Compute Hue (H)
            if delta == 0:
                h = 0
            elif max_val == r:
                h = (60 * ((g - b) / delta) + 360) % 360
            elif max_val == g:
                h = (60 * ((b - r) / delta) + 120) % 360
            else:
                h = (60 * ((r - g) / delta) + 240) % 360

            # Compute Saturation (S)
            s = (delta / max_val) if max_val != 0 else 0
            s *= saturation_factor  # Apply saturation adjustment
            s = min(max(s, 0.0), 1.0)  # Clip saturation

            # Convert back to RGB
            if s == 0:
                r, g, b = max_val, max_val, max_val
            else:
                c = s * max_val
                x_val = c * (1 - abs((h / 60) % 2 - 1))
                m = max_val - c

                if 0 <= h < 60:
                    r, g, b = c, x_val, 0
                elif 60 <= h < 120:
                    r, g, b = x_val, c, 0
                elif 120 <= h < 180:
                    r, g, b = 0, c, x_val
                elif 180 <= h < 240:
                    r, g, b = 0, x_val, c
                elif 240 <= h < 300:
                    r, g, b = x_val, 0, c
                else:
                    r, g, b = c, 0, x_val

                r, g, b = r + m, g + m, b + m  # Add m to shift brightness

            # ✅ Fix: Explicitly cast indices to integers
            output[int(y), int(x), 0] = r
            output[int(y), int(x), 1] = g
            output[int(y), int(x), 2] = b

    return output




@njit(parallel=True, fastmath=True)
def applySCNR_numba(image_array):
    height, width, _ = image_array.shape
    output = np.empty_like(image_array, dtype=np.float32)

    for y in prange(int(height)):
        for x in prange(int(width)):
            r, g, b = image_array[y, x]
            g = min(g, (r + b) / 2)  # Reduce green to the average of red & blue
            
            # ✅ Fix: Assign channels individually instead of a tuple
            output[int(y), int(x), 0] = r
            output[int(y), int(x), 1] = g
            output[int(y), int(x), 2] = b


    return output

# D65 reference
_Xn, _Yn, _Zn = 0.95047, 1.00000, 1.08883

# Matrix for RGB -> XYZ (sRGB => D65)
_M_rgb2xyz = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float32)

# Matrix for XYZ -> RGB (sRGB => D65)
_M_xyz2rgb = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
], dtype=np.float32)



@njit(parallel=True, fastmath=True)
def apply_lut_gray(image_in, lut):
    """
    Numba-accelerated application of 'lut' to a single-channel image_in in [0..1].
    'lut' is a 1D array of shape (size,) also in [0..1].
    """
    out = np.empty_like(image_in)
    height, width = image_in.shape
    size_lut = len(lut) - 1

    for y in prange(height):
        for x in range(width):
            v = image_in[y, x]
            idx = int(v * size_lut + 0.5)
            if idx < 0: idx = 0
            elif idx > size_lut: idx = size_lut
            out[y, x] = lut[idx]

    return out

@njit(parallel=True, fastmath=True)
def apply_lut_color(image_in, lut):
    """
    Numba-accelerated application of 'lut' to a 3-channel image_in in [0..1].
    'lut' is a 1D array of shape (size,) also in [0..1].
    """
    out = np.empty_like(image_in)
    height, width, channels = image_in.shape
    size_lut = len(lut) - 1

    for y in prange(height):
        for x in range(width):
            for c in range(channels):
                v = image_in[y, x, c]
                idx = int(v * size_lut + 0.5)
                if idx < 0: idx = 0
                elif idx > size_lut: idx = size_lut
                out[y, x, c] = lut[idx]

    return out

@njit(parallel=True, fastmath=True)
def apply_lut_mono_inplace(array2d, lut):
    """
    In-place LUT application on a single-channel 2D array in [0..1].
    'lut' has shape (size,) also in [0..1].
    """
    H, W = array2d.shape
    size_lut = len(lut) - 1
    for y in prange(H):
        for x in prange(W):
            v = array2d[y, x]
            idx = int(v * size_lut + 0.5)
            if idx < 0:
                idx = 0
            elif idx > size_lut:
                idx = size_lut
            array2d[y, x] = lut[idx]

@njit(parallel=True, fastmath=True)
def apply_lut_color_inplace(array3d, lut):
    """
    In-place LUT application on a 3-channel array in [0..1].
    'lut' has shape (size,) also in [0..1].
    """
    H, W, C = array3d.shape
    size_lut = len(lut) - 1
    for y in prange(H):
        for x in prange(W):
            for c in range(C):
                v = array3d[y, x, c]
                idx = int(v * size_lut + 0.5)
                if idx < 0:
                    idx = 0
                elif idx > size_lut:
                    idx = size_lut
                array3d[y, x, c] = lut[idx]

@njit(parallel=True, fastmath=True)
def rgb_to_xyz_numba(rgb):
    """
    Convert an image from sRGB to XYZ (D65).
    rgb: float32 array in [0..1], shape (H,W,3)
    returns xyz in [0..maybe >1], shape (H,W,3)
    """
    H, W, _ = rgb.shape
    out = np.empty((H, W, 3), dtype=np.float32)
    for y in prange(H):
        for x in prange(W):
            r = rgb[y, x, 0]
            g = rgb[y, x, 1]
            b = rgb[y, x, 2]
            # Multiply by M_rgb2xyz
            X = _M_rgb2xyz[0,0]*r + _M_rgb2xyz[0,1]*g + _M_rgb2xyz[0,2]*b
            Y = _M_rgb2xyz[1,0]*r + _M_rgb2xyz[1,1]*g + _M_rgb2xyz[1,2]*b
            Z = _M_rgb2xyz[2,0]*r + _M_rgb2xyz[2,1]*g + _M_rgb2xyz[2,2]*b
            out[y, x, 0] = X
            out[y, x, 1] = Y
            out[y, x, 2] = Z
    return out

@njit(parallel=True, fastmath=True)
def xyz_to_rgb_numba(xyz):
    """
    Convert an image from XYZ (D65) to sRGB.
    xyz: float32 array, shape (H,W,3)
    returns rgb in [0..1], shape (H,W,3)
    """
    H, W, _ = xyz.shape
    out = np.empty((H, W, 3), dtype=np.float32)
    for y in prange(H):
        for x in prange(W):
            X = xyz[y, x, 0]
            Y = xyz[y, x, 1]
            Z = xyz[y, x, 2]
            # Multiply by M_xyz2rgb
            r = _M_xyz2rgb[0,0]*X + _M_xyz2rgb[0,1]*Y + _M_xyz2rgb[0,2]*Z
            g = _M_xyz2rgb[1,0]*X + _M_xyz2rgb[1,1]*Y + _M_xyz2rgb[1,2]*Z
            b = _M_xyz2rgb[2,0]*X + _M_xyz2rgb[2,1]*Y + _M_xyz2rgb[2,2]*Z
            # Clip to [0..1]
            if r < 0: r = 0
            elif r > 1: r = 1
            if g < 0: g = 0
            elif g > 1: g = 1
            if b < 0: b = 0
            elif b > 1: b = 1
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b
    return out

@njit
def f_lab_numba(t):
    delta = 6/29
    out = np.empty_like(t, dtype=np.float32)
    for i in range(t.size):
        val = t.flat[i]
        if val > delta**3:
            out.flat[i] = val**(1/3)
        else:
            out.flat[i] = val/(3*delta*delta) + (4/29)
    return out

@njit(parallel=True, fastmath=True)
def xyz_to_lab_numba(xyz):
    """
    xyz => shape(H,W,3), in D65. 
    returns lab in shape(H,W,3): L in [0..100], a,b in ~[-128..127].
    """
    H, W, _ = xyz.shape
    out = np.empty((H,W,3), dtype=np.float32)
    for y in prange(H):
        for x in prange(W):
            X = xyz[y, x, 0] / _Xn
            Y = xyz[y, x, 1] / _Yn
            Z = xyz[y, x, 2] / _Zn
            fx = (X)**(1/3) if X > (6/29)**3 else X/(3*(6/29)**2) + 4/29
            fy = (Y)**(1/3) if Y > (6/29)**3 else Y/(3*(6/29)**2) + 4/29
            fz = (Z)**(1/3) if Z > (6/29)**3 else Z/(3*(6/29)**2) + 4/29
            L = 116*fy - 16
            a = 500*(fx - fy)
            b = 200*(fy - fz)
            out[y, x, 0] = L
            out[y, x, 1] = a
            out[y, x, 2] = b
    return out

@njit(parallel=True, fastmath=True)
def lab_to_xyz_numba(lab):
    """
    lab => shape(H,W,3): L in [0..100], a,b in ~[-128..127].
    returns xyz shape(H,W,3).
    """
    H, W, _ = lab.shape
    out = np.empty((H,W,3), dtype=np.float32)
    delta = 6/29
    for y in prange(H):
        for x in prange(W):
            L = lab[y, x, 0]
            a = lab[y, x, 1]
            b = lab[y, x, 2]
            fy = (L+16)/116
            fx = fy + a/500
            fz = fy - b/200

            if fx > delta:
                xr = fx**3
            else:
                xr = 3*delta*delta*(fx - 4/29)
            if fy > delta:
                yr = fy**3
            else:
                yr = 3*delta*delta*(fy - 4/29)
            if fz > delta:
                zr = fz**3
            else:
                zr = 3*delta*delta*(fz - 4/29)

            X = _Xn * xr
            Y = _Yn * yr
            Z = _Zn * zr
            out[y, x, 0] = X
            out[y, x, 1] = Y
            out[y, x, 2] = Z
    return out

@njit(parallel=True, fastmath=True)
def rgb_to_hsv_numba(rgb):
    H, W, _ = rgb.shape
    out = np.empty((H,W,3), dtype=np.float32)
    for y in prange(H):
        for x in prange(W):
            r = rgb[y,x,0]
            g = rgb[y,x,1]
            b = rgb[y,x,2]
            cmax = max(r,g,b)
            cmin = min(r,g,b)
            delta = cmax - cmin
            # Hue
            h = 0.0
            if delta != 0.0:
                if cmax == r:
                    h = 60*(((g-b)/delta) % 6)
                elif cmax == g:
                    h = 60*(((b-r)/delta) + 2)
                else:
                    h = 60*(((r-g)/delta) + 4)
            # Saturation
            s = 0.0
            if cmax > 0.0:
                s = delta / cmax
            v = cmax
            out[y,x,0] = h
            out[y,x,1] = s
            out[y,x,2] = v
    return out

@njit(parallel=True, fastmath=True)
def hsv_to_rgb_numba(hsv):
    H, W, _ = hsv.shape
    out = np.empty((H,W,3), dtype=np.float32)
    for y in prange(H):
        for x in prange(W):
            h = hsv[y,x,0]
            s = hsv[y,x,1]
            v = hsv[y,x,2]
            c = v*s
            hh = (h/60.0) % 6
            x_ = c*(1 - abs(hh % 2 - 1))
            m = v - c
            r = 0.0
            g = 0.0
            b = 0.0
            if 0 <= hh < 1:
                r,g,b = c,x_,0
            elif 1 <= hh < 2:
                r,g,b = x_,c,0
            elif 2 <= hh < 3:
                r,g,b = 0,c,x_
            elif 3 <= hh < 4:
                r,g,b = 0,x_,c
            elif 4 <= hh < 5:
                r,g,b = x_,0,c
            else:
                r,g,b = c,0,x_
            out[y,x,0] = (r + m)
            out[y,x,1] = (g + m)
            out[y,x,2] = (b + m)
    return out

@njit(parallel=True, fastmath=True)
def _cosmetic_correction_numba_fixed(corrected, H, W, C, hot_sigma, cold_sigma):
    """
    Optimized Numba-compiled local outlier correction.
    - Computes median and standard deviation from 8 surrounding pixels (excluding center).
    - If the center pixel is greater than (median + hot_sigma * std_dev), it is replaced with the median.
    - If the center pixel is less than (median - cold_sigma * std_dev), it is replaced with the median.
    - Edge pixels are skipped (avoiding padding artifacts).
    """
    local_vals = np.empty(9, dtype=np.float32)  # Holds 8 surrounding pixels

    # Process pixels in parallel, skipping edges
    for y in prange(1, H - 1):  # Skip first and last rows
        for x in range(1, W - 1):  # Skip first and last columns
            # If the image is grayscale, set C=1 and handle accordingly
            for c_i in prange(C if corrected.ndim == 3 else 1):
                k = 0
                for dy in range(-1, 2):  # -1, 0, +1
                    for dx in range(-1, 2):  # -1, 0, +1
                        if corrected.ndim == 3:  # Color image
                            local_vals[k] = corrected[y + dy, x + dx, c_i]
                        else:  # Grayscale image
                            local_vals[k] = corrected[y + dy, x + dx]
                        k += 1

                # Compute median
                M = np.median(local_vals)

                # Compute MAD manually
                abs_devs = np.abs(local_vals - M)
                MAD = np.median(abs_devs)

                # Convert MAD to an approximation of standard deviation
                sigma_mad = 1.4826 * MAD  

                # Get center pixel
                if corrected.ndim == 3:
                    T = corrected[y, x, c_i]
                else:
                    T = corrected[y, x]

                threshold_high = M + (hot_sigma * sigma_mad)
                threshold_low = M - (cold_sigma * sigma_mad)

                # **Apply correction ONLY if center pixel is an outlier**
                if T > threshold_high or T < threshold_low:
                    if corrected.ndim == 3:
                        corrected[y, x, c_i] = M  # Replace center pixel in color image
                    else:
                        corrected[y, x] = M  # Replace center pixel in grayscale image


def bulk_cosmetic_correction_numba(image, hot_sigma=3.0, cold_sigma=3.0, window_size=3):
    """
    Optimized local outlier correction using Numba.
    - Identifies hot and cold outliers based on local neighborhood statistics.
    - Uses median and standard deviation from surrounding pixels to detect and replace outliers.
    - Applies separate hot_sigma and cold_sigma thresholds.
    - Skips edge pixels to avoid padding artifacts.
    """

    was_gray = False

    if image.ndim == 2:  # Convert grayscale to 3D
        H, W = image.shape
        C = 1
        was_gray = True
        image = image[:, :, np.newaxis]  # Explicitly add a color channel dimension

    else:
        H, W, C = image.shape

    # Copy the image for modification
    corrected = image.astype(np.float32).copy()

    # Apply fast correction (no padding, edges skipped)
    _cosmetic_correction_numba_fixed(corrected, H, W, C, hot_sigma, cold_sigma)

    if was_gray:
        corrected = corrected[:, :, 0]  # Convert back to 2D if originally grayscale

    return corrected

def evaluate_polynomial(H: int, W: int, coeffs: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluates the polynomial function over the entire image domain.
    """
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    A_full = build_poly_terms(xx.ravel(), yy.ravel(), degree)
    return (A_full @ coeffs).reshape(H, W)



@njit(parallel=True, fastmath=True)
def numba_mono_final_formula(rescaled, median_rescaled, target_median):
    """
    Applies the final formula *after* we already have the rescaled values.
    
    rescaled[y,x] = (original[y,x] - black_point) / (1 - black_point)
    median_rescaled = median(rescaled)
    
    out_val = ((median_rescaled - 1) * target_median * r) /
              ( median_rescaled*(target_median + r -1) - target_median*r )
    """
    H, W = rescaled.shape
    out = np.empty_like(rescaled)

    for y in prange(H):
        for x in range(W):
            r = rescaled[y, x]
            numer = (median_rescaled - 1.0) * target_median * r
            denom = median_rescaled * (target_median + r - 1.0) - target_median * r
            if np.abs(denom) < 1e-12:
                denom = 1e-12
            out[y, x] = numer / denom

    return out

@njit(parallel=True, fastmath=True)
def numba_color_final_formula_linked(rescaled, median_rescaled, target_median):
    """
    Linked color transform: we use one median_rescaled for all channels.
    rescaled: (H,W,3), already = (image - black_point)/(1 - black_point)
    median_rescaled = median of *all* pixels in rescaled
    """
    H, W, C = rescaled.shape
    out = np.empty_like(rescaled)

    for y in prange(H):
        for x in range(W):
            for c in range(C):
                r = rescaled[y, x, c]
                numer = (median_rescaled - 1.0) * target_median * r
                denom = median_rescaled * (target_median + r - 1.0) - target_median * r
                if np.abs(denom) < 1e-12:
                    denom = 1e-12
                out[y, x, c] = numer / denom

    return out

@njit(parallel=True, fastmath=True)
def numba_color_final_formula_unlinked(rescaled, medians_rescaled, target_median):
    """
    Unlinked color transform: a separate median_rescaled per channel.
    rescaled: (H,W,3), where each channel is already (val - black_point[c]) / (1 - black_point[c])
    medians_rescaled: shape (3,) with median of each channel in the rescaled array.
    """
    H, W, C = rescaled.shape
    out = np.empty_like(rescaled)

    for y in prange(H):
        for x in range(W):
            for c in range(C):
                r = rescaled[y, x, c]
                med = medians_rescaled[c]
                numer = (med - 1.0) * target_median * r
                denom = med * (target_median + r - 1.0) - target_median * r
                if np.abs(denom) < 1e-12:
                    denom = 1e-12
                out[y, x, c] = numer / denom

    return out


def build_poly_terms(x_array: np.ndarray, y_array: np.ndarray, degree: int) -> np.ndarray:
    """
    Precomputes polynomial basis terms efficiently using NumPy, supporting up to degree 6.
    """
    ones = np.ones_like(x_array, dtype=np.float32)

    if degree == 1:
        return np.column_stack((ones, x_array, y_array))

    elif degree == 2:
        return np.column_stack((ones, x_array, y_array, 
                                x_array**2, x_array * y_array, y_array**2))

    elif degree == 3:
        return np.column_stack((ones, x_array, y_array, 
                                x_array**2, x_array * y_array, y_array**2, 
                                x_array**3, x_array**2 * y_array, x_array * y_array**2, y_array**3))

    elif degree == 4:
        return np.column_stack((ones, x_array, y_array, 
                                x_array**2, x_array * y_array, y_array**2, 
                                x_array**3, x_array**2 * y_array, x_array * y_array**2, y_array**3,
                                x_array**4, x_array**3 * y_array, x_array**2 * y_array**2, x_array * y_array**3, y_array**4))

    elif degree == 5:
        return np.column_stack((ones, x_array, y_array, 
                                x_array**2, x_array * y_array, y_array**2, 
                                x_array**3, x_array**2 * y_array, x_array * y_array**2, y_array**3,
                                x_array**4, x_array**3 * y_array, x_array**2 * y_array**2, x_array * y_array**3, y_array**4,
                                x_array**5, x_array**4 * y_array, x_array**3 * y_array**2, x_array**2 * y_array**3, x_array * y_array**4, y_array**5))

    elif degree == 6:
        return np.column_stack((ones, x_array, y_array, 
                                x_array**2, x_array * y_array, y_array**2, 
                                x_array**3, x_array**2 * y_array, x_array * y_array**2, y_array**3,
                                x_array**4, x_array**3 * y_array, x_array**2 * y_array**2, x_array * y_array**3, y_array**4,
                                x_array**5, x_array**4 * y_array, x_array**3 * y_array**2, x_array**2 * y_array**3, x_array * y_array**4, y_array**5,
                                x_array**6, x_array**5 * y_array, x_array**4 * y_array**2, x_array**3 * y_array**3, x_array**2 * y_array**4, x_array * y_array**5, y_array**6))

    else:
        raise ValueError(f"Unsupported polynomial degree={degree}. Max supported is 6.")




def generate_sample_points(image: np.ndarray, num_points: int = 100) -> np.ndarray:
    """
    Generates sample points uniformly across the image.

    - Places points in a uniform grid (no randomization).
    - Avoids border pixels.
    - Skips any points with value 0.000 or above 0.85.

    Returns:
        np.ndarray: Array of shape (N, 2) containing (x, y) coordinates of sample points.
    """
    H, W = image.shape[:2]
    points = []

    # Create a uniform grid (avoiding the border)
    grid_size = int(np.sqrt(num_points))  # Roughly equal spacing
    x_vals = np.linspace(10, W - 10, grid_size, dtype=int)  # Avoids border
    y_vals = np.linspace(10, H - 10, grid_size, dtype=int)

    for y in y_vals:
        for x in x_vals:
            # Skip values that are too dark (0.000) or too bright (> 0.85)
            if np.any(image[int(y), int(x)] == 0.000) or np.any(image[int(y), int(x)] > 0.85):
                continue  # Skip this pixel

            points.append((int(x), int(y)))

            if len(points) >= num_points:
                return np.array(points, dtype=np.int32)  # Return only valid points

    return np.array(points, dtype=np.int32)  # Return all collected points

@njit(parallel=True, fastmath=True)
def numba_unstretch(image: np.ndarray, stretch_original_medians: np.ndarray, stretch_original_mins: np.ndarray) -> np.ndarray:
    """
    Numba-optimized function to undo the unlinked stretch.
    Restores each channel separately.
    """
    H, W, C = image.shape
    out = np.empty_like(image, dtype=np.float32)

    for c in prange(C):  # Parallelize per channel
        cmed_stretched = np.median(image[..., c])
        orig_med = stretch_original_medians[c]
        orig_min = stretch_original_mins[c]

        if cmed_stretched != 0 and orig_med != 0:
            for y in prange(H):
                for x in range(W):
                    r = image[y, x, c]
                    numerator = (cmed_stretched - 1) * orig_med * r
                    denominator = cmed_stretched * (orig_med + r - 1) - orig_med * r
                    if denominator == 0:
                        denominator = 1e-6  # Avoid division by zero
                    out[y, x, c] = numerator / denominator

            # Restore the original black point
            out[..., c] += orig_min

    return np.clip(out, 0, 1)  # Clip to valid range


@njit(fastmath=True)
def drizzle_deposit_numba_naive(img_data, transform, drizzle_buffer, coverage_buffer,
                                drizzle_factor, frame_weight):
    """
    Naive deposit: each input pixel is mapped to exactly one output pixel,
    ignoring drop_shrink. 2D single-channel version for brevity.
    """
    h, w = img_data.shape
    out_h, out_w = drizzle_buffer.shape

    a, b, tx = transform[0]
    c, d, ty = transform[1]

    for y in range(h):
        for x in range(w):
            val = img_data[y, x]
            if val == 0:
                continue
            X = a*x + b*y + tx
            Y = c*x + d*y + ty

            # multiply by drizzle_factor
            Xo = int(X * drizzle_factor)
            Yo = int(Y * drizzle_factor)

            if 0 <= Xo < out_w and 0 <= Yo < out_h:
                drizzle_buffer[Yo, Xo] += val * frame_weight
                coverage_buffer[Yo, Xo] += frame_weight

    return drizzle_buffer, coverage_buffer


@njit(fastmath=True)
def drizzle_deposit_numba_footprint(img_data, transform, drizzle_buffer, coverage_buffer,
                                    drizzle_factor, drop_shrink, frame_weight):
    """
    Footprint deposit: each input pixel is distributed over a bounding box
    of width = drop_shrink. 2D single-channel version for brevity.
    """
    h, w = img_data.shape
    out_h, out_w = drizzle_buffer.shape

    a, b, tx = transform[0]
    c, d, ty = transform[1]

    footprint_radius = drop_shrink * 0.5

    for y in range(h):
        for x in range(w):
            val = img_data[y, x]
            if val == 0:
                continue

            X = a*x + b*y + tx
            Y = c*x + d*y + ty
            Xo = X * drizzle_factor
            Yo = Y * drizzle_factor

            min_x = int(np.floor(Xo - footprint_radius))
            max_x = int(np.floor(Xo + footprint_radius))
            min_y = int(np.floor(Yo - footprint_radius))
            max_y = int(np.floor(Yo + footprint_radius))

            if max_x < 0 or min_x >= out_w or max_y < 0 or min_y >= out_h:
                continue
            min_x = max(min_x, 0)
            max_x = min(max_x, out_w - 1)
            min_y = max(min_y, 0)
            max_y = min(max_y, out_h - 1)

            width_foot = max_x - min_x + 1
            height_foot = max_y - min_y + 1
            area_pixels = width_foot * height_foot
            if area_pixels <= 0:
                continue

            deposit_val = (val * frame_weight) / area_pixels
            for oy in range(min_y, max_y+1):
                for ox in range(min_x, max_x+1):
                    drizzle_buffer[oy, ox] += deposit_val
                    coverage_buffer[oy, ox] += frame_weight

    return drizzle_buffer, coverage_buffer

@njit(parallel=True)
def finalize_drizzle_2d(drizzle_buffer, coverage_buffer, final_out):
    """
    parallel-friendly final step: final_out = drizzle_buffer / coverage_buffer,
    with coverage < 1e-8 => 0
    """
    out_h, out_w = drizzle_buffer.shape
    for y in prange(out_h):
        for x in range(out_w):
            cov = coverage_buffer[y, x]
            if cov < 1e-8:
                final_out[y, x] = 0.0
            else:
                final_out[y, x] = drizzle_buffer[y, x] / cov
    return final_out

@njit(fastmath=True)
def drizzle_deposit_color_naive(
    img_data,          # shape (H,W,C)
    transform,         # shape (2,3)
    drizzle_buffer,    # shape (outH,outW,C)
    coverage_buffer,   # shape (outH,outW,C)
    drizzle_factor,
    drop_shrink,       # not used here, but included for signature consistency
    frame_weight
):
    """
    Naive color deposit:
    Each input pixel (for each channel) is mapped to exactly one output pixel in (outH,outW,C).
    We ignore drop_shrink and place all flux into a single pixel.

    Parameters
    ----------
    img_data : np.ndarray, shape (H,W,C)
        The color input image data for one frame.
    transform : np.ndarray, shape (2,3)
        The affine matrix [ [a,b,tx], [c,d,ty] ] from registration.
    drizzle_buffer : np.ndarray, shape (outH,outW,C)
        The upsampled output array where we accumulate flux.
    coverage_buffer : np.ndarray, shape (outH,outW,C)
        Parallel buffer tracking coverage for each channel.
    drizzle_factor : float
        E.g. 2.0 for 2× upsampling.
    drop_shrink : float
        Not used in this naive approach, but included for function signature compatibility.
    frame_weight : float
        A per-frame weight (e.g. from star-count weighting).

    Returns
    -------
    drizzle_buffer, coverage_buffer : updated arrays with the newly deposited flux.
    """

    H, W, channels = img_data.shape
    outH, outW, outC = drizzle_buffer.shape

    # Unpack affine transform
    a, b, tx = transform[0]
    c_, d, ty = transform[1]

    for y in range(H):
        for x in range(W):
            # 1) Compute transformed coordinates
            X = a*x + b*y + tx
            Y = c_*x + d*y + ty

            # 2) Upsample
            Xo = int(X * drizzle_factor)
            Yo = int(Y * drizzle_factor)

            # 3) Bounds check
            if 0 <= Xo < outW and 0 <= Yo < outH:
                # 4) Loop over channels
                for cidx in range(channels):
                    val = img_data[y, x, cidx]
                    if val != 0:
                        drizzle_buffer[Yo, Xo, cidx] += val * frame_weight
                        coverage_buffer[Yo, Xo, cidx] += frame_weight

    return drizzle_buffer, coverage_buffer

@njit(fastmath=True)
def drizzle_deposit_color_footprint(
    img_data,          # shape (H,W,C)
    transform,         # shape (2,3)
    drizzle_buffer,    # shape (outH,outW,C)
    coverage_buffer,   # shape (outH,outW,C) or (outH,outW) if you prefer
    drizzle_factor, 
    drop_shrink,
    frame_weight
):
    """
    Distributes each input pixel (for each channel) over a bounding-box footprint
    of width=drop_shrink in the output plane. 
    'img_data' is (H,W,C).
    'drizzle_buffer' and 'coverage_buffer' are (outH,outW,C).
    """

    H, W, channels = img_data.shape
    outH, outW, outC = drizzle_buffer.shape

    # Unpack affine transform
    a, b, tx = transform[0]
    c_, d, ty = transform[1]

    # The half-width of the footprint in output coords
    footprint_radius = drop_shrink * 0.5

    for y in range(H):
        for x in range(W):
            # We'll handle each channel separately
            # so we compute the transform once
            X = a*x + b*y + tx
            Y = c_*x + d*y + ty

            # Multiply by drizzle_factor => upsampled coords
            Xo = X * drizzle_factor
            Yo = Y * drizzle_factor

            # bounding box in output coords
            min_x = int(np.floor(Xo - footprint_radius))
            max_x = int(np.floor(Xo + footprint_radius))
            min_y = int(np.floor(Yo - footprint_radius))
            max_y = int(np.floor(Yo + footprint_radius))

            # Clip to output
            if max_x < 0 or min_x >= outW or max_y < 0 or min_y >= outH:
                continue
            if min_x < 0:
                min_x = 0
            if max_x >= outW:
                max_x = outW - 1
            if min_y < 0:
                min_y = 0
            if max_y >= outH:
                max_y = outH - 1

            width_foot = (max_x - min_x + 1)
            height_foot = (max_y - min_y + 1)
            area_pixels = width_foot * height_foot
            if area_pixels <= 0:
                continue

            for cidx in range(channels):
                val = img_data[y, x, cidx]
                if val == 0:
                    continue
                deposit_val = (val * frame_weight) / area_pixels

                # deposit in bounding box
                for oy in range(min_y, max_y+1):
                    for ox in range(min_x, max_x+1):
                        drizzle_buffer[oy, ox, cidx] += deposit_val
                        coverage_buffer[oy, ox, cidx] += frame_weight

    return drizzle_buffer, coverage_buffer

@njit
def finalize_drizzle_3d(drizzle_buffer, coverage_buffer, final_out):
    """
    final_out[y,x,c] = drizzle_buffer[y,x,c] / coverage_buffer[y,x,c]
    if coverage < 1e-8 => 0
    """
    outH, outW, channels = drizzle_buffer.shape
    for y in range(outH):
        for x in range(outW):
            for cidx in range(channels):
                cov = coverage_buffer[y, x, cidx]
                if cov < 1e-8:
                    final_out[y, x, cidx] = 0.0
                else:
                    final_out[y, x, cidx] = drizzle_buffer[y, x, cidx] / cov
    return final_out



@njit
def piecewise_linear(val, xvals, yvals):
    """
    Performs piecewise linear interpolation:
    Given a scalar 'val', and arrays xvals, yvals (each of length N),
    finds i s.t. xvals[i] <= val < xvals[i+1],
    then returns the linear interpolation between yvals[i], yvals[i+1].
    If val < xvals[0], returns yvals[0].
    If val > xvals[-1], returns yvals[-1].
    """
    if val <= xvals[0]:
        return yvals[0]
    for i in range(len(xvals)-1):
        if val < xvals[i+1]:
            # Perform a linear interpolation in interval [xvals[i], xvals[i+1]]
            dx = xvals[i+1] - xvals[i]
            dy = yvals[i+1] - yvals[i]
            ratio = (val - xvals[i]) / dx
            return yvals[i] + ratio * dy
    return yvals[-1]

@njit(parallel=True, fastmath=True)
def apply_curves_numba(image, xvals, yvals):
    """
    Numba-accelerated routine to apply piecewise linear interpolation 
    to each pixel in 'image'.
    - image can be (H,W) or (H,W,3).
    - xvals, yvals are the curve arrays in ascending order.
    Returns the adjusted image as float32.
    """
    if image.ndim == 2:
        H, W = image.shape
        out = np.empty((H, W), dtype=np.float32)
        for y in prange(H):
            for x in range(W):
                val = image[y, x]
                out[y, x] = piecewise_linear(val, xvals, yvals)
        return out
    elif image.ndim == 3:
        H, W, C = image.shape
        out = np.empty((H, W, C), dtype=np.float32)
        for y in prange(H):
            for x in range(W):
                for c in range(C):
                    val = image[y, x, c]
                    out[y, x, c] = piecewise_linear(val, xvals, yvals)
        return out
    else:
        # Unexpected shape
        return image  # Fallback
