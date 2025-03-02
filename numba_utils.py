import numpy as np
from numba import njit, prange
import cv2 

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