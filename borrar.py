import cv2

def process_chunk(chunk):
    for i in range(2):
        frames, x, y, channel = chunk.shape
        alpha = 0.5

        # Create a new array for saving the new chunk
        new_chunk = np.zeros((frames*2-1, x, y, channel), dtype=chunk.dtype)

        # Copy the original frames
        new_chunk[::2] = chunk

        # Interpolate the frames and normalize new frame
        for i in range(frames-1):
            blurred_frame1 = cv2.GaussianBlur(chunk[i], (0, 0), 5)
            blurred_frame2 = cv2.GaussianBlur(chunk[i+1], (0, 0), 5)
            new_frame = (1 - alpha) * blurred_frame1 + alpha * blurred_frame2
            new_chunk[i*2+1] = new_frame

        # Interpolate the last frame
        new_chunk[-1] = chunk[-1]

        chunk = new_chunk
