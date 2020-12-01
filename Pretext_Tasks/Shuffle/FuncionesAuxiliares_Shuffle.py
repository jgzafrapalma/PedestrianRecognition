import numpy as np
import random



def ShuffleFrames(frames, n_swaps):
    frames_Shuffle = np.ndarray(frames.shape)

    n_frames = frames.shape[0]

    indexes_start = np.zeros(n_frames)

    # Generación de los indices iniciales de los frames
    for index in range(n_frames):
        indexes_start[index] = index

    while True:

        indexes_end = np.array(indexes_start, copy=True)

        for i in range(n_frames - 1, n_frames - 1 - n_swaps, -1):
            j = random.randint(0, n_frames - 1)

            indexes_end[i], indexes_end[j] = indexes_end[j], indexes_end[i]

        if not equal_arrays(indexes_start, indexes_end):
            break

    # A partir de los indices se obtiene la imagen final
    for i in range(n_frames):
        for j in range(n_frames):
            if indexes_end[j] == i:
                frames_Shuffle[j] = frames[i]


    return frames_Shuffle


def equal_arrays(array1, array2):

    n_elements = len(array1)

    if n_elements != len(array2):
        return False

    for i in range(len(array1)):
        if array1[i] != array2[i]:
            return False

    return True