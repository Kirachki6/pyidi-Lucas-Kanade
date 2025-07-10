import multiprocessing as mp          # needed only for freeze_support()
from pathlib import Path
import numpy as np
from pyidi import VideoReader, LucasKanade
import matplotlib.pyplot as plt

# Need to use main() otherwise problems
def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]          # ..\pyidi-Lucas-Kanade
    cih_file  = repo_root / "data" / "data_synthetic.cih"      # these 2 lines are to recuperate the files being in whatever folder
    # NOTE: pyidi 1.2.0 expects str, not Path → cast!   , newer version might take path directly
    video = VideoReader(str(cih_file))

    lk = LucasKanade(video)     # instantiates the Lucas-Kanade method

    # regular 12 × 40 grid across the whole frame
    n_rows, n_cols = 12, 40
    ys = np.linspace(0, video.image_height - 1, n_rows, dtype=int)
    xs = np.linspace(0, video.image_width  - 1, n_cols, dtype=int)
    points = np.array([[y, x] for y in ys for x in xs])    # map all the points to create a matrix, each row = points coordinates

    lk.set_points(points)                              # what pixels to follow (defined by points)

    lk.configure(
        roi_size=(15, 15),    # subset window
        pad=3,                # guard-band
        processes=4           # use 4 CPU cores; change to 1 to disable parallel
    )

    disp_px = lk.get_displacements(autosave=False)                   # (N_frames, N_pts, 2)

    # ------------------------------------------------------------------
    # 4.  Save quick copy of the results
    # ------------------------------------------------------------------
    #np.save("disp_synthetic.npy", disp_px)


    t_vec = np.arange(0, len(disp_px[0])) * video.fps
    fig, ax = plt.subplots()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Displacement (pixels)')
    ax.plot(t_vec, disp_px[0, :, 0], 'r:', label='2D - x', alpha=0.5)
    ax.plot(t_vec, disp_px[0, :, 1], 'r--', label='2D - y', alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()



if __name__ == "__main__":
    mp.freeze_support()    # harmless on every platform; required for PyInstaller exe
    main()



