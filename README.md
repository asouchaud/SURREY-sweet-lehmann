# SWEET-SPS: Sparse Weighted Error Iteration for Spatial Sound

SWEET-SPS is a Python implementation of the framework introduced in Izquierdo *et al* “Towards Maximizing a Perceptual *Sweet Spot* for Spatial Sound with Loudspeakers.” 

## Dependencies

To run the scripts and code you need to install the following Python packages:

- `numpy`
- `scipy`
- `cvxpy`
- `matplotlib`
- `sps`
- `json`
- `warnings`
- `abc`
- `time`
- `collections`

## Reproducibility

The notebook `sweet_figures.ipynb` reproduces Fig. 1 and 2 in the paper. 

## Citation

If you use our code in your research, we kindly ask you to cite it as follows.

> P. Izquierdo Lehmann, R. Cádiz, and C. A. Sing Long. “Towards Maximizing a Perceptual *Sweet Spot* for Spatial Sound with Loudspeakers.'” To appear in IEEE Transactions on Audio, Speech and Language Processing.

## Other sources

- The .mat files in the *data* carpets related to coloration evaluation and azimuth localization estimation were constructed using the itd2angle.mat and mckenzie2021.mat files from the AMToolbox, available to the open source community, at https://amtoolbox.org/, and exposed at

> P.Majdak, C.Hollomey, andR.Baumgartner, “Amt 1.0: The toolbox for reproducible research in auditory modeling,” submitted to Acta Acustica, 2021.

- The .mat files in the *hrtfs_wierstorf* are available to the open source community, at DOI 10.5281/zenodo.4459910 (https://zenodo.org/record/4459911), and exposed at

> H. Wierstorf, M. Geier, and S. Spors, “A free database of head related impulse response measurements in the horizontal plane with multiple distances,” in Audio Engineering Society Convention 130. Audio Engineering Society, 2011.

