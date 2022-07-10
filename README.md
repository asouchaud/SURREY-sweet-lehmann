# SWEET-SPS: Sparse Weighted Error Iteration for Spatial Sound

SWEET-SPS is a Python implementation of the framework introduced in Izquierdo *et al*. ''Towards Maximizing a Perceptual *Sweet Spot* for Spatial Sound with Loudspeakers.'' 

## Dependencies

To run the scripts and code you need to install the following Python packages:

> `numpy`, `scipy`, `cvxpy`, `matplotlib`, `sps`, `json`, `warnings`, `abc`,  `time`, `collections`


## Citation

If you use our code in your research, we kindly ask you to cite it as follows.

> P. Izquierdo Lehmann, R. Cádiz, and C. A. Sing Long. “Towards Maximizing a Perceptual *Sweet Spot* for Spatial Sound with Loudspeakers.'” To appear in IEEE Transactions on Audio, Speech and Language Processing.

## Reproducibility

The repository includes scripts to reproduce the figures appearing in published papers by the authors. These can be found in the folder ``experiments``.

- The Jupyter notebook `sweet_figures.ipynb` reproduces Fig. 1 and 2 in the paper ''Towards Maximizing a Perceptual *Sweet Spot* for Spatial Sound with Loudspeakers'' by Izquierdo *et al*. 

## Acknowledgements

The scripts reproducing the figures use data obtained with publicly available toolboxes and code. 

- The ``.mat`` files related to the evaluation of coloration and azimuth localization estimation in the folder *data* were computed using the files ``itd2angle.mat`` and ``mckenzie2021.mat`` in AMToolbox. AMToolbox is open source and available to the community at https://amtoolbox.org/. You can use the following citation when using AMToolbox in your work: 

    > P. Majdak, C. Hollomey, and R. Baumgartner, ''AMT 1.x: A toolbox for reproducible research in auditory modeling.'' Acta Acustica. 2022; 6:19. DOI: 10.1051/aacus/2022011.

- The ``.mat`` files in the folder *hrtfs_wierstorf* are available to the open source community at https://zenodo.org/record/4459911 (DOI: 10.5281/zenodo.4459910). You can use the following citation when using these files in your work:

    > H. Wierstorf, M. Geier, A. Raake and S. Spors. ''A free database of head related impulse response measurements in the horizontal plane with multiple distances.'' 130th Audio Engineering Society Convention. January, 2011.

