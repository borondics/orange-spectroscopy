# File readers specific for the SOLEIL synchrotron

import numpy as np
import h5py

from Orange.data.io import FileFormat
from orangecontrib.spectroscopy.data import SpectralFileFormat, _spectra_from_image


class HDF5Reader_HERMES(FileFormat, SpectralFileFormat):
    """ A very case specific reader for HDF5 files from the HEREMES beamline in SOLEIL"""
    EXTENSIONS = ('.hdf5',)
    DESCRIPTION = 'HDF5 file @HERMRES/SOLEIL'

    def read_spectra(self):
        hdf5_file = h5py.File(self.filename)
        if 'entry1/collection/beamline' in hdf5_file and \
                hdf5_file['entry1/collection/beamline'][()].astype('str') == 'Hermes':
            x_locs = np.array(hdf5_file['entry1/Counter0/sample_x'])
            y_locs = np.array(hdf5_file['entry1/Counter0/sample_y'])
            energy = np.array(hdf5_file['entry1/Counter0/energy'])
            intensities = np.array(hdf5_file['entry1/Counter0/data']).T
            return _spectra_from_image(intensities, energy, x_locs, y_locs)
        else:
            raise IOError("Not an HDF5 HERMES file")


class HDF5Reader_ROCK(FileFormat, SpectralFileFormat):
    """ A very case specific reader for hyperspectral imaging HDF5
    files from the ROCK beamline in SOLEIL"""
    EXTENSIONS = ('.h5',)
    DESCRIPTION = 'HDF5 file @ROCK(hyperspectral imaging)/SOLEIL'

    @property
    def sheets(self):
        with h5py.File(self.filename, "r") as dataf:
            cube_nbrs = range(1, len(dataf["data"].keys())+1)

        return list(map(str, cube_nbrs))

    def read_spectra(self):
        if self.sheet:
            cube_nb = int(self.sheet)
        else:
            cube_nb = 1

        with h5py.File(self.filename, "r") as dataf:
            cube_h5 = dataf["data/cube_{:0>5d}".format(cube_nb)]

            # directly read into float64 so that Orange.data.Table does not
            # convert to float64 afterwards (if we would not read into float64,
            # the memory use would be 50% greater)
            cube_np = np.empty(cube_h5.shape, dtype=np.float64)
            cube_h5.read_direct(cube_np)

            energies = np.array(dataf['context/energies'])

        intensities = np.transpose(cube_np, (1, 2, 0))
        height, width, _ = np.shape(intensities)
        x_locs = np.arange(width)
        y_locs = np.arange(height)

        return _spectra_from_image(intensities, energies, x_locs, y_locs)
