import struct
import os

import numpy as np

from Orange.data import Table, Domain, FileFormat

from orangecontrib.spectroscopy.io.util import SpectralFileFormat, _spectra_from_image_2d


### This file is based on the development chain indicated below:

## The below file readers are based on file readers written by Specio
## (https://github.com/paris-saclay-cds/specio) with the following
## license:

# BSD 3-Clause License

# Copyright (c) 2017, Guillaume Lemaitre
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This code is partially based on software developed in the Diamond synchrotron.

class BlockReader:
    def __init__(self, data, start=0):
        self.data = data
        self.start = start

    def peek(self, step, format=None, expect_tuple=False):
        block = self.data[self.start: self.start + step]
        return BlockReader.format(block,
                                  format,
                                  expect_tuple=expect_tuple)

    def read(self, step, format=None, expect_tuple=False):
        block = self.peek(step,
                          format=format,
                          expect_tuple=expect_tuple)
        self.step(step)
        return block

    def step(self, step):
        self.start += step

    def atEnd(self, step=0):
        return self.start >= self.size - step

    @property
    def size(self):
        return len(self.data)

    @staticmethod
    def readData(data, start, step, format=None, expect_tuple=False):
        block = data[start: start + step]
        return BlockReader.format(block,
                                  format,
                                  expect_tuple=expect_tuple)

    @staticmethod
    def format(data, format, expect_tuple=False):
        DECODE_FORMATS = [
            "utf-8",
        ]

        if format == None:
            return data

        if format in DECODE_FORMATS:
            assert (not expect_tuple)

            return BlockReader._decode(data, format)

        unpacked = BlockReader._unpack(data, format)

        if expect_tuple:
            return unpacked

        return BlockReader._asSingular(unpacked)

    @staticmethod
    def _decode(data, format):
        return data.decode(format)

    @staticmethod
    def _unpack(data, format):
        return struct.unpack(format, data)

    @staticmethod
    def _asSingular(unpacked):
        assert (len(unpacked) == 1)

        return unpacked[0]


class PerkinElmer:
    @staticmethod
    def varIdDecode(data, var_id, format, step, expect_tuple=False):
        _var_id = BlockReader.readData(data, 0, 2, "<H", False)

        if var_id == _var_id:
            return BlockReader.readData(data, 2, step, format=format,
                                        expect_tuple=expect_tuple)

        raise struct.error(f"Expected {var_id}, but got {_var_id}.")

    @staticmethod
    def createMeta(values, attrs, meta=dict()):
        for attr_name, attr_index in attrs:
            try:
                meta[attr_name] = values[attr_index]
            except IndexError:
                meta[attr_name] = None

        return meta

    @staticmethod
    def decode5100(data):
        name_size = BlockReader.readData(data,
                                         0,
                                         2,
                                         format="<h",
                                         expect_tuple=False)

        name = BlockReader.readData(data, 2, name_size, format="utf-8")

        header_format = '<ddddddddddiiihBhBhBhB'
        header_size = 104

        values = BlockReader.readData(data, name_size + 2,
                                      header_size,
                                      format=header_format,
                                      expect_tuple=True)

        meta = PerkinElmer.createMeta(
            values,
            [
                ("x_delta", 0),
                ("y_delta", 1),
                ("z_delta", 2),
                ("z_start", 3),
                ("z_end", 4),
                ("z_4d_start", 5),
                ("z_4d_end", 6),
                ("x_init", 7),
                ("y_init", 8),
                ("z_init", 9),
                ("n_x", 10),
                ("n_y", 11),
                ("n_z", 12),
                ("text1", 14),
                ("text2", 16),
                ("resolution", 17),
                ("text3", 18),
                ("transmission", 19),
                ("text4", 20),
            ],
            meta={"name": name, }
        )

        return meta

    @staticmethod
    def decode5104(data):
        values = []

        block_reader = BlockReader(data)

        while not block_reader.atEnd(2):
            tag = block_reader.read(2)

            if tag == b'#u':
                value_size = block_reader.read(2,
                                               format="<h",
                                               expect_tuple=False)

                values.append(block_reader.read(value_size,
                                                format="utf-8"))

                block_reader.step(6)

            elif tag == b'$u':
                values.append(block_reader.read(2,
                                                format="<h",
                                                expect_tuple=False))

                block_reader.step(6)

            elif tag == b',u':
                values.append(block_reader.read(2,
                                                format="<h",
                                                expect_tuple=False))

            else:
                block_reader.step(-1)

        meta = PerkinElmer.createMeta(
            values,
            [
                ("analyst", 0),
                ("date", 2),
                ("image_name", 4),
                ("instrument_model", 5),
                ("instrument_serial_number", 6),
                ("instrument_software_version", 7),
                ("accumulations", 9),
                ("detector", 11),
                ("source", 12),
                ("beam_splitter", 13),
                ("apodization", 15),
                ("spectrum_type", 16),
                ("beam_type", 17),
                ("phase_correction", 20),
                ("ir_accessory", 26),
                ("igram_type", 28),
                ("scan_direction", 29),
                ("background_scans", 32),
                ("ir_laser_wave_number_unit", 67),
            ],
        )

        return meta

    @staticmethod
    def decode5105(data):
        return np.frombuffer(data, dtype=np.float32)

    @staticmethod
    def decode25739(data):
        var_size = PerkinElmer.varIdDecode(data,
                                           29987,
                                           "<H",
                                           2,
                                           expect_tuple=False)

        file_path = BlockReader.readData(data,
                                         4,
                                         var_size,
                                         format="utf-8")

        return {'file_path': file_path}

    @staticmethod
    def decode35698(data):
        min_w, max_w = PerkinElmer.varIdDecode(data,
                                               29981,
                                               "<dd",
                                               16,
                                               expect_tuple=True)

        return {'min_wavelength': min_w, 'max_wavelength': max_w}

    @staticmethod
    def decode35699(data):
        min_abs, max_abs = PerkinElmer.varIdDecode(data,
                                                   29981,
                                                   "<dd",
                                                   16,
                                                   expect_tuple=True)

        return {'min_absolute': min_abs, 'max_absolute': max_abs}

    @staticmethod
    def decode35700(data):
        wavelength_step = PerkinElmer.varIdDecode(data,
                                                  29979,
                                                  "<d",
                                                  8,
                                                  expect_tuple=False)

        return {'wavelength_step': wavelength_step}

    @staticmethod
    def decode35701(data):
        n_points = PerkinElmer.varIdDecode(data,
                                           29995,
                                           "<I",
                                           4,
                                           expect_tuple=False)

        return {'n_points': n_points}

    @staticmethod
    def decode35708(data):
        var_size = PerkinElmer.varIdDecode(data,
                                           29974,
                                           "<I",
                                           4,
                                           expect_tuple=False)

        return np.frombuffer(
            BlockReader.readData(data, 6, var_size, format=None),
            dtype=np.float64
        )


class PerkinElmerReader(FileFormat, SpectralFileFormat):
    EXTENSIONS = (".sp", ".fsm",)
    DESCRIPTION = "Perkin Elmer"

    def read_sp(self):
        f = open(self.filename, "rb")
        data = f.read()
        f.close()

        reader = BlockReader(data)

        info = {
            "signature": reader.read(4, format="utf-8"),
            "description": reader.read(40, format="utf-8"),
        }

        decoders = {
            25739: PerkinElmer.decode25739,
            35698: PerkinElmer.decode35698,
            35699: PerkinElmer.decode35699,
            35700: PerkinElmer.decode35700,
            35701: PerkinElmer.decode35701,
            35708: PerkinElmer.decode35708,
        }

        stops = []
        spectrum = []

        block_id, block_size = reader.read(6,
                                           format="<Hi",
                                           expect_tuple=True)

        stops.append(reader.start + block_size)

        while block_id != 122 and not reader.atEnd(2):
            next_block_id = reader.peek(2)

            if next_block_id[1] == 117:
                reader.start = stops[-1]
                stops = stops[:-1]

                while reader.start >= stops[-1]:
                    stops = stops[:-1]

            else:
                block_id, block_size = reader.read(6,
                                                   format="<Hi",
                                                   expect_tuple=True)

                stops.append(reader.start + block_size)

        info.update(PerkinElmer.decode5104(reader.read(block_size)))

        reader.start = stops[1]

        while not reader.atEnd():
            block_id, block_size = reader.read(6,
                                               format="<Hi",
                                               expect_tuple=True)

            if block_id in decoders.keys():
                decoded = decoders[block_id](reader.peek(block_size))

                if isinstance(decoded, dict):
                    info.update(decoded)

                else:
                    spectrum = decoded

            reader.start += block_size

        wavenumbers = np.linspace(info['min_wavelength'],
                                  info['max_wavelength'],
                                  info['n_points'])

        datavals = np.array(spectrum)[None, ...]

        domain = Domain([], None)
        meta_data = Table.from_numpy(domain,
                                     X=np.zeros((len(datavals), 0)))

        meta_data.attributes = info

        return wavenumbers, datavals, meta_data

    def read_fsm(self):
        f = open(self.filename, "rb")
        data = f.read()
        f.close()

        reader = BlockReader(data)

        meta = {
            "signature": reader.read(4, format="utf-8"),
            "description": reader.read(40, format="utf-8"),
        }

        decoders = {
            5100: PerkinElmer.decode5100,
            5104: PerkinElmer.decode5104,
            5105: PerkinElmer.decode5105,
        }

        spectrum = []

        while not reader.atEnd(6):
            block_id, block_size = reader.read(6,
                                               format="<Hi",
                                               expect_tuple=True)

            decoded = decoders[block_id](reader.read(block_size))

            if isinstance(decoded, dict):
                meta.update(decoded)
            else:
                spectrum.append(decoded)

        wavenumbers = np.arange(meta['z_start'],
                                meta['z_end'] + meta['z_delta'],
                                meta['z_delta'])

        datavals = np.squeeze(spectrum)

        domain = Domain([], None)
        meta_data = Table.from_numpy(domain,
                                     X=np.zeros((len(datavals), 0)))

        meta_data.attributes = meta

        return datavals, wavenumbers, meta_data

    def read_spectra(self):
        if self.filename[-2:] == "sp":
            return self.read_sp()

        elif self.filename[-3:] == "fsm":
            intensities, wn, m = self.read_fsm()

            # TODO position calculation should be done by a helper function that accepts units as well
            xpositions = m.attributes['x_init'] + m.attributes['x_delta'] * np.arange(m.attributes['n_x'])
            ypositions = m.attributes['y_init'] + m.attributes['y_delta'] * np.arange(m.attributes['n_y'])

            # TODO this is not nice - we need new helpers to build the coordinates
            y_loc = np.repeat(np.arange(m.attributes['n_y']), m.attributes['n_x'])
            x_loc = np.tile(np.arange(m.attributes['n_x']), m.attributes['n_y'])

            features, final_data, locs_table = _spectra_from_image_2d(intensities, wn,
                                                                      xpositions[x_loc],
                                                                      ypositions[y_loc])

            return features, final_data, locs_table
