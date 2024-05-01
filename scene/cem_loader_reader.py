# load + read cryoEM data
import numpy as np
import pandas as pd
import struct

from dataset_readers import CameraInfo
from utils.graphics_utils import getWorld2View2, focal2fov

from PIL import Image

# data for a single mrcs, star pair
class MrcsStarData:
    def __init__(self, mrcs_fn, star_fn):
        self.mrcs_fn = mrcs_fn
        self.star_fn = star_fn

        # process .mrcs header
        with open(mrcs_fn, 'rb') as f:
            # read header; assume little-endian
            header = f.read(1024)

            self.nxyz = [0] * 3 # total dimensions of particle stack
            self.offset = 1024

            mode = 0

            # extract a bunch of ints
            for idx, inttup in enumerate(struct.iter_unpack('<i', header[:40])):
                dt = inttup[0]
                if idx < 3:
                    self.nxyz[idx] = dt
                elif idx == 3:
                    self.mode = dt
                elif idx < 7:
                    assert dt == 0, 'Unsupported nxyzstart'
                elif idx < 10:
                    a = [self.nxyz[0], self.nxyz[1], 1]
                    assert dt == a[idx-7], 'Not a valid particle stack'

            # handle mode
            # unsupported modes: complex, 4-bit packed 2 per byte
            dtype = None
            if mode == 0:
                dtype = 'b' # byte / signed 8-bit int
            elif mode == 1:
                dtype = 'h' # short
            elif mode == 2:
                dtype = 'f' # float
            elif mode == 6:
                dtype = 'H' # unsigned short
            elif mode == 12:
                dtype = 'e' # half-precision float
            
            assert dtype != None, 'Unsupported mode: ' + str(mode)

            self.mode = mode
            self.dtype = dtype

            # compute bytes per element
            bpe = 0
            if dtype == 'b':
                bpe = 1
            elif dtype == 'h' or dtype == 'e' or dtype == 'H':
                bpe = 2
            elif dtype == 'f':
                bpe = 4
            self.bytes_per_elem = bpe

            # additional offset if there's an extended header
            self.offset += struct.unpack('<i', header[92:96])[0]

        # process .star data; assume file is small and we can just store all of it in memory
        with open(star_fn) as f:
            data = f.read()

            lines = data.split('\n')

            vars = []
            positions = []
            at_loop = False
            data_start_idx = 0
            for idx, line in enumerate(lines):
                # iterate/skip until loop_
                if not at_loop and line.strip() == 'loop_':
                    at_loop = True
                elif at_loop:
                    # continue until we run out of vars; skip empty lines
                    ln = line.strip()
                    if len(ln) == 0:
                        continue
                    if ln[0] != '_':
                        data_start_idx = idx
                        break

                    name, pos = ln.split(' ')

                    vars.append(name[4:]) # remove _rln prefix
                    positions.append(int(pos[1:]) - 1) # remove #, convert to int then from 1-indexed to 0-indexed
            
            # reindex vars in case positions are listed out of order
            vars = np.array(vars)[positions]
            
            # get list of rows, split into columns and remove empty rows
            data = lines[data_start_idx:]
            dt = [row.split() for row in data if len(row.strip()) > 0]

            # make dataframe, convert to numeric where possible
            df = pd.DataFrame(dt, columns=vars)
            df = df.apply(pd.to_numeric, errors='ignore')

            # put constants in separate dictionary
            const_cols = df.columns[df.nunique() <= 1]
            self.const_star_data = {name:val for (name,val) in zip(const_cols, df[const_cols].iloc[0])}

            # remove constant columns, index on image name
            df = df.drop(columns=const_cols)
            df = df.set_index('ImageName')
            self.star_df = df
    
    # on len(), return number of particle images in stack
    def __len__(self):
        return self.nxyz[-1]
    
    # get single particle image from .mrcs file
    def readImage(self, idx):
        image_size_bytes = self.nxyz[0] * self.nxyz[1] * self.bytes_per_elem
        byte_offset = self.offset + idx * image_size_bytes

        with open(self.mrcs_fn, 'rb') as f:
            f.seek(byte_offset)
            bytes_data = f.read(image_size_bytes) 
            image_data = np.zeros(self.nxyz[:2])

            for idx, floattup in enumerate(struct.iter_unpack('<' + self.dtype, bytes_data)):
                image_data[idx // self.nxyz[0], idx % self.nxyz[0]] = floattup[0]

            # rescale and convert to PIL
            im = image_data - np.min(image_data)
            im = im * 255 / np.max(im)
            im = Image.fromarray(np.uint8(im), mode='L')

            return im

    # compute 2D CTF for an image
    def readCTF(self, idx, boxsz=512):
        # boxsz: usually 512(x512) or 256
        # retrieve necessary variables
        varlist = ['Voltage', 'SphericalAberration', 'DetectorPixelSize', 'Magnification', 'AmplitudeContrast', 'DefocusU', 'DefocusV', 'DefocusAngle']
        vardict = {}
        for var in varlist:
            if var in self.const_star_data.keys():
                vardict[var] = self.const_star_data[var]
            elif var in self.star_df.columns:
                vardict[var] = self.star_df[var][idx]
            else:
                print('Failed to recover CTF function; missing var', var)
                return None
        
        # compute wavelength
        h = 6.626e-34
        e = 1.602e-19
        c = 3.0e8
        m0 = 9.109e-31
        kV = vardict['Voltage'] * 1000
        wavelength = h / np.sqrt(2*m0*e*kV) / np.sqrt(1+e*kV / (2*m0*c**2)) * 1e10
        #wavelength = 12.26 / np.sqrt(vardict['Voltage']+0.9785*vardict['Voltage']**2/(10.0**6.0))

        # set other constants
        w1 = np.sqrt(1-vardict['AmplitudeContrast']**2)
        w2 = vardict['AmplitudeContrast']
        Cs = vardict['SphericalAberration'] * 1e7 # convert from mm
        df1 = vardict['DefocusU']
        df2 = vardict['DefocusV']
        angle_ast = vardict['DefocusAngle'] * np.pi / 180

        # x, y coordinate arrays
        xs = (np.arange(boxsz, dtype=float) - boxsz/2) / boxsz / vardict['DetectorPixelSize']
        ys = xs

        xs = np.tile(xs, (boxsz,1)).T
        ys = np.tile(ys, (boxsz,1))

        # compute defocus
        angles_spatial = np.arctan2(xs,ys)
        angles_diff = angles_spatial - angle_ast
        defocus = 0.5*(df1+df2 + np.cos(2*angles_diff)*(df1-df2))

        # compute g^2 value matrix
        g2s = xs ** 2 + ys ** 2

        # compute chi
        c1 = np.pi*wavelength*g2s
        c2 = -c1*Cs*g2s*(wavelength**2)/2
        chi = c1*defocus + c2
        
        # finally return CTF
        return -w1 * np.sin(chi) - w2 * np.cos(chi)
    
    # get caminfo object for particle image
    def readCamInfo(self, idx, focal=1e7):
        # focal = arbitrarily large focal length to simulate orthographic projection without modifying renderer

        # euler angles to rot matrix; adapted from RELION, 3x3 (non-homogeneous) case
        def euler2rotmat(rot, tilt, psi):
            # convert to rad
            d2r = np.pi / 180
            rot = rot * d2r
            tilt = tilt * d2r
            psi = psi * d2r

            # this is probably normal but i'm just copying from relion without thinking tbh
            ca = np.cos(rot)
            cb = np.cos(tilt)
            cg = np.cos(psi)
            sa = np.sin(rot)
            sb = np.sin(tilt)
            sg = np.sin(psi)
            cc = cb * ca
            cs = cb * sa
            sc = sb * ca
            ss = sb * sa

            # populate matrix 
            R = np.zeros((3,3))
            R[0, 0] =  cg * cc - sg * sa
            R[0, 1] =  cg * cs + sg * ca
            R[0, 2] = -cg * sb
            R[1, 0] = -sg * cc - cg * sa
            R[1, 1] = -sg * cs + cg * ca
            R[1, 2] = sg * sb
            R[2, 0] =  sc
            R[2, 1] =  ss
            R[2, 2] = cb

            return R

        # shift; convert to pixel space
        if 'OriginX' in self.star_df.columns:
            shiftx = self.star_df['OriginX'][idx]
            shifty = self.star_df['OriginY'][idx]
        elif 'OriginXAngstroms' in self.star_df.columns:
            shiftx = self.star_df['OriginXAngstroms'][idx] / self.const_star_data['DetectorPixelSize']
            shifty = self.star_df['OriginYAngstroms'][idx] / self.const_star_data['DetectorPixelSize']
        else:
            shiftx = 0
            shifty = 0

        # get Euler angles, convert to 3x3 rotation matrix
        rot = self.star_df['AngleRot'][idx]
        tilt = self.star_df['AngleTilt'][idx]
        psi = self.star_df['AnglePsi'][idx]

        R = euler2rotmat(rot,tilt,psi)

        # get 3D translation vector
        # here we have already applied the rotation matrix R, i.e. we're in camera coordinate space
        # so just shift within camera coordinates
        T = np.zeros(3)
        T[0] = shiftx
        T[1] = shifty
        T[2] = focal # focal / z ~= 1

        # assemble into CameraInfo object
        w = self.nxyz[0]
        h = self.nxyz[1]
        return CameraInfo(uid=idx, R=R, T=T, FovX=focal2fov(focal, w), FovY=focal2fov(focal, h), 
                          image=self.readImage(idx), image_name=self.star_df.index[idx], image_path=self.mrcs_fn)