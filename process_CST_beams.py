"""
process_CST_beams.py
Jordan Norris

Processes Beam Models created by Dr. Ahmed Elmakadema in CST by utilising the uvbeam object to interpolate the beam into healpix
and subsequently upscaling to the same n-side as the GSM maps produces by pygdsm (CITE)

"""



import multiprocessing
import numpy as np
from pyuvdata import UVBeam
import healpy as hp
import os

import argparse


def parse_filename(txt_string, string_leader):
    """
    Parses the frequency of the beam pattern from the file name.
    Input:
        txt_string: (str) Filepath of the .txt string
        string_leader: (str) String leading the frequency i.e string_leader70.0.txt
    Output:
        freq: (float) Frequency in Hz of the File 
    """
    txt_string = txt_string.split("/")[-1]
    txt_string = txt_string.replace(string_leader, '')
    freq=txt_string[:-4]
    freq = float(freq) * 10**6
    return freq

def get_UV_beam_from_txt(filepath, feedname='', feedversion='',model_name='', model_version='' , telescope_name='', centre_freq='70e6'):
    """
    Returns a uvbeam object for a given frequency based on the filepath of the .txt file produces in the CST simulation
    Input:
        filepath: (str) Filepath of the .txt string
    Output:
        uvb: (uvBeam object) 
    """
    file = np.loadtxt(filepath, skiprows=2)
    theta = file[:,0]
    phi = file[:,1]
    za = np.deg2rad(theta)
    az = np.deg2rad(phi)
    unique_theta = np.unique(theta)
    unique_phi = np.unique(phi)
    dir_dbi = file[:,2]
    linear = 10**(dir_dbi/10)
     #reshape the linear list into an array based on the za and az , sort into rows of constant za

    za = np.deg2rad(unique_theta)
    az = np.deg2rad(unique_phi)

    uvb = UVBeam() # set up uvbeam object
    uvb.antenna_type = "simple"
    uvb.beam_type = "power"
    uvb.Naxes_vec = 1
    uvb.Nfreqs = 1
    uvb.data_array = np.zeros((1, 1, 1, za.size, az.size))      # (Naxes_vec, 1, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)
    
    for i in np.arange(len(unique_theta)):
        for j in np.arange(len(unique_phi)):
            dummy_lin = linear[theta==unique_theta[i]]
            dummy_phi = phi[theta==unique_theta[i]]
            uvb.data_array[0,0,0,i,j] = dummy_lin[dummy_phi==unique_phi[j]]

    #plt.matshow(uvb.data_array[0, 0,  0])

    #plt.ylabel(r'$\theta$ [deg.]')
    #plt.xlabel(r'$\phi$ [deg.]')
    #plt.colorbar(label='Gain on Isotropic [linear]')
    #plt.show()
    uvb.feed_name = feedname
    uvb.feed_version = feedversion
    uvb.model_name = model_name
    uvb.model_version = model_version
    uvb.telescope_name = telescope_name
    uvb.pixel_coordinate_system = "az_za"
    uvb.Naxes1 = az.size
    uvb.Naxes2 = za.size
    uvb.Npols=1
    uvb.axis1_array = az
    uvb.axis2_array = za
    uvb.data_normalization = "solid_angle"
    uvb.polarization_array = np.array([1])
    uvb.freq_array = np.array([float(centre_freq)])
    uvb.bandpass_array = np.array([float(1)])#set to delta function with np.ones
    uvb.history = "Created by get_UV_beam_from_txt function:  "+str(filepath)

    uvb.check(run_check_acceptability=True)

    return uvb

def upscale_beam_2hpx(uvb, nside):
    """
    Upscales the uvb object to a higher resolution and then converts to a healpix data format 
    Input:
        uvb: (uvbeam object):
        nside:(int) nside of reference map or desired nside
    Output:
        hpx_uvb (uvbeam object): uvbeam object in healpix data format
    """
    az_array = np.deg2rad(np.linspace(0,359.9,3599))
    za_array = np.deg2rad(np.linspace(0,180,1800))
    new_uvb = uvb.interp(az_array=az_array, za_array=za_array, az_za_grid=True, new_object=True)
    hpx_uvb = new_uvb.to_healpix(nside=nside, interpolation_function='az_za_simple', inplace=False)
    return hpx_uvb

def mp_processor(args):
    txt, nside, beam_string, beam_folder = args
    freq = parse_filename(txt, beam_string)
    print(freq)
    beam = get_UV_beam_from_txt(txt)
    hpx_beam = upscale_beam_2hpx(uvb=beam, nside=nside)
    hpx_beam.write_beamfits(beam_folder+'/'+beam_string+str(freq/10**6)+'.fits')

def upscale_cst_beams_and_save(beam_string, beam_folder):
    
    ref_sky_map = np.load('ReferenceMaps/ref_map.npy')
    txt_list = [beam_folder+'/' + s for s in os.listdir(beam_folder)]
    txt_list = [i for i in txt_list if i.endswith('.txt')]
    n_side = hp.get_nside(ref_sky_map)

    arguments = [(txt, n_side, beam_string, beam_folder) for txt in txt_list]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(mp_processor, arguments)
    print('processed')



if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("beam_string", help='.txt String Leader to Parse')
    parser.add_argument('beam_folder', help='Name of Beam Folder containing .fits files. e.g beam_folder/beam_string70.0.fits')
    args = parser.parse_args()
    beam_string = args.beam_string
    if args.beam_folder == None:
        beam_folder = 'Beams/'+beam_string
    beam_folder = 'Beams/'+args.beam_folder

    upscale_cst_beams_and_save(beam_string, beam_folder)