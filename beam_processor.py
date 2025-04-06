import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from pyuvdata import UVBeam
import astropy
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
import h5py

import datetime
import os

import ephem
import pygdsm
import healpy as hp

import argparse


def parse_filename(txt_string, string_leader):
    txt_string = txt_string.split("/")[-1]
    txt_string = txt_string.replace(string_leader, '')
    freq=txt_string[:-4]
    freq = float(freq) * 10**6
    return freq

def read_in_beam_pattern(folder_path, string_leader, freq_mhz):
    uv_beam = UVBeam()
    uv_beam.read_beamfits(filename=folder_path+'/'+string_leader+str(freq_mhz)+'.fits')
    hpx_beam_pattern = uv_beam.data_array[0,0,0]
    
    hpx_beam_pattern = hpx_beam_pattern / np.sum(hpx_beam_pattern)

    theta_array, phi_array = hp.pix2ang(uv_beam.nside, np.arange(hp.nside2npix(uv_beam.nside)))

    mask_values = np.where(np.degrees(theta_array) < 90, 1, 0)

    hpx_beam_pattern *= mask_values
    
    hpx_beam_pattern = get_rotated_beam(hpx_beam_pattern)

    return hpx_beam_pattern

def get_rotated_beam(hpx_beam):
    r = hp.Rotator(rot=[0,90], deg=True)
    rotated_beam = r.rotate_map_pixel(hpx_beam) #rotate beam by 90 deg so in same orientation as map. Zenith in centre
    return rotated_beam

def return_sky_map(freq_mhz, basemap, betas, base_freq_mhz, t_cmb=2.726):
    sky_map = ((basemap - t_cmb)*(freq_mhz / base_freq_mhz) ** betas ) + t_cmb
    return sky_map

def sky2hpix(nside: int, sc: SkyCoord) -> np.ndarray:
    """Convert a SkyCoord into a healpix pixel_id

    Args:
        nside (int): Healpix NSIDE parameter
        sc (SkyCoord): Astropy sky coordinates array

    Returns:
        pix (np.array): Array of healpix pixel IDs
    """
    gl, gb = sc.galactic.l.to("deg").value, sc.galactic.b.to("deg").value
    pix = hp.ang2pix(nside, gl, gb, lonlat=True)
    return pix

def map_rotator(map, obstime = datetime.datetime(2025,1, 1, 0,0,0), longitude=-2.3044349, latitude=53.235002,
                         height=77., frequency=70.):
    n_pix = hp.get_map_size(map)
    n_side = hp.npix2nside(n_pix)
    theta, phi = hp.pix2ang(n_side, np.arange(n_pix))
    observer = ephem.Observer()

    observer.date = obstime
    observer.lon = np.deg2rad(longitude)
    observer.lat = np.deg2rad(latitude)
    observer.elev = height

    ra_zen, dec_zen = observer.radec_of(0, np.pi / 2)
    sc_zen = SkyCoord(ra_zen, dec_zen, unit=("rad", "rad"))
    pix_zen = sky2hpix(nside=n_side, sc=sc_zen)
    vec_zen = hp.pix2vec(n_side, pix_zen)

    ra_zen *= 180 / np.pi
    dec_zen *= 180 / np.pi

    rot = hp.Rotator(coord=["G", "C"])
    eq_theta, eq_phi = rot(theta, phi)

    dec = 90.0 - np.abs(eq_theta * (180 / np.pi))
    ra = ((eq_phi + 2 * np.pi) % (2 * np.pi)) * (180 / np.pi)

    hrot = hp.Rotator(rot=[ra_zen, dec_zen], coord=["G", "C"], inv=True)
    g0, g1 = hrot(theta, phi)
    pix0 = hp.ang2pix(n_side, g0, g1)

    sky_rotated = map[pix0]

    return sky_rotated


def compute_antenna_temp_mp(args):
    t_obs, freq_mhz, beam_folder, beam_string = args
    reference_map = np.load('/Users/user/Desktop/RHINO/BeamWork/ReferenceMaps/ref_map.npy')
    beta_map = np.load('/Users/user/Desktop/RHINO/BeamWork/BetaMaps/beta_map.npy')
    ref_map_freq_mhz = np.load('/Users/user/Desktop/RHINO/BeamWork/ReferenceMaps/ref_freq.npy')
    beam_pattern = read_in_beam_pattern(beam_folder, beam_string, freq_mhz)
    sky_map_f = return_sky_map(freq_mhz, reference_map, beta_map, ref_map_freq_mhz, t_cmb=2.726)

    rotated_map = map_rotator(sky_map_f, t_obs)
    antenna_temp = np.sum(rotated_map * beam_pattern)

    return antenna_temp


def beta_manipulation(base_beta, smoothing_deg, rms):
    base_beta += np.random.normal(loc=0, scale=rms, size=len(base_beta))

    base_beta = - np.abs(base_beta)
    base_beta = hp.smoothing(base_beta, sigma=np.deg2rad(smoothing_deg))

    return base_beta

def main_serial(beam_folder, beam_string, n_times, beta_smoothing_deg, beta_rms, process_i):
    betas = np.load('ReferenceMaps/beta_map.npy')

    betas = beta_manipulation(betas, beta_smoothing_deg, beta_rms)

    reference_map = np.load('ReferenceMaps/ref_map.npy')
    reference_freq_mhz = np.load('ReferenceMaps/ref_freq.npy')

    txt_list = [beam_folder+'/' + s for s in os.listdir(beam_folder)]
    txt_list = [i for i in txt_list if i.endswith('.txt')]

    obs_freqs = [parse_filename(txt, beam_string) for txt in txt_list]
    obs_freqs = np.array(obs_freqs)
    obs_freqs = np.sort(obs_freqs)
    obs_freqs_mhz = obs_freqs / 10**6

    beam_patterns = [read_in_beam_pattern(beam_folder, beam_string, freq) for freq in obs_freqs_mhz]

    time_array = np.linspace(datetime.datetime(2025,1, 1, 17,15,0), 
                             datetime.datetime(2025,1, 1, 17,15,0) + datetime.timedelta(hours=23, minutes=56, seconds=4), n_times, endpoint=False)

    antenna_temp_matrix = np.ones(shape=(len(time_array), len(obs_freqs)))

    for a, t in enumerate(time_array):
        rotated_base_map = map_rotator(reference_map, t)
        rotated_beta_map = map_rotator(betas, t)
        for b, f_mhz in enumerate(obs_freqs_mhz):
            sky_map = return_sky_map(f_mhz, rotated_base_map, rotated_beta_map, reference_freq_mhz)
            ant_temp = np.sum(beam_patterns[b] * sky_map)
            antenna_temp_matrix[a, b] = ant_temp
    

    return antenna_temp_matrix





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
    ## Read in Beam Folder and Beam String from argparse


    n_times = 24*6

    n_realisations = 3
    
    time_array = np.linspace(datetime.datetime(2025,1, 1, 17,15,0), 
                             datetime.datetime(2025,1, 1, 17,15,0) + datetime.timedelta(hours=23, minutes=56, seconds=4), n_times, endpoint=False)
    LST_array = [Time(t, scale='utc').sidereal_time('mean', longitude=np.deg2rad(-2.3044349)).hourangle for t in time_array]
    LST_array =  np.array(LST_array) 


    smoothing_list_deg = [1]#, 2, 4, 8, 16]
    beta_rms_list = [0.005]#, 0.01 , 0.05, 0.1, 0.5]

    txt_list = [beam_folder+'/' + s for s in os.listdir(beam_folder)]
    txt_list = [i for i in txt_list if i.endswith('.txt')]


    obs_freqs_mhz = [parse_filename(txt, beam_string) for txt in txt_list]
    obs_freqs_mhz = np.sort(np.array(obs_freqs_mhz) / 10**6)
    print(obs_freqs_mhz)

    with h5py.File('Generated_Data/'+beam_string+'.hd5f', mode='a') as f:
                f.create_dataset('LST_Times', data=LST_array, dtype=LST_array.dtype)
                f.create_dataset('Frequencies_MHz', data=obs_freqs_mhz, dtype=obs_freqs_mhz.dtype)

                for s in smoothing_list_deg:
                    smooth_group = f.create_group('SmoothScale_'+str(s))
                    for r in beta_rms_list:
                        rms_group = smooth_group.create_group('rms_'+str(r))
    

    for s in smoothing_list_deg:
        for r in beta_rms_list:
            ant_temp_list = []
            for i in np.arange(n_realisations):
                antenna_temp = main_serial(beam_folder, beam_string, n_times, s, r, i) # paralise this section. order doesn't matter so just regular pool.
                ant_temp_list.append(antenna_temp)
                print(i)
            ant_temp_list = np.array(ant_temp_list)

            with h5py.File('Generated_Data/'+beam_string+'.hd5f', mode='a') as f:
                smooth_group = f['SmoothScale_'+str(s)]
                rms_group = smooth_group['rms_'+str(r)]
                for n,a in enumerate(ant_temp_list):
                    rms_group.create_dataset('AntTemps_'+str(s)+'_'+str(r)+'_'+str(n), data=a, dtype=a.dtype)
    print('done..')
                



            # Write to hd5f using s and r for the group and for each element in the list, generate a dataset using the index to number.



