import numpy as np
import beam_processor_wip
import process_CST_beams
import datetime
import os
from astropy.time import Time
import h5py



def main(beam_folder, beam_string, save_path, save_name):
    ## set up times of observation

    n_times = 24*6
    ground_temp = 283 #K
    time_array = np.linspace(datetime.datetime(2025,1, 1, 17,15,0), 
                             datetime.datetime(2025,1, 1, 17,15,0) + datetime.timedelta(hours=23, minutes=56, seconds=4), n_times, endpoint=False)
    LST_array = [Time(t, scale='utc').sidereal_time('mean', longitude=np.deg2rad(-2.3044349)).hourangle for t in time_array]
    LST_array =  np.array(LST_array) 

    txt_list = [beam_folder+'/' + s for s in os.listdir(beam_folder)]
    txt_list = [i for i in txt_list if i.endswith('.txt')]

    obs_freqs = [beam_processor_wip.parse_filename(txt, beam_string) for txt in txt_list]
    obs_freqs = np.array(obs_freqs)
    obs_freqs = np.sort(obs_freqs)
    obs_freqs_mhz = obs_freqs / 10**6

    antenna_temp_matrix = np.ones(shape=(len(time_array), len(obs_freqs)))

    reference_map = np.load('ReferenceMaps/ref_map.npy')
    reference_freq_mhz = np.load('ReferenceMaps/ref_freq.npy')

    betas = np.load('ReferenceMaps/beta_map.npy')

    bcf_ref_freq_mhz = 70
    bcf_ref_freq_index = int(np.where(bcf_ref_freq_mhz==obs_freqs_mhz)[0][0])
    print(f'BCF 70 MHz index is {bcf_ref_freq_index}')

    global_21cm_params = [80, -0.1, 15] # get actual values
    print(f'Global Signal Gaussian:')
    print(f'centre_freq = {global_21cm_params[0]} MHz')
    print(f'amplitude = {global_21cm_params[1]} K')
    print(f'sigma = {global_21cm_params[2]} MHz')

    beam_patterns = [beam_processor_wip.read_in_beam_pattern(beam_folder, beam_string, freq) for freq in obs_freqs_mhz] # first index is sky beam, second is ground beam
    print('Beams Loaded...')

    bcf_matrix = np.ones(shape=(len(time_array), len(obs_freqs)))

    # horizon map
    # calculate beam correction factors...
    for a, t in enumerate(time_array):
        rotated_base_map = beam_processor_wip.map_rotator(reference_map, t)
        rotated_beta_map = beam_processor_wip.map_rotator(betas, t)
        bcf_ref_sky_map = beam_processor_wip.return_sky_map(bcf_ref_freq_mhz, rotated_base_map, rotated_beta_map, reference_freq_mhz)
        bcf_ref_sky_temp = np.sum(bcf_ref_sky_map  * beam_patterns[bcf_ref_freq_index][0])
        bcf_ref_beam_int = np.sum(beam_patterns[bcf_ref_freq_index][0])
        for b, f_mhz in enumerate(obs_freqs_mhz):
            bcf_int_numerator = np.sum(beam_patterns[b][0] * bcf_ref_sky_map)
            bcf_int_beam_int = np.sum(beam_patterns[b][0])
            bcf = (bcf_int_numerator * bcf_ref_beam_int) / (bcf_ref_sky_temp * bcf_int_beam_int)
            bcf_matrix[a, b] = bcf
    print('Beams Correction Factors Done...')


    antenna_temp_matrix = np.ones(shape=(len(time_array), len(obs_freqs)))
    # calculate antenna temperatures
    for a, t in enumerate(time_array):
        rotated_base_map = beam_processor_wip.map_rotator(reference_map, t)
        rotated_beta_map = beam_processor_wip.map_rotator(betas, t)
        for b, f_mhz in enumerate(obs_freqs_mhz):
            sky_map = beam_processor_wip.return_sky_map(f_mhz, rotated_base_map, rotated_beta_map, reference_freq_mhz)
            ant_temp = np.sum(beam_patterns[b][0] * sky_map) + (np.sum(beam_patterns[b][1]) * ground_temp)
            antenna_temp_matrix[a, b] = ant_temp
    print('Antenna Temps Done ...')


    antenna_temp_matrix_21cm = np.ones(shape=(len(time_array), len(obs_freqs)))
    ground_reception = np.ones(shape=len(obs_freqs_mhz))
    # calculate antenna temperatures with 21 cm signal
    for a, t in enumerate(time_array):
        rotated_base_map = beam_processor_wip.map_rotator(reference_map, t)
        rotated_beta_map = beam_processor_wip.map_rotator(betas, t)
        for b, f_mhz in enumerate(obs_freqs_mhz):
            sky_map = beam_processor_wip.return_sky_map(f_mhz, rotated_base_map, rotated_beta_map, reference_freq_mhz,
                                                    include21cm=True, gaussian_params_21cm=global_21cm_params)
            ant_temp = np.sum(beam_patterns[b][0] * sky_map) + (np.sum(beam_patterns[b][1]) * ground_temp)
            antenna_temp_matrix_21cm[a, b] = ant_temp
            ground_reception[b] = np.sum(beam_patterns[b][1])
    print('Antenna Temps with 21cm Done ...')
    print('     saving...')

    global_21cm_params = np.array(global_21cm_params)

    with h5py.File(f'{save_path}/{save_name}.hd5f', mode='a') as f:
        f.create_dataset('beam_correction_factors', data=bcf_matrix, dtype=bcf_matrix.dtype)
        f.create_dataset('obs_freqs_mhz', data=obs_freqs_mhz, dtype=obs_freqs_mhz.dtype)
        f.create_dataset('LST', data=LST_array, dtype=LST_array.dtype)

        f.create_dataset('antenna_temps', data=antenna_temp_matrix, dtype=antenna_temp_matrix.dtype)
        f.create_dataset('antenna_temps_21cm', data=antenna_temp_matrix_21cm, dtype=antenna_temp_matrix_21cm.dtype)
        f.create_dataset('global_signal_params', data=global_21cm_params, dtype=global_21cm_params.dtype)
        f.create_dataset('ground_reception', data=ground_reception, dtype=ground_reception.dtype)

    print('    Finised ...')
    print(f'File saved at {save_path}/{save_name}.hd5f')

if __name__ == '__main__':
    
    main(beam_folder='/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/Beams_06_19/HornDryConGround',
         beam_string='DryHornGround',
         save_path='/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/Beams_06_19',
         save_name='HornDryGround')
    
    
    #main('/Users/user/Desktop/RHINO/BeamWork/RHINONoMetalGround',
    #     'HornNoGround',
    #     '/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/Export_v4',
    #     'HornVacuum')
    
    #print('... Redo Wet and Dry Horn Cases ...')

    #main('/Users/user/Desktop/RHINO/BeamWork/UpdatedBeams/HornDry',
    #     'HornDry',
    #     '/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/Export_v4',
    #     'HornDry')
    #main('/Users/user/Desktop/RHINO/BeamWork/UpdatedBeams/HornWet',
    #     'HornWet',
    #     '/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/Export_v4',
    #     'HornWet')
    
    #print('... Short Horn Cases ...')
    
    #main('/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/AhmedBeams2025_04_28/SmallHornPatterns/HornDry',
    #     'HornDry',
    #     '/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/Export_v4',
    #     'SmallHornDry')
    #main('/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/AhmedBeams2025_04_28/SmallHornPatterns/HornWet',
    #     'HornWet',
    #     '/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/Export_v4',
    #     'SmallHornWet')
    #main('/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/AhmedBeams2025_04_28/SmallHornPatterns/HornWetGround',
    #     'HornWetGround',
    #     '/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/Export_v4',
    #     'SmallHornWetGround')
    #main('/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/AhmedBeams2025_04_28/SmallHornPatterns/HornDryGround',
     #    'HornDryGround',
      #   '/Users/user/Desktop/RHINO/Spectral_Indices_RHINO_Paper/Export_v4',
       #  'SmallHornDryGround')
     
    
    

