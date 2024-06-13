import os
import glob
import h5py
import numpy as np
import pandas as pd
import json

def load_spectrum_data(path, spectrum_type):
    files = glob.glob(path)
    num_files = len(files)
    hdf = h5py.File(files[0], 'r')
    data = hdf.get(spectrum_type)
    dataset = np.array(data)
    wavelength = dataset[:, 0]
    intensity = dataset[:, 1]
    for i in range(1, num_files):
        hdf = h5py.File(files[i], 'r')
        data = hdf.get(spectrum_type)
        dataset = np.array(data)
        intensity = np.vstack((intensity, dataset[:, 1]))
    intensity = intensity.T
    return wavelength, intensity

def calculate_average_and_std(intensity):
    intensity_avg = np.mean(intensity, axis=1)
    intensity_std = np.std(intensity, axis=1)
    return intensity_avg, intensity_std

def subtract_background(intensity, background_intensity_avg):
    intensity_sub_bkg = intensity - background_intensity_avg
    return intensity_sub_bkg

def calculate_correction_factor(file_path, fibre_value):
    file_cp_steller = pd.read_excel(file_path)
    cp_steller = np.array(file_cp_steller) #uJ/count
    wavelength = cp_steller[:,0]
    dLp_steller = np.diff(cp_steller[:,0]) # nm
    dLp_steller = np.append(dLp_steller, dLp_steller[0])
    Araw_steller = 3.1416*((fibre_value*10**(-4))/2)**2; # Collection area in cm^2 (thorlabs fiber: d = 3*125.38 um), Steller_Net fiber: d=600 um
    Traw_steller = 100*10**(-6); #%intigration time or life time plasma (in sec.)
    crr_factor_steller = (10**(-6)*cp_steller[:,1])/(Traw_steller*Araw_steller*dLp_steller)
    crr_factor_steller = crr_factor_steller.reshape((crr_factor_steller.shape[0],1))
    return crr_factor_steller

def apply_correction_factor(intensity, correction_factor):
    corrected_intensity = intensity * correction_factor
    return corrected_intensity

def load_data(path, background_path, spectrum_type, correction_factor_file_path):
    wavelength, intensity = load_spectrum_data(path, spectrum_type)
    _, background_intensity = load_spectrum_data(background_path, spectrum_type)
    background_intensity_avg, _ = calculate_average_and_std(background_intensity)
    intensity_sub_bkg = subtract_background(intensity, background_intensity_avg)
    correction_factor = calculate_correction_factor(correction_factor_file_path)
    corrected_intensity = apply_correction_factor(intensity_sub_bkg, correction_factor)
    intensity_avg, intensity_std = calculate_average_and_std(corrected_intensity)
    return wavelength, corrected_intensity, intensity_avg, intensity_std

def load_all_data(config_file, spectrum_type):
    with open(config_file) as f:
        config = json.load(f)
    all_data = []
    for data_config in config['data']:
        path = data_config['path']
        background_path = data_config['background_path']
        correction_factor_file_path = data_config['correction_factor_file_path']
        wavelength, corrected_intensity, intensity_avg, intensity_std = load_data(path, background_path, spectrum_type, correction_factor_file_path)
        all_data.append(corrected_intensity.T)
    return all_data