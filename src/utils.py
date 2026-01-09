"""
Utility functions.
This module provides various utility functions for calculations, histogram conversions,
and configuration parsing.
"""
import json
import numpy as np
from uncertainties import unumpy
from ROOT import TH1, TH1D, TH2D, TH3D  # type: ignore # pylint: disable=no-name-in-module
import hist

def parse_main_config()->dict:
    """
    Parses the main_config.cfg file and returns a dictionary with the key-value pairs
    
    Returns:
        :return main_config_dict:
            A dictionary with the key-value pairs from the main_config.cfg file
    """
    # Open the main_config.cfg file
    with open('./main_config.cfg', encoding='utf-8') as main_config:
        main_config_dict = {}
        for line in main_config:
            # Skip comments and empty lines
            if line.startswith('#') or line == '\n':
                continue

            # Split the line into key and value
            config_key, value = line.split('=')

            # If the key is signals, split the value by commas
            if config_key.strip() in ['signals', 'channels', 'eras']:
                main_config_dict[config_key.strip()] = value.strip().split(',')
            else:
                if '${' in value and '}' in value:
                    # Extract the variable name from the value
                    var_name = value[value.find('${') + 2:value.find('}')]
                    var_value = main_config_dict.get(var_name, '')
                    value = value.replace('${'+var_name+'}', var_value)
                # Otherwise, just add the key-value pair to the dictionary
                main_config_dict[config_key.strip()] = value.strip()
        return main_config_dict

def initial_loading():
    """
    Initial loading of the main configuration, processes, and systematics.
    This function is called at the beginning of the script to set up the environment.
    """

    main_config = parse_main_config()
    with open(f'./config/processes/{main_config["processes"]}.json',
                encoding='utf-8') as proc_file:
        processes = json.load(proc_file)
    with open(f'./config/systematics/{main_config["systematics"]}.json',
                encoding='utf-8') as syst_file:
        systematics = json.load(syst_file)

    return main_config, processes, systematics

def convert_hist_to_uarray(histogram):
    """
    Coverts a boost-histogram into an uncertainties NumPy uarray
    
    Args:
        :param histogram: The histogram you wish to convert into an uncertainties NumPy uarray
        :return: The uncertainties NumPy uarray
    """
    my_array = unumpy.uarray(
        histogram.values(),
        np.sqrt(histogram.variances())
    )
    return my_array


def get_bin_num(i, nbins):
    """
    Gets the ith bin number accounting for underflow and overflow
        for a given number of bins along an axis

    Args:
        :param i: The ith bin number to return
        :param nbins: The number of bins along an axis
        :return bin_num: The correct bin number accounting for underflow and overflow
    """
    if i == 0: # need to fill underflow
        bin_num = hist.underflow
    elif i == nbins + 1: # need to fill overflow
        bin_num = hist.overflow
    else:
        bin_num = i - 1
    return bin_num

def convert_thx_to_hist(thx: TH1) -> hist.Hist:
    """
    Coverts a THX into a boost-histogram
    
    Args:
        :param thx: The histogram you wish to convert into a boost-histogram
        :return: The boost-histogram
    """
    axes = []
    dim = thx.GetDimension()
    axes.append(
        hist.axis.Variable([
            thx.GetXaxis().GetBinLowEdge(i + 1)
            for i in range(thx.GetNbinsX())
        ] + [thx.GetXaxis().GetBinUpEdge(thx.GetNbinsX())])
    )
    if dim > 1:
        axes.append(
            hist.axis.Variable([
                thx.GetYaxis().GetBinLowEdge(i + 1)
                for i in range(thx.GetNbinsY())
            ] + [thx.GetYaxis().GetBinUpEdge(thx.GetNbinsY())])
        )
    if dim > 2:
        axes.append(
            hist.axis.Variable([
                thx.GetZaxis().GetBinLowEdge(i + 1)
                for i in range(thx.GetNbinsZ())
            ] + [thx.GetZaxis().GetBinUpEdge(thx.GetNbinsZ())])
        )

    h_hist = hist.Hist(*axes, storage=hist.storage.Weight())

    # Fill histogram based on dimension
    if dim == 1:
        for i in range(thx.GetNbinsX() + 2):
            bin_num = get_bin_num(i, thx.GetNbinsX())
            h_hist[bin_num] = [thx.GetBinContent(i), thx.GetBinError(i) ** 2]
    elif dim == 2:
        for i in range(thx.GetNbinsX() + 2):
            for j in range(thx.GetNbinsY() + 2):
                bin_num_x = get_bin_num(i, thx.GetNbinsX())
                bin_num_y = get_bin_num(j, thx.GetNbinsY())

                h_hist[bin_num_x, bin_num_y] = [thx.GetBinContent(i, j),
                                                thx.GetBinError(i, j) ** 2]
    elif dim == 3:
        for i in range(thx.GetNbinsX() + 2):
            for j in range(thx.GetNbinsY() + 2):
                for k in range(thx.GetNbinsZ() + 2):
                    bin_num_x = get_bin_num(i, thx.GetNbinsX())
                    bin_num_y = get_bin_num(j, thx.GetNbinsY())
                    bin_num_z = get_bin_num(k, thx.GetNbinsZ())

                    h_hist[bin_num_x, bin_num_y, bin_num_z] = [thx.GetBinContent(i, j, k),
                                                               thx.GetBinError(i, j, k) ** 2]

    return h_hist

def convert_hist_to_thx(histogram):
    """
    Coverts a boost-histogram into a THX

    Args:
        :param thx: The histogram you wish to convert into a boost-histogram
        :return: The THX histogram
    """
    num_axes = len(histogram.axes)
    num_bins = [len(histogram.axes[i].edges) - 1 for i in range(num_axes)]

    values = histogram.values(flow=True)
    variances = histogram.variances(flow=True)

    if num_axes == 1:
        thx_hist = TH1D('', '', num_bins[0], histogram.axes[0].edges)
        for i in range(num_bins[0] + 2):
            thx_hist.SetBinContent(i, values[i])
            thx_hist.SetBinError(i, np.sqrt(variances[i]))
    elif num_axes == 2:
        thx_hist = TH2D('', '', num_bins[0], histogram.axes[0].edges,
                        num_bins[1], histogram.axes[1].edges)
        for i in range(num_bins[0] + 2):
            for j in range(num_bins[1]):
                thx_hist.SetBinContent(i, j, values[i][j])
                thx_hist.SetBinError(i, j, np.sqrt(variances[i][j]))
    elif num_axes == 3:
        thx_hist = TH3D('', '', num_bins[0], histogram.axes[0].edges,
                        num_bins[1], histogram.axes[1].edges, num_bins[2], histogram.axes[2].edges)
        for i in range(num_bins[0] + 2):
            for j in range(num_bins[1]):
                for k in range(num_bins[2]):
                    thx_hist.SetBinContent(i, j, k, values[i][j][k])
                    thx_hist.SetBinError(i, j, k, np.sqrt(variances[i][j][k]))
    else:
        print('Error! Cannot make THX above TH3!')
        raise NotImplementedError

    return thx_hist

def convert_uarray_to_hist(histogram, my_array):
    """
    Coverts an uncertainties NumPy uarray into a boost-histogram
    
    Args:
        :param histogram: The histogram you want to be filled with the @param my_array content
        :param my_array:
            The uncertainties NumPy uarray that you are using
            to fill the contents of the boost-histogram
        :return: The boost-histogram after it has been filled with my_array
    """
    histogram[...] = np.concatenate(
        (unumpy.nominal_values(my_array)[..., None], (unumpy.std_devs(my_array) ** 2)[..., None]),
        axis=-1
    )
    return histogram
