""" # pylint: disable=invalid-name
    Module for applying EGM corrections.
"""
import correctionlib
import awkward as ak
import yaml
import numpy as np
from selection_utils import add_to_obj

def electron_sf(obj, working_point, cfg):
    """
    Apply electron scale factors
    Parameters:
    obj: awkward array
        The electron collection with Pt and Eta attributes
    working_point: str
        Working Point of choice : Loose, Medium etc.
        Values: Loose, Medium, Reco20to75, RecoAbove75, RecoBelow20,
                Tight, Veto, wp80iso, wp80noiso, wp90iso, wp90noiso
    cfg: dict
        Configuration dictionary containing 'data_dir' and 'era' keys
    Returns:
    awkward array of float
        Scale factors for each electron
    """

    # Load EGM configuration file
    with open(cfg["data_dir"]+"/Corrections/EGM/electron.yml", 'r', encoding='utf-8') as f:
        egm_cfg = yaml.safe_load(f)["electron"][cfg["era"]]

    # Load correction set
    egm_corr = correctionlib.CorrectionSet.from_file(egm_cfg["file"])
    elec_sf = egm_corr[egm_cfg["correction_name"]]

    obj['SCeta'] = obj.deltaEtaSC + obj.Eta
    val_type_idx = None
    inputs = []
    for corr_input in egm_cfg["inputs"]:
        match corr_input:
            case 'year':
                inputs.append(egm_cfg["year"])
            case 'ValType':
                val_type_idx = len(inputs)
                inputs.append("sf")
            case 'WorkingPoint':
                inputs.append(working_point)
            case _:
                if corr_input in obj.fields:
                    inputs.append(obj[corr_input])
                else:
                    raise ValueError(f"Input {corr_input} not found in electron object fields.")
    if val_type_idx is None:
        raise ValueError("ValType input not found in configuration inputs.")
    inputs_UP = inputs.copy()
    inputs_DOWN = inputs.copy()
    inputs_UP[val_type_idx] = "sfup"
    inputs_DOWN[val_type_idx] = "sfdown"
    obj = add_to_obj(
        obj,
        {
            "electronIDWeight": elec_sf.evaluate(*inputs),
            "electronIDWeight_UP": elec_sf.evaluate(*inputs_UP),
            "electronIDWeight_DOWN": elec_sf.evaluate(*inputs_DOWN),
        }
    )
    print("Computed electron ID SFs.")
    return obj

def electron_corr(events, cfg):
    """
    Apply electron energy scale corrections
    """
    # Load EGM configuration file
    with open(cfg["data_dir"]+"/Corrections/EGM/electronSS_EtDependent.yml",
            'r', encoding='utf-8') as f:
        egm_cfg = yaml.safe_load(f)["electronSS_EtDependent"][cfg["era"]]

    # Load correction set
    egm_corr = correctionlib.CorrectionSet.from_file(egm_cfg["file"])
    elec_scale = egm_corr.compound["Scale"]
    elec_smear = egm_corr["SmearAndSyst"]

    if cfg["isData"] == "True":
        # Apply scale correction for data
        scale = elec_scale.evaluate(
            "scale",
            events.runNumber,
            events.Electron.deltaEtaSC + events.Electron.eta,
            events.Electron.r9,
            events.Electron.pt,
            events.Electron.seedGain,
        )
        pt_corr = events.Electron.pt * scale
    else:
        # Apply smear correction for MC
        smear = elec_smear.evaluate(
            "smear",
            events.Electron.pt,
            events.Electron.r9,
            events.Electron.deltaEtaSC + events.Electron.eta,
        )
        nevents = cfg["nEntriesBeforeSelection"]
        if not isinstance(nevents, int):
            nevents = nevents.compute()
        rng = np.random.normal(loc=0.0, scale=1.0, size=nevents)
        smearing = 1 + smear * rng
        pt_corr = events.Electron.pt * smearing

    events.Electron = add_to_obj(
        events.Electron, {"corr_pt": pt_corr}
    )
    print("Applied electron energy corrections.")
    return events
