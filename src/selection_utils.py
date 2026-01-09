"""
Utility functions for object selection in the analysis framework.
"""
import operator
# import functools
import numpy as np
import vector
import awkward as ak
from coffea.lumi_tools import LumiMask
from corrections.JME import veto_map

def apply_golden_json(events, era):
    """Apply golden JSON mask to data events based on the era."""
    match era:
        case "2022preEE":
            golden_json = LumiMask("/depot/cms/top/jduarteq/run3/top-spincorr-framework/"
                            "data/GoldenJson/Cert_Collisions2022_355100_362760_Golden.json")
            mask = golden_json(events.runNumber, events.lumiBlock)
            return events[mask]
        case "2022postEE":
            golden_json = LumiMask("/depot/cms/top/jduarteq/run3/top-spincorr-framework/"
                            "data/GoldenJson/Cert_Collisions2022_355100_362760_Golden.json")
            mask = golden_json(events.runNumber, events.lumiBlock)
            return events[mask]
        case "2023preBPix":
            golden_json = LumiMask("/depot/cms/top/jduarteq/run3/top-spincorr-framework/"
                            "data/GoldenJson/Cert_Collisions2023_366442_370790_Golden.json")
            mask = golden_json(events.runNumber, events.lumiBlock)
            return events[mask]
        case "2023postBPix":
            golden_json = LumiMask("/depot/cms/top/jduarteq/run3/top-spincorr-framework/"
                            "data/GoldenJson/Cert_Collisions2023_366442_370790_Golden.json")
            mask = golden_json(events.runNumber, events.lumiBlock)
            return events[mask]
        case "2024":
            golden_json = LumiMask("/depot/cms/top/jduarteq/run3/top-spincorr-framework/"
                            "data/GoldenJson/Cert_Collisions2024_378981_386951_Golden.json")
            mask = golden_json(events.runNumber, events.lumiBlock)
            return events[mask]
        case "2025":
            golden_json = LumiMask("/depot/cms/top/jduarteq/run3/top-spincorr-framework/"
                            "data/GoldenJson/Cert_Collisions2025_391658_398860_Golden.json")
            mask = golden_json(events.runNumber, events.lumiBlock)
            return events[mask]
        case _:
            raise ValueError(f"Unsupported era for golden JSON application: {era}")

def add_to_obj(obj, new_fields: dict):
    """Add new fields to an object."""
    new_obj = {}
    for field in obj.fields:
        new_obj[field] = obj[field]

    for field_name, field_value in new_fields.items():
        new_obj[field_name] = field_value

    return ak.zip(new_obj)

def dilepton_pairing(lepton):
    """Create dilepton pairs from the lepton collection."""
    lep = {}
    lbar = {}
    for field in lepton.fields:
        _field = lepton[field]
        _neg_field = ak.pad_none(_field[lepton["Charge"] == -1], target=1, axis=1)
        _pos_field = ak.pad_none(_field[lepton["Charge"] == 1], target=1, axis=1)
        _neg_field = ak.fill_none(_neg_field, -999)
        _pos_field = ak.fill_none(_pos_field, -999)
        lep[field] = _neg_field[:,0] # pylint: disable=unsubscriptable-object
        lbar[field] = _pos_field[:,0] # pylint: disable=unsubscriptable-object
    return ak.zip(lep), ak.zip(lbar)

def lepton_merging(events, include_tau=True):
    """Merge all leptons (e,mu,tau) into a single lepton collection sorted by Pt."""
    ## Lepton objects
    lepton_fields = []
    e_fields = list(events.electron.fields)
    mu_fields = list(events.muon.fields)

    for field in e_fields:
        if field in mu_fields and field not in lepton_fields or "Weight" in field:
            lepton_fields.append(field)
    for field in mu_fields:
        if field in e_fields and field not in lepton_fields or "Weight" in field:
            lepton_fields.append(field)

    if include_tau:
        # Remove fields that are not common between (e,mu) and tau
        tau_fields = list(events.tau.fields)
        _new_lepton_fields = []
        for field in lepton_fields:
            if field in tau_fields and field not in _new_lepton_fields or "Weight" in field:
                _new_lepton_fields.append(field)
        lepton_fields = _new_lepton_fields

    lepton = {}
    lepton["pt"] = ak.concatenate([events.electron.pt, events.muon.pt], axis=1)
    pt_argsort = ak.argsort(lepton["pt"], axis=1, ascending=False)
    lepton["pt"] = lepton["pt"][pt_argsort]
    for field in lepton_fields:
        if field == "pt":
            continue
        muon_arr = events.muon[field] if field in events.muon.fields \
            else ak.ones_like(events.muon.pt)
        elec_arr = events.electron[field] if field in events.electron.fields \
            else ak.ones_like(events.electron.pt)
        lepton[field] = ak.concatenate( # pylint: disable=unsubscriptable-object
            [elec_arr,muon_arr],
            axis=1)[pt_argsort]

    if include_tau:
        # Adds taus to the lepton collection
        lepton["pt"] = ak.concatenate([lepton["pt"], events.tau.pt], axis=1)
        pt_argsort = ak.argsort(lepton["pt"], axis=1, ascending=False)
        lepton["pt"] = lepton["pt"][pt_argsort]
        for field in lepton_fields:
            if field == "pt":
                continue
            tau_arr = events.tau[field] if field in events.tau.fields \
                else ak.ones_like(events.tau.pt)
            lepton[field] = ak.concatenate( # pylint: disable=unsubscriptable-object
                [lepton[field], tau_arr],
                axis=1)[pt_argsort]

    return ak.zip(lepton)

def detector_defects_mask(events, era, cfg):
    """Apply detector defects mask based on the era."""
    match era:
        case "2022postEE":
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis#Notes_on_addressing_EE_issue_in
            # only apply to electrons (photons are not used in our analysis normally)
            electron = events.electron
            leak_mask = ~((electron.seediPhiOriY > 72) & \
                        (electron.seediEtaOriX < 45) & \
                        (electron.Eta > 1.556))
            electron = electron[leak_mask]
            events.electron = electron
            vetomap_mask = veto_map(
                events.jetsAK4, "jetvetomap_eep", cfg
            )
            _mask = ak.all(vetomap_mask, axis=1)
            events = events[_mask]
            return events
        case _:
            return events

def mllbar(lep, lbar):
    """Calculate the invariant mass of the dilepton system."""
    lep_p4 = vector.zip(
        {
            "pt": lep.pt,
            "eta": lep.Eta,
            "phi": lep.Phi,
            "mass": lep.M
        }
    )
    lbar_p4 = vector.zip(
        {
            "pt": lbar.pt,
            "eta": lbar.Eta,
            "phi": lbar.Phi,
            "mass": lbar.M
        }
    )
    return (lep_p4 + lbar_p4).mass

def delta_r(obj1, obj2):
    """Computes DeltaR between two objects"""
    return np.sqrt(
        (obj1.Eta - obj2.Eta)**2 +
        (obj1.Phi - obj2.Phi)**2
    )

ops = {
    '>': operator.gt,
    '<': operator.lt,
    '==': operator.eq,
    '!=': operator.ne,
    '>=': operator.ge,
    '<=': operator.le,
    '&': operator.and_,
    '|': operator.or_,
    '~': operator.invert, # operator.not_
}

def make_snapshot(events, structure, empty_reco=False):
    """
    Create a snapshot of the events based on the provided structure.
    """
    print("Creating snapshot...")
    minitree = {}
    for key, value in structure.items():
        entry = value.split(".")
        entry.append(None)
        field, subfield = entry[:2]
        if field in events.fields:
            if empty_reco and "gen" not in field:
                minitree[key] = ak.values_astype(ak.ones_like(events["eventNumber"]), float) * -999
            else:
                if subfield is None:
                    minitree[key] = events[field]
                else:
                    if subfield not in events[field].fields:
                        print(f"WARNING: Subfield {subfield} not found in {field}.")
                        continue
                    else:
                        minitree[key] = events[field][subfield]
        else:
            print(f"WARNING: Field {field} not found in events.")
            # minitree[key] = ak.values_astype(ak.ones_like(events["eventNumber"]), float) * -999
    return minitree

def get_4vector_sum(obj1, obj2, corrected=False):
    """
    Returns the sum of two 4-vectors.
    """
    pt_1 = obj1.corr_pt if corrected else obj1.pt
    pt_2 = obj2.corr_pt if corrected else obj2.pt
    p4_1 = vector.zip(
        {
            "pt": pt_1,
            "eta": obj1.Eta,
            "phi": obj1.Phi,
            "mass": obj1.M
        }
    )
    p4_2 = vector.zip(
        {
            "pt": pt_2,
            "eta": obj2.Eta,
            "phi": obj2.Phi,
            "mass": obj2.M
        }
    )

    p412 = p4_1 + p4_2
    res = {
        "pt": p412.pt,
        "Eta": p412.eta,
        "Phi": p412.phi,
        "M": p412.mass
    }
    return ak.zip(res)

def make_weights_fields(events, weights_config, ban_weights=None):
    """
    Create weight fields in the events based on the provided weights configuration.
    """
    if ban_weights is None:
        ban_weights = []
    for weight_name, weight_fields in weights_config.items():
        print(f"Creating weight field: {weight_name}")
        print(f"  Composing from fields: {weight_fields}")
        total_weight = ak.values_astype(ak.ones_like(events["eventNumber"]), float)
        for field in weight_fields:
            if field in ban_weights:
                print(f"  Skipping banned weight field: {field}")
                continue
            if "." in field:
                entry = field.split(".")
                field, subfield = entry[:2]
                if field not in events.fields:
                    print(f"  WARNING: Field {field} not found in events. Skipping.")
                    continue
                if subfield not in events[field].fields:
                    print(f"  WARNING: Subfield {subfield} not found in {field}. Skipping.")
                    continue
                new_weight = events[field][subfield]
            else:
                if field not in events.fields:
                    print(f"  WARNING: Field {field} not found in events. Skipping.")
                    continue
                new_weight = events[field]
            if new_weight.layout.minmax_depth != (1,1):
                new_weight = ak.prod(new_weight, axis=1)
            total_weight = total_weight * new_weight
        events[weight_name] = total_weight
    return events
