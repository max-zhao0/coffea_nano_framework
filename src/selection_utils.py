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

def add_to_obj(events, obj, new_fields: dict):
    """Add new fields to an object."""
    if events is not None and isinstance(obj, str):
        new_obj = events[obj]
        print(f"Adding {new_fields.keys()} to {obj} in events.")
    else:
        new_obj = obj
        print(f"Adding {new_fields.keys()} to provided object.")
    for field_name, field_value in new_fields.items():
        new_obj = ak.with_field(new_obj, field_value, field_name)

    if events is None:
        return new_obj
    return ak.with_field(events, new_obj, obj)

def update_collection(events, obj_name, new_obj):
    """Update an object collection in events."""
    print(f"Editing collection {obj_name} in events.")
    return ak.with_field(events, new_obj, obj_name)

def dilepton_pairing(lepton):
    """Create dilepton pairs from the lepton collection."""
    lep = {}
    lbar = {}
    for field in lepton.fields:
        _field = lepton[field]
        _neg_field = ak.pad_none(_field[lepton.charge == -1], target=1, axis=1)
        _pos_field = ak.pad_none(_field[lepton.charge == 1], target=1, axis=1)
        _neg_field = ak.fill_none(_neg_field, -999)
        _pos_field = ak.fill_none(_pos_field, -999)
        lep[field] = _neg_field[:,0] # pylint: disable=unsubscriptable-object
        lbar[field] = _pos_field[:,0] # pylint: disable=unsubscriptable-object
    return ak.zip(lep), ak.zip(lbar)

def lepton_merging(events, include_tau=True, sort_by_corr_pt=True):
    """Merge all leptons (e,mu,tau) into a single lepton collection sorted by Pt."""
    ## Lepton objects
    lepton_fields = []
    e_fields = list(events.Electron.fields)
    mu_fields = list(events.Muon.fields)

    for field in e_fields:
        if field in mu_fields and field not in lepton_fields or "Weight" in field:
            lepton_fields.append(field)
    for field in mu_fields:
        if field in e_fields and field not in lepton_fields or "Weight" in field:
            lepton_fields.append(field)

    if include_tau:
        # Remove fields that are not common between (e,mu) and tau
        events = add_to_obj(
            events, "Tau", {
                "pdgId": -events.Tau.charge * 15
            }
        )
        tau_fields = list(events.Tau.fields)
        _new_lepton_fields = []
        for field in lepton_fields:
            if field in tau_fields and field not in _new_lepton_fields or "Weight" in field:
                _new_lepton_fields.append(field)
        lepton_fields = _new_lepton_fields

    lepton = {}
    lepton["pt"] = ak.concatenate([events.Electron.pt, events.Muon.pt], axis=1)
    pt_argsort = ak.argsort(lepton["pt"], axis=1, ascending=False)
    if sort_by_corr_pt:
        e_corr_pt = events.Electron.corr_pt if "corr_pt" in events.Electron.fields \
            else events.Electron.pt
        mu_corr_pt = events.Muon.corr_pt if "corr_pt" in events.Muon.fields \
            else events.Muon.pt
        lepton["corr_pt"] = ak.concatenate(
            [e_corr_pt, mu_corr_pt], axis=1)
        pt_argsort = ak.argsort(lepton["corr_pt"], axis=1, ascending=False)
        lepton["corr_pt"] = lepton["corr_pt"][pt_argsort]
    lepton["pt"] = lepton["pt"][pt_argsort]
    for field in lepton_fields:
        if field == "pt":
            continue
        muon_arr = events.Muon[field] if field in events.Muon.fields \
            else ak.ones_like(events.Muon.pt)
        elec_arr = events.Electron[field] if field in events.Electron.fields \
            else ak.ones_like(events.Electron.pt)
        lepton[field] = ak.concatenate( # pylint: disable=unsubscriptable-object
            [elec_arr,muon_arr],
            axis=1)[pt_argsort]

    if include_tau:
        # Adds taus to the lepton collection
        lepton["pt"] = ak.concatenate([lepton["pt"], events.Tau.pt], axis=1)
        pt_argsort = ak.argsort(lepton["pt"], axis=1, ascending=False)
        if sort_by_corr_pt:
            tau_corr_pt = events.Tau.corr_pt if "corr_pt" in events.Tau.fields \
                else events.Tau.pt
            lepton["corr_pt"] = ak.concatenate(
                [lepton["corr_pt"], tau_corr_pt], axis=1)
            pt_argsort = ak.argsort(lepton["corr_pt"], axis=1, ascending=False)
            lepton["corr_pt"] = lepton["corr_pt"][pt_argsort]
        lepton["pt"] = lepton["pt"][pt_argsort]
        for field in lepton_fields:
            if field == "pt":
                continue
            tau_arr = events.Tau[field] if field in events.Tau.fields \
                else ak.ones_like(events.Tau.pt)
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
            electron = events.Electron
            leak_mask = ~((electron.seediPhiOriY > 72) & \
                        (electron.seediEtaOriX < 45) & \
                        (electron.Eta > 1.556))
            electron = electron[leak_mask]
            events.Electron = electron
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
            "eta": lep.eta,
            "phi": lep.phi,
            "mass": lep.mass
        }
    )
    lbar_p4 = vector.zip(
        {
            "pt": lbar.pt,
            "eta": lbar.eta,
            "phi": lbar.phi,
            "mass": lbar.mass
        }
    )
    return (lep_p4 + lbar_p4).mass

def delta_r(obj1, obj2):
    """Computes DeltaR between two objects"""
    return np.sqrt(
        (obj1.eta - obj2.eta)**2 +
        (obj1.phi - obj2.phi)**2
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
        if value[-1] == ".":
            # Add entire collection
            field = value[:-1]
            if field in events.fields:
                saved_obj = {}
                for subfield in events[field].fields:
                    if len(str(events[field][subfield].type).split("* var")) > 2:
                        print(f"WARNING: Subfield {subfield} in {field} has more than 2 var levels. Skipping...")
                        continue
                    if empty_reco and "gen" not in field:
                        saved_obj[subfield] = ak.values_astype(
                            ak.ones_like(events["event"]), float) * -999
                    else:
                        saved_obj[subfield] = events[field][subfield]
                minitree[key] = ak.zip(saved_obj)
            else:
                print(f"WARNING: Field {field} not found in events.")
        else:
            entry = value.split(".")
            entry.append(None)
            field, subfield = entry[:2]
            if field in events.fields:
                if empty_reco and "gen" not in field:
                    minitree[key] = ak.values_astype(ak.ones_like(events["event"]), float) * -999
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
                # minitree[key] = ak.values_astype(ak.ones_like(events["event"]), float) * -999
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
            "eta": obj1.eta,
            "phi": obj1.phi,
            "mass": obj1.mass
        }
    )
    p4_2 = vector.zip(
        {
            "pt": pt_2,
            "eta": obj2.eta,
            "phi": obj2.phi,
            "mass": obj2.mass
        }
    )

    p412 = p4_1 + p4_2
    res = {
        "pt": p412.pt,
        "eta": p412.eta,
        "phi": p412.phi,
        "mass": p412.mass
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
        total_weight = ak.values_astype(ak.ones_like(events["event"]), float)
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
