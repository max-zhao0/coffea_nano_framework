"""
    BTV b-tagging corrections 
"""
import numpy as np
import correctionlib
import awkward as ak
import yaml
from selection_utils import add_to_obj

def btagging(events, jets_field, tagger, working_point, cfg, correction_type="shape"):
    """
    Apply b-tagging scale factors
    Parameters:
    jets: awkward array
        The jet collection with Pt, Eta, and BTag attributes
    tagger: str
        The b-tagging algorithm to use (e.g., "deepJet", "particleNet",
        "robustParticleTransformer", "UParTAK4")
    working_point: str
        Working Point of choice : L, M, T, XT, XXT
    cfg: dict
        Configuration dictionary containing 'data_dir' and 'era' keys
    correction_type: str
        Type of correction to apply (shape/kinfit/comb/mujets/light)
        For the working point corrections, the SFs in 'mujets' and 'comb' are for b/c jets.
        The 'mujets' SFs contain only corrections derived in QCD-enriched regions.
        The 'comb' SFs contain corrections derived in QCD and ttbar-enriched regions.
        Hence, 'comb' SFs can be used everywhere, except for ttbar-dileptonic
            enriched analysis regions.
        For the ttbar-dileptonic regions the 'mujets' SFs should be used.
        The 'light' correction is for light-flavoured jets.
    Returns:
    awkward array of float
        Scale factors for each jet
    """

    # Load BTV configuration file
    with open(cfg["data_dir"]+"/Corrections/BTV/btagging.yml", 'r', encoding='utf-8') as f:
        btv_cfg = yaml.safe_load(f)["btagging"][cfg["era"]]

    if tagger not in btv_cfg["taggers"]:
        raise ValueError(f"Tagger {tagger} not found in configuration for era {cfg['era']}:"
                            f" {btv_cfg['taggers']}.")

    tagger_fields = {
        "deepJet": "btagDeepFlavB",
        "particleNet": "btagPNetB",
        "robustParticleTransformer": "btagRobustParTAK4B",
        "UParTAK4": "btagUParTAK4B"
    }

    # Load correction set
    btv_corr = correctionlib.CorrectionSet.from_file(btv_cfg["file"])
    btag_wp = btv_corr[f"{tagger}_wp_values"].evaluate(working_point)

    if "preliminary" in btv_cfg and btv_cfg["preliminary"]:
        btv_cfg["file"] = btv_cfg["file"].replace(".json.gz", "_preliminary.json.gz")
        btv_corr = correctionlib.CorrectionSet.from_file(btv_cfg["file"])

    btag_shape = btv_corr[f"{tagger}_{correction_type}"]

    jets = events[jets_field]

    if cfg["isData"] == "True":
        events = add_to_obj(
            events, jets_field, {'bShapeWeight': ak.ones_like(jets.pt),
                                'bKinfFitWeight': ak.ones_like(jets.pt)}
        )
    else:
        score = jets[tagger_fields[tagger]]
        nan_mask = np.isnan(score)
        score = ak.where(nan_mask, 0.0, score)
        match correction_type:
            case "shape":
                weights = btag_shape.evaluate(
                    "central", jets.hadronFlavour, np.abs(jets.eta), jets.pt, score
                )
                events = add_to_obj(
                    events, jets_field, {'bShapeWeight': ak.where(
                                        nan_mask, 1.0, weights
                                    )
                                    }
                )
            case "kinfit":
                weights = btag_shape.evaluate(
                    "central", working_point, jets.hadronFlavour, np.abs(jets.eta), jets.pt
                )
                events = add_to_obj(
                    events, jets_field, {'bKinfFitWeight': ak.where(
                                        nan_mask, 1.0, weights
                                    )
                                    }
                )
    bmask = events[jets_field][tagger_fields[tagger]] > btag_wp
    bjets = events[jets_field][bmask]

    # TODO: add systematic variations

    return events, bjets
