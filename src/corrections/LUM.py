"""
    Module for applying corrections from LUM recommendations
"""
import correctionlib
import yaml
import awkward as ak

def pileup_weights(events, cfg):
    """
    Apply LUM recommended pileup weights
    Parameters:
    events: awkward array
        The event collection with Pileup_nTrueInt attribute
    cfg: dict
        Configuration dictionary containing 'data_dir' and 'era' keys
    Returns:
    awkward array
        Pileup weights for each event
    """

    if cfg["isData"] == "True":
        events["puWeight"] = ak.ones_like(events.eventNumber)
        events["puWeight_UP"] = ak.ones_like(events.eventNumber)
        events["puWeight_DOWN"] = ak.ones_like(events.eventNumber)
        return events

    # Load LUM configuration file
    with open(cfg["data_dir"]+"/Corrections/LUM/puWeights.yml", 'r', encoding='utf-8') as f:
        lum_cfg = yaml.safe_load(f)["puWeights"][cfg["era"]]

    # Load correction set
    lum_corr = correctionlib.CorrectionSet.from_file(lum_cfg["file"])
    pu_weight = lum_corr[lum_cfg["correction_name"]]

    # Evaluate pileup weights for each event
    events["puWeight"] = pu_weight.evaluate(events.pileUp.nTrueInt, "nominal")
    events["puWeight_UP"] = pu_weight.evaluate(events.pileUp.nTrueInt, "up")
    events["puWeight_DOWN"] = pu_weight.evaluate(events.pileUp.nTrueInt, "down")

    return events
