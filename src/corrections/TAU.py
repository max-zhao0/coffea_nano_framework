"""
    Module for applying corrections to taus, based on TAU recommendations.
"""
import correctionlib
import yaml
from selection_utils import add_to_obj, update_collection

def tau_sf_corr(events, working_points: dict, cfg: dict, dependency="pt"):
    """
    Apply tau scale factors
    Parameters:
    events: awkward array
        The events containing the tau collection
    working_points: dict
        Dictionary contaiking tau working points for:
            - e_to_tau : fake rate (VVLoose, Tight)
            - mu_to_tau : fake rate (Loose, Medium, Tight, VLoose)
            - jet_to_tau : fake rate (Loose, Medium, Tight, VTight)

    cfg: dict
        Configuration dictionary containing 'data_dir' and 'era' keys
    dependency: str
        Flag: 'pt' = pT-dependent SFs, 'dm' = DM-dependent SFs
    Returns:
    awkward array
        The events with updated tau scale factors
    """

    # Load TAU configuration file
    with open(cfg["data_dir"]+"/Corrections/TAU/tau.yml", 'r', encoding='utf-8') as f:
        tau_cfg = yaml.safe_load(f)["tau"][cfg["era"]]

    # Load correction set
    tau_corr = correctionlib.CorrectionSet.from_file(tau_cfg["file"])

    print("Applying tau ID scale factors...")
    tau = events.Tau
    # Exclude DM 5 and 6
    # https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun3#Decay_mode_selection
    tau = tau[(tau.decayMode <= 1) | (tau.decayMode >= 10)]  # Only 0,1,10,11
    events = update_collection(events, "Tau", tau)
    # Compute scale factors
    tau = events.Tau
    tau_vs_e_sf = tau_corr["DeepTau2018v2p5VSe"].evaluate(
        tau.eta, tau.decayMode, tau.genPartFlav,
        working_points["e_to_tau"], "nom"
    )
    events["Tau", "tauEFakeWeight"] = tau_vs_e_sf
    tau_vs_mu_sf = tau_corr["DeepTau2018v2p5VSmu"].evaluate(
        tau.eta, tau.genPartFlav,
        working_points["mu_to_tau"], working_points["e_to_tau"],
        working_points["jet_to_tau"], "nom"
    )
    events["Tau", "tauMuFakeWeight"] = tau_vs_mu_sf
    tau_vs_jet_sf = tau_corr["DeepTau2018v2p5VSjet"].evaluate(
        tau.pt, tau.decayMode, tau.genPartFlav,
        working_points["jet_to_tau"], working_points["e_to_tau"], "nom",
        dependency
    )
    events["Tau", "tauJetFakeWeight"] = tau_vs_jet_sf
    # Energy scale correction
    # Due to pythia bug, we set scale to 1
    # https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun3#Decay_mode_selection
    tau_e_scale = 0*tau_corr["tau_energy_scale"].evaluate(
        tau.pt, tau.eta, tau.decayMode, tau.genPartFlav, "DeepTau2018v2p5",
        working_points["jet_to_tau"], working_points["e_to_tau"], "nom"
    ) + 1.0
    events["Tau", "corr_pt"] = tau.pt * tau_e_scale
    events["Tau", "corr_mass"] = tau.mass * tau_e_scale
    events["Tau", "scale_correction"] = tau_e_scale

    print("Computed tau ID SFs.")
    return events