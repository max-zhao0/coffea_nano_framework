"""
    Module for applying corrections to jets, based on JME recommendations
"""
import correctionlib
import yaml
import awkward as ak

def veto_map(obj, correction_type, cfg):
    """
    Apply JME recommended jet veto map
    Parameters:
    obj: awkward array
        The jet collection with Eta and Phi attributes
    correction_type: str
        The type of correction to apply (e.g., "jetvetomap", "jetvetomap_all", ...)
    cfg: dict
        Configuration dictionary containing 'data_dir' and 'era' keys
    Returns:
    awkward array of bool
        Mask indicating whether each jet passes the veto map
    """

    # Load JME configuration file
    with open(cfg["data_dir"]+"/Corrections/JME/jetvetomaps.yml", 'r', encoding='utf-8') as f:
        jme_cfg = yaml.safe_load(f)["jetvetomaps"][cfg["era"]]

    # Load correction set
    jme_corr = correctionlib.CorrectionSet.from_file(jme_cfg["file"])
    jet_veto_map = jme_corr[jme_cfg["correction_name"]]

    # Evaluate veto map for each jet
    return jet_veto_map.evaluate(correction_type, obj.eta, obj.phi) == 0

def jet_id(obj, corr_type, cfg):
    """
    Apply JME recommended jet ID
    Parameters:
    obj: awkward array
        The jet collection with attributes needed for jet ID
    corr_type: str
        The type of correction to apply:
            - AK4PUPPI_TightLeptonVeto
            - AK4PUPPI_Tight
            - AK4CHS_TightLeptonVeto
            - AK4CHS_Tight
            - AK8PUPPI_TightLeptonVeto
            - AK8PUPPI_Tight
            - AK8CHS_TightLeptonVeto
            - AK8CHS_Tight
    cfg: dict
        Configuration dictionary containing 'data_dir' and 'era' keys
    Returns:
    awkward array of bool
        Mask indicating whether each jet passes the jet ID
    """

    # Load JME configuration file
    with open(cfg["data_dir"]+"/Corrections/JME/jetid.yml", 'r', encoding='utf-8') as f:
        jme_cfg = yaml.safe_load(f)["jetid"][cfg["era"]]

    # Load correction set
    jme_corr = correctionlib.CorrectionSet.from_file(jme_cfg["file"])
    jet_id_corr = jme_corr[corr_type]

    # Evaluate jet ID for each jet
    jet_id_eval = jet_id_corr.evaluate(
        obj.eta, obj.chHEF, obj.neHEF, obj.chEmEF, obj.neEmEF, obj.muEF, obj.chMultiplicity,
        obj.neMultiplicity, obj.chMultiplicity + obj.neMultiplicity
    )
    print(f"Applied jet ID: {corr_type}")
    return obj[jet_id_eval == 1]

def jet_jerc(events, obj, cfg):
    """
    Apply Jet Energy Corrections (JEC) function.
    Parameters:
    obj: awkward array
        The jet collection with attributes needed for JEC
    cfg: dict
        Configuration dictionary containing 'data_dir' and 'era' keys
    """

    # Load JME configuration file
    with open(cfg["data_dir"]+"/Corrections/JME/jet_jerc.yml", 'r', encoding='utf-8') as f:
        jme_cfg = yaml.safe_load(f)["jet_jerc"][cfg["era"]]

    jme_corr = correctionlib.CorrectionSet.from_file(jme_cfg["file"])
    raw_pt = obj.pt * (1 - obj.rawFactor)
    raw_mass = obj.mass * (1 - obj.rawFactor)

    if cfg["isData"] == "True":
        corr_str = jme_cfg["data_correction"]
        # L1
        L1_corr = jme_corr[corr_str+"_L1FastJet_AK4PFPuppi"].evaluate(obj.area, obj.eta, raw_pt, events.Rho.fixedGridRhoFastjetAll)
        corr_pt = raw_pt * L1_corr
        corr_mass = raw_mass * L1_corr
        # L2Relative
        L2_rel_corr = jme_corr[corr_str+"_L2Relative_AK4PFPuppi"].evaluate(obj.eta, corr_pt)
        corr_pt = corr_pt * L2_rel_corr
        corr_mass = corr_mass * L2_rel_corr
        # L3Absolute
        L3_abs_corr = jme_corr[corr_str+"_L3Absolute_AK4PFPuppi"].evaluate(obj.eta, corr_pt)
        corr_pt = corr_pt * L3_abs_corr
        corr_mass = corr_mass * L3_abs_corr
        # Residuals
        Residual_corr = jme_corr[corr_str+"_L2L3Residual_AK4PFPuppi"].evaluate(obj.eta, corr_pt)
        corr_pt = corr_pt * Residual_corr
        corr_mass = corr_mass * Residual_corr
    else:
        corr_str = jme_cfg["mc_correction"]
        # L1
        L1_corr = jme_corr[corr_str+"_L1FastJet_AK4PFPuppi"].evaluate(obj.area, obj.eta, raw_pt, events.Rho.fixedGridRhoFastjetAll)
        corr_pt = raw_pt * L1_corr
        corr_mass = raw_mass * L1_corr
        # L2Relative
        if len(jme_corr[corr_str+"_L2Relative_AK4PFPuppi"].inputs) == 3:
            L2_rel_corr = jme_corr[corr_str+"_L2Relative_AK4PFPuppi"].evaluate(
                obj.eta, obj.phi, corr_pt
            )
        else:
            L2_rel_corr = jme_corr[corr_str+"_L2Relative_AK4PFPuppi"].evaluate(
                obj.eta, corr_pt
                )
        corr_pt = corr_pt * L2_rel_corr
        corr_mass = corr_mass * L2_rel_corr
        # L3Absolute
        L3_abs_corr = jme_corr[corr_str+"_L3Absolute_AK4PFPuppi"].evaluate(obj.eta, corr_pt)
        corr_pt = corr_pt * L3_abs_corr
        corr_mass = corr_mass * L3_abs_corr
        # L2L3Residuals
        Residual_corr = jme_corr[corr_str+"_L2L3Residual_AK4PFPuppi"].evaluate(obj.eta, corr_pt)
        corr_pt = corr_pt * Residual_corr
        corr_mass = corr_mass * Residual_corr
    
    obj = ak.with_field(obj, corr_pt, "corr_pt")
    obj = ak.with_field(obj, corr_mass, "corr_mass")
    print("Applied JEC to jets.")
    return obj
