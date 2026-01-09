"""
    Module for applying corrections to jets, based on JME recommendations
"""
import correctionlib
import yaml

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

def jerc(obj, algorithm, cfg):
    """
    Apply Jet Energy Corrections (JEC) function.
    Parameters:
    obj: awkward array
        The jet collection with attributes needed for JEC
    algorithm: str
        The jet algorithm (e.g., "AK4PUPPI", "AK4CHS")
    cfg: dict
        Configuration dictionary containing 'data_dir' and 'era' keys

    TODO: Implement JEC application 
    Check: https://cms-jerc.web.cern.ch/ApplicationTutorial/#see-what-cat-provides
    """

    pass
