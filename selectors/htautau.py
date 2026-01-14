"""
    Processor for dilepton selection
"""
import numpy as np
import awkward as ak
from processor import SelectionProcessor
from object_selection import trailing_selection
from selection_utils import lepton_merging, dilepton_pairing, get_4vector_sum,\
    delta_r, add_to_obj, update_collection
import corrections.JME as JME
import corrections.LUM as LUM
import corrections.EGM as EGM
import corrections.MUO as MUO
import corrections.TAU as TAU

class Selector(SelectionProcessor):
    """Processor for dilepton ttbar event selection and tree creation."""
    def __init__(self, selection_cfg):
        super().__init__(selection_cfg)
        self.step_tag = "tree_variables_"
        # Additional initialization can be added here

    def pre_selection(self, events):
        """Pre-selection steps before main selection process."""
        super().pre_selection(events)
        # Any pre-selection steps can be added here
        events = LUM.pileup_weights(events, self.cfg)

        ### Object selection
        ## Correction
        events = EGM.electron_corr(events, self.cfg)
        ## Electron selection
        electron = events.Electron
        # Pt cut
        electron = electron[electron.corr_pt >= 10.0]
        # Eta cut
        electron = electron[np.abs(electron.eta) <= 2.5]
        # Eta clustering cut
        electron = electron[
            (np.abs(electron.eta) > 1.566) | (np.abs(electron.eta) < 1.4442)
            ]
        # ID cut
        electron = electron[electron.cutBased >= 4] # Tight
        electron = EGM.electron_sf(electron, "Tight", self.cfg)
        # dxy cut
        electron = electron[np.abs(electron.dxy) <= 0.045]
        # dz cut
        electron = electron[np.abs(electron.dz) <= 0.02]
        # Iso cut
        electron = electron[electron.miniPFRelIso_all <= 0.5]

        events = update_collection(events, "Electron", electron)

        ## Muon selection
        ## correction
        events = MUO.muon_corr(events, self.cfg)
        muon = events.Muon
        # Pt cut
        muon = muon[muon.corr_pt >= 10.0]
        # Eta cut
        muon = muon[np.abs(muon.eta) <= 2.4]
        # Iso cut
        muon = muon[muon.pfRelIso04_all <= 0.5]
        # ID cut
        muon = muon[muon.tightId] # boolean mask
        muon = MUO.muon_sf(muon, "NUM_TightID_DEN_TrackerMuons", self.cfg)
        #muon = MUO.muon_sf(muon, "NUM_TightPFIso_DEN_TightID", self.cfg)
        # dxy cut
        muon = muon[np.abs(muon.dxy) <= 0.045]
        # dz cut
        muon = muon[np.abs(muon.dz) <= 0.02]

        events = update_collection(events, "Muon", muon)

        ## Tau selection
        ## correction
        events = TAU.tau_sf_corr(events,
                            working_points={
                                "e_to_tau": "Tight",
                                "mu_to_tau": "Tight",
                                "jet_to_tau": "Tight"
                            },
                            cfg=self.cfg,
                            dependency="pt"
                            )
        tau = events.Tau
        # Pt cut
        tau = tau[tau.pt >= 25.0]
        # Eta cut
        tau = tau[np.abs(tau.eta) <= 2.5]
        # ID cut
        tau = tau[tau.idDeepTau2018v2p5VSe >= 6] # Tight
        tau = tau[tau.idDeepTau2018v2p5VSmu >= 4] # Tight
        tau = tau[tau.idDeepTau2018v2p5VSjet >= 6] # Tight
        # dz cut
        tau = tau[np.abs(tau.dz) <= 0.02]

        events = update_collection(events, "Tau", tau)

        ## Merge electrons and muons into leptons
        events["lepton"] = lepton_merging(events, include_tau=True)
        events["lep"], events["lbar"] = dilepton_pairing(events.lepton)
        events["llbar"] = get_4vector_sum(events.lep, events.lbar, corrected=True)

        ## Define reco channels
        pdg_lep = events.lep.pdgId
        pdg_lbar = events.lbar.pdgId
        self.channels = {
            "etau": (pdg_lep == 11) & (pdg_lbar == -15) | (pdg_lep == 15) & (pdg_lbar == -11),
            "mutau": (pdg_lep == 13) & (pdg_lbar == -15) | (pdg_lep == 15) & (pdg_lbar == -13),
            "tautau": (pdg_lep == 15) & (pdg_lbar == -15)
        }

        ## Add to jetsAK4
        events = add_to_obj(
            events, "Jet",
            {
                "DeltaR_lep": delta_r(events.Jet, events.lep),
                "DeltaR_lbar": delta_r(events.Jet, events.lbar)
            }
        )

        ## jetsAK4 selection
        jets = events.Jet
        # Remove Lepton Overlap
        jets_idx = ak.local_index(jets.pt)
        print(jets_idx)
        lep_mask = ~(jets_idx == events.lep.jetIdx)
        lbar_mask = ~(jets_idx == events.lbar.jetIdx)
        jets = jets[lep_mask & lbar_mask]
        # ID selection
        if "jetId" in jets.fields:
            # Following JME recommendations for jet ID
            # https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV
            mask1 = (np.abs(jets.eta) <= 2.7) & (jets.jetId >= 2)
            mask2 = ((np.abs(jets.eta) > 2.7) & (np.abs(jets.eta) <= 3.0)) & \
                (jets.jetId >= 2) & (jets.neHEF < 0.99)
            mask3 = (np.abs(jets.eta) > 3.0) & (jets.jetId >= 2) & (jets.neEmEF < 0.4)
            #mask_tight = mask1 | mask2 | mask3
            mask4 = (np.abs(jets.eta) <= 2.7) & mask1 & (jets.muEF < 0.8) & (jets.chEmEF < 0.8)
            mask_lepveto = mask4 | mask2 | mask3
            jets = jets[mask_lepveto]
        else:
            jets = JME.jet_id(jets, "AK4PUPPI_TightLeptonVeto", self.cfg)
        # jet energy correction
        jets = JME.jet_jerc(events, jets, self.cfg)
        # Pt cut
        jets = jets[jets.corr_pt > 30.0]
        # Eta cut
        jets = jets[np.abs(jets.eta) < 2.5]
        # # cleaning cut
        # jets = jets[
        #     (jets.DeltaR_lep > 0.4) & (jets.DeltaR_lbar > 0.4)
        # ]
        # veto map
        jets = jets[JME.veto_map(jets,"jetvetomap",self.cfg)]

        events = update_collection(events, "Jet_selected", jets)

        # tagger = "UParTAK4" if self.cfg["era"] in ["2024", "2025"]\
        #     else "robustParticleTransformer"
        # corr_type = "kinfit" if self.cfg["era"] in ["2024", "2025"] else "shape"
        # ## B-Jet selection
        # print(f"Applying BTV corrections with tagger {tagger} and correction type {corr_type}")
        # events, bjets = BTV.btagging(events, "Jet_selected", tagger,
        #                                     "M", self.cfg, correction_type=corr_type)
        # events["bJetsAK4"] = bjets

        ## Gen Information (Must Compute It)
        # if self.cfg['isSignal'] == "True":
        #     events["genTTbar"] = get_4vector_sum(events.genTop, events.genTBar)
        #     events["genLLbar"] = get_4vector_sum(events.genLepton, events.genLepBar)

        #     # gen-level dilepton channels
        #     gen_pdg_lep = events.genLepton.pdgId
        #     gen_pdg_lbar = events.genLepBar.pdgId
        #     self.gen_channels = {
        #         "ee": (gen_pdg_lep == 11) & (gen_pdg_lbar == -11),
        #         "mumu": (gen_pdg_lep == 13) & (gen_pdg_lbar == -13),
        #         "emu": ((gen_pdg_lep == 13) & (gen_pdg_lbar == -11))\
        #             | ((gen_pdg_lep == 11) & (gen_pdg_lbar == -13))
        #     }
        return events

    def event_selection(self, events):
        """Dilepton selection process."""
        super().event_selection(events)

        # self.step0_snapshot(events)

        ### Main event selection
        # initialize selector
        self.init_selection()

        # Based on twiki recommendations
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2
        flags = [
            'goodVertices',
            'globalSuperTightHalo2016Filter',
            'EcalDeadCellTriggerPrimitiveFilter',
            'BadPFMuonFilter',
            'BadPFMuonDzFilter',
            'hfNoisyHitsFilter',
            'eeBadScFilter'
        ]
        if self.cfg['era'] == '2024':
            flags.append('ecalBadCalibFilter')

        met_filters = events.event > 0 # initialize to true mask
        for flag in flags:
            met_filters = met_filters & events.Flag[flag]

        # step1a
        self.add_selection_step(
            step_label="METFilters",
            mask=met_filters,
            parent="init"
        )


        # step1b
        # self.add_selection_step(
        #     step_label="Triggers",
        #     mask={
        #         chan: self.dilepton_hlt_mask(
        #             events, chan, self.mappings['HLTPaths'], self.cfg
        #         )
        #         for chan in self.channels
        #     },
        #     channel_wise=True,
        #     parent = "METFilters"
        # )

        # step1
        self.add_selection_step(
            step_label="PrimaryVertex",
            mask=(events.PV.npvsGood > 0),
            parent="METFilters"
        )

        # step3
        self.add_selection_step(
            step_label="LeptonInvariantMass",
            mask=(events.llbar.mass > 20),
            parent="PrimaryVertex"
        )

        # step4
        self.add_selection_step(
            step_label="JetMultiplicity",
            mask=(ak.num(events.Jet_selected, axis=1) >= 2),
            parent="LeptonInvariantMass"
        )

        # self.create_cutflow_histograms(events, step7)

        self.make_snapshot(events, "METFilters", step_name="stepMET")
        self.make_snapshot(events, "PrimaryVertex", step_name="stepPV")
        self.make_snapshot(events, "LeptonInvariantMass", step_name="stepLepInvMass")
        self.make_snapshot(events, "JetMultiplicity", step_name="stepJetMult")

        return events

    def dilepton_hlt_mask(self, events, channel, hlt_map, cfg):
        """
        Create HLT mask for dilepton channels
        """
        pass
