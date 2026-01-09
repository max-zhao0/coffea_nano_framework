"""
    Processor for dilepton selection
"""
import numpy as np
import awkward as ak
from processor import SelectionProcessor
from object_selection import trailing_selection
from selection_utils import lepton_merging, dilepton_pairing, get_4vector_sum,\
    delta_r, add_to_obj
import corrections.JME as JME
import corrections.BTV as BTV
import corrections.LUM as LUM
import corrections.EGM as EGM
import corrections.MUO as MUO

class Selector(SelectionProcessor):
    """Processor for dilepton ttbar event selection and minitree creation."""
    def __init__(self, selection_cfg):
        super().__init__(selection_cfg)
        self.step_tag = "ttBar_treeVariables_"
        # Additional initialization for dilepton selection can be added here

    def pre_selection(self, events):
        """Pre-selection steps before main selection process."""
        super().pre_selection(events)
        # Any pre-selection steps specific to dilepton selection can be added here
        events = LUM.pileup_weights(events, self.cfg)

        ### Object selection
        ## Correction
        events = EGM.electron_corr(events, self.cfg)
        ## Electron selection
        electron = events.electron
        # Pt cut
        electron = electron[trailing_selection(
            leading_mask=(electron.corr_pt > 25.0),
            subleading_mask=(electron.corr_pt > 20.0),
            obj_var=electron.corr_pt
        )]
        electron = electron[electron.corr_pt > 20.0]
        # Eta cut
        electron = electron[np.abs(electron.eta) < 2.4]
        # Eta clustering cut
        electron = electron[
            (np.abs(electron.eta) > 1.566) | (np.abs(electron.eta) < 1.4442)
            ]
        # ID cut
        electron = electron[electron.CutBased >= 4] # Tight

        electron = EGM.electron_sf(electron, "Tight", self.cfg)

        events.electron = electron

        ## Muon selection
        ## correction
        events = MUO.muon_corr(events, self.cfg)
        muon = events.muon
        # Pt cut
        muon = muon[trailing_selection(
            leading_mask=(muon.corr_pt > 25.0),
            subleading_mask=(muon.corr_pt > 20.0),
            obj_var=muon.corr_pt
        )]
        muon = muon[muon.corr_pt > 20.0]
        # Eta cut
        muon = muon[np.abs(muon.eta) < 2.4]
        # Iso cut
        muon = muon[muon.pfRelIso04_all < 0.15]
        # ID cut
        muon = muon[muon.tightId] # boolean mask

        muon = MUO.muon_sf(muon, "NUM_TightID_DEN_TrackerMuons", self.cfg)
        muon = MUO.muon_sf(muon, "NUM_TightPFIso_DEN_TightID", self.cfg)

        events.muon = muon

        ## Merge electrons and muons into leptons
        events["lepton"] = lepton_merging(events)

        ## Choose dilepton pairs
        events["lep"], events["lbar"] = dilepton_pairing(events.lepton)
        events["llbar"] = get_4vector_sum(events.lep, events.lbar, corrected=True)

        ## Define reco channels
        pdg_lep = events.lep.pdgId
        pdg_lbar = events.lbar.pdgId
        self.channels = {
            "ee": (pdg_lep == 11) & (pdg_lbar == -11),
            "mumu": (pdg_lep == 13) & (pdg_lbar == -13),
            "emu": ((pdg_lep == 13) & (pdg_lbar == -11)) | ((pdg_lep == 11) & (pdg_lbar == -13))
        }

        ## Add to jetsAK4
        events.Jet = add_to_obj(
            events.Jet,
            {
                "DeltaR_lep": delta_r(events.Jet, events.lep),
                "DeltaR_lbar": delta_r(events.Jet, events.lbar)
            }
        )

        ## jetsAK4 selection
        jets = events.Jet
        # Pt cut
        jets = jets[jets.Pt > 30.0]
        # Eta cut
        jets = jets[np.abs(jets.eta) < 2.4]
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
        # cleaning cut
        jets = jets[
            (jets.DeltaR_lep > 0.4) & (jets.DeltaR_lbar > 0.4)
        ]
        # veto map
        jets = jets[JME.veto_map(jets,"jetvetomap",self.cfg)]

        events["Jet_selected"] = jets

        tagger = "UParTAK4" if self.cfg["era"] in ["2024", "2025"]\
            else "robustParticleTransformer"
        corr_type = "kinfit" if self.cfg["era"] in ["2024", "2025"] else "shape"
        ## B-Jet selection
        print(f"Applying BTV corrections with tagger {tagger} and correction type {corr_type}")
        events, bjets = BTV.btagging(events, "Jet_selected", tagger,
                                            "M", self.cfg, correction_type=corr_type)
        events["bJetsAK4"] = bjets

        ## Gen Information
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

        self.step0_snapshot(events)

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
        self.add_selection_step(
            step_label="Triggers",
            mask={
                chan: self.dilepton_hlt_mask(
                    events, chan, self.mappings['HLTPaths'], self.cfg
                )
                for chan in self.channels
            },
            channel_wise=True,
            parent = "METFilters"
        )

        # step1
        self.add_selection_step(
            step_label="PrimaryVertex",
            mask=(events.PV.npvsGood > 0),
            parent="Triggers"
        )

        # step2
        self.add_selection_step(
            step_label="LeptonMultiplicity",
            mask=(ak.num(events.lepton, axis=1) == 2),
            parent="PrimaryVertex"
        )

        # step3
        self.add_selection_step(
            step_label="LeptonInvariantMass",
            mask=(events.llbar.M > 20),
            parent="LeptonMultiplicity"
        )

        # step4
        z_window = (events.llbar.M < 76) | (events.llbar.M > 106)
        self.add_selection_step(
            step_label="Zwindow",
            mask={
                "ee": z_window,
                "mumu": z_window,
                "emu": events.llbar.M > 0  # always true
            },
            channel_wise=True,
            parent="LeptonInvariantMass"
        )

        self.add_selection_step(
            step_label="InvertZwindow",
            mask={
                "ee": ~z_window,
                "mumu": ~z_window,
                "emu": events.llbar.M < 0  # always false
            },
            channel_wise=True,
            parent="LeptonInvariantMass"
        )

        # step5
        self.add_selection_step(
            step_label="JetMultiplicity",
            mask=(ak.num(events.Jet_selected, axis=1) >= 2),
            parent="Zwindow"
        )

        self.add_selection_step(
            step_label="JetMultiplicity_zWindow",
            mask=(ak.num(events.Jet_selected, axis=1) >= 2),
            parent="InvertZwindow"
        )

        # step6
        self.add_selection_step(
            step_label="MET",
            mask={
                "ee": (events.PuppiMET.Pt > 40),
                "mumu": (events.PuppiMET.Pt > 40),
                "emu": (events.PuppiMET.Pt > 0) # always true
            },
            channel_wise=True,
            parent="JetMultiplicity"
        )

        self.add_selection_step(
            step_label="MET_zWindow",
            mask={
                "ee": (events.PuppiMET.Pt > 40),
                "mumu": (events.PuppiMET.Pt > 40),
                "emu": (events.PuppiMET.Pt > 0) # always true
            },
            channel_wise=True,
            parent="JetMultiplicity_zWindow"
        )

        # step7
        self.add_selection_step(
            step_label="BJetMultiplicity",
            mask=(ak.num(events.bJetsAK4, axis=1) >= 1),
            parent="MET"
        )

        self.add_selection_step(
            step_label="BJetMultiplicity_zWindow",
            mask=(ak.num(events.bJetsAK4, axis=1) >= 1),
            parent="MET_zWindow"
        )

        # self.create_cutflow_histograms(events, step7)

        self.make_snapshot(events, "METFilters", step_name="step1a")
        self.make_snapshot(events, "PrimaryVertex", step_name="step1")
        self.make_snapshot(events, "LeptonMultiplicity", step_name="step2")
        self.make_snapshot(events, "LeptonInvariantMass", step_name="step3")
        self.make_snapshot(events, "Zwindow", step_name="step4")
        self.make_snapshot(events, "InvertZwindow", step_name="step4_zWindow")
        self.make_snapshot(events, "JetMultiplicity", step_name="step5")
        self.make_snapshot(events, "JetMultiplicity_zWindow", step_name="step5_zWindow")
        self.make_snapshot(events, "MET", step_name="step6")
        self.make_snapshot(events, "MET_zWindow", step_name="step6_zWindow")
        self.make_snapshot(events, "BJetMultiplicity", step_name="step7")
        self.make_snapshot(events, "BJetMultiplicity_zWindow", step_name="step7_zWindow")

        return events

    def dilepton_hlt_mask(self, events, channel, hlt_map, cfg):
        """
        Create HLT mask for dilepton channels
        """
        # Build a robust index map from HLT path name -> index
        try:
            # hlt_map may be list/tuple, Awkward, or Dask-Awkward
            names = hlt_map
            if not isinstance(names, (list, tuple)):
                names = ak.to_list(hlt_map)
            hlt_index_map = {}
            for idx, name in enumerate(names):
                if name in hlt_index_map:
                    break
                hlt_index_map[name] = idx
            #hlt_index_map = {name: idx for idx, name in enumerate(names)}
            print(f"HLT index map created with {len(hlt_index_map)} entries.")
        except Exception as e:
            print(f"ERROR: Failed to create HLT index map: {e}")
            # return false mask of correct shape
            return ak.full_like(events.event, False, dtype=bool)

        def false_mask():
            return ak.full_like(events.event, False, dtype=bool)

        # Build masks for all groups present in cfg["HLT"]
        tot_masks = {}
        for grp, grp_hlt in cfg["HLT"].items():
            # For data, if dataset is incompatible with this group, use false mask
            if cfg["isData"] == "True":
                dataset = cfg["process"].split("_")[-1]
                dataset = dataset[0].upper() + dataset[1:] if dataset else ""
                if dataset not in grp_hlt["datasets"]:
                    print(f"Dataset {dataset} not in datasets for group {grp}. Using false mask.")
                    tot_masks[grp] = false_mask()
                    continue

            # Collect indices for the target HLT paths
            idxs = []
            for path in grp_hlt["triggers"]:
                idx = hlt_index_map.get(path)
                if idx is None:
                    print(f"WARNING: HLT path {path} not found in mappings.")
                else:
                    idxs.append(idx)

            if not idxs:
                print(f"WARNING: No valid HLT paths for group {grp}. Using false mask.")
                tot_masks[grp] = false_mask()
                continue

            # OR all target indices for this group
            mask = ak.zeros_like(events.event, dtype=bool)
            for idx in idxs:
                mask = mask | ak.any(events.HLTidx == idx, axis=1)
            tot_masks[grp] = mask

        # Helper to get a mask safely
        def M(name):
            return tot_masks.get(name, false_mask())

        # Combine per data/MC and channel
        if cfg["isData"] == "False":
            # MC: use union of the groups relevant for this dilepton channel
            match channel:
                case "ee":
                    tot_mask = M("ee") | M("se")
                case "mumu":
                    tot_mask = M("mumu") | M("smu")
                case "emu":
                    tot_mask = M("emu") | M("se") | M("smu")
                case _:
                    raise ValueError(f"Channel {channel} not supported.")
            return tot_mask
        else:
            # Data: dataset-specific logic with anti-overlaps
            if channel == "ee":
                match dataset:
                    case "EGamma":
                        tot_mask = M("ee") | M("se")
                    case _:
                        print(f"Dataset {dataset} not supported for channel {channel} in data."
                            " Returning false mask.")
                        # use events.event to create a false mask
                        tot_mask = false_mask()
            elif channel == "emu":
                match dataset:
                    case "MuonEG":
                        tot_mask = M("emu")
                    case "EGamma":
                        tot_mask = M("se") & ~M("emu")
                    case "SingleMuon":
                        tot_mask = M("smu") & ~M("emu") & ~M("se")
                    case "Muon":
                        tot_mask = M("smu") & ~M("emu") & ~M("se")
                    case _:
                        print(f"Dataset {dataset} not supported for channel {channel} in data. "
                            "Returning false mask.")
                        tot_mask = false_mask()
            elif channel == "mumu":
                match dataset:
                    case "Muon":
                        tot_mask = M("mumu") | M("smu")
                    case "SingleMuon":
                        tot_mask = ~M("mumu") & M("smu")
                    case "DoubleMuon":
                        tot_mask = M("mumu")
                    case _:
                        print(f"Dataset {dataset} not supported for channel {channel} in data. "
                            "Returning false mask.")
                        tot_mask = false_mask()
            else:
                raise ValueError(f"Channel {channel} not supported.")
            return tot_mask
