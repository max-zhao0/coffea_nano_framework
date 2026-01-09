"""
    Basic processor module for coffea framework.
"""
import uproot
import numpy as np
import hist
import dask
import awkward as ak
import copy
from coffea import processor
from coffea.analysis_tools import PackedSelection
from selection_utils import apply_golden_json, detector_defects_mask,\
    make_weights_fields, make_snapshot

class step:
    """
    Docstring for step
    """
    def __init__(self, name, mask_label, parent=None, metadata=None):
        """Initialize step"""
        self.name = name
        self.mask_label = mask_label
        self.metadata = metadata
        if parent:
            self.mask_labels = copy.deepcopy(parent.mask_labels)
            if isinstance(mask_label, dict):
                for chan in self.mask_labels:
                    self.mask_labels[chan] += mask_label[chan]
            else:
                for chan in self.mask_labels:
                    self.mask_labels[chan] += mask_label
            self.has_parent = True
            self.number_of_steps = parent.number_of_steps + 1
        else:
            self.mask_labels = mask_label
            self.has_parent = False
            self.number_of_steps = 1

        print(f"Initialized step {self.name} with mask labels {self.mask_labels}")


class SelectionProcessor(processor.ProcessorABC):
    """Processor template for event selection and minitree creation."""
    def __init__(self, selection_cfg, mode="eager"):
        """Initialize the selection processor with configuration."""
        assert mode in ["eager", "virtual", "dask"]
        self._mode = mode
        self.cfg = selection_cfg
        self.minitree = {}
        self.histograms = {}
        self.initialize_non_ntuple()
        self.step_tag = ""
        self.channels = {}
        self.gen_channels = {}
        self.selector = PackedSelection()
        self.steps = {}
        self.output_mode = "minitree"
        self._make_selection_histograms = True
        self.ban_weights = []

    def initialize_non_ntuple(self):
        """Initialize any non-ntuple data needed for processing"""
        self.mappings = {}
        with uproot.open(self.cfg['file']) as f:
            for key in f["Mapping"].keys():
                self.mappings[key] = f["Mapping"][key].array()

            if self.cfg['isData'] == "False":
                self.weighted_events = {}
                for key in f["EventsBeforeSelection"].keys():
                    value = ak.sum(f["EventsBeforeSelection"][key].array())
                    self.weighted_events[key] = hist.Hist(hist.axis.Variable([0,1],
                                                name="weightedEvents", label="weightedEvents"),
                                                storage=hist.storage.Weight())
                    self.weighted_events[key].fill([0.5], weight=[value])
            else:
                self.weighted_events = None

    def step0_snapshot(self, events):
        """Create a snapshot of events after initial selection"""
        if not self.gen_channels and not self.channels:
            raise ValueError("GenChannels and Channels not defined before step0_snapshot.")
        if not self.gen_channels and self.channels:
            self.gen_channels = self.channels
            print("Warning: GenChannels was empty, set to Channels.")
        if self.gen_channels.keys() != self.channels.keys():
            raise ValueError("GenChannels and Channels keys do not match.")

        if self.cfg['isSignal'] == "True":
            for gen_channel, chan_mask in self.gen_channels.items():
                if gen_channel not in self.minitree:
                    self.minitree[gen_channel] = {}
                self.minitree[gen_channel][self.step_tag+"step0"] = make_snapshot(
                    events[chan_mask],
                    self.cfg['structure'], empty_reco=True
                )

    def make_snapshot(self, events, step_label, step_name=""):
        """Create a snapshot of events at the current selection step"""
        for chan in self.channels:
            if chan not in self.minitree:
                self.minitree[chan] = {}
            print(self.steps[step_label].mask_labels[chan])
            selected_events = events[self.selector.all(*self.steps[step_label].mask_labels[chan])]
            self.minitree[chan][self.step_tag + step_name] = make_snapshot(
                selected_events,
                self.cfg['structure']
            )

    def init_selection(self, metadata=None):
        """Initialize the main event selection process"""
        if not self.channels:
            raise ValueError("Channels dictionary is empty. "
                            "Define channels before initializing selection.")
        mask_labels = {chan: [chan] for chan in self.channels}
        self.steps["init"] = step("init", mask_labels,
                                  metadata=metadata)
        for chan, chan_mask in self.channels.items():
            self.selector.add(chan,chan_mask)

    def add_selection_step(self, step_label, mask, parent, channel_wise=False, metadata=None):
        """Add a selection step to the PackedSelection"""
        if parent is None:
            raise ValueError("Parent step must be defined for add_selection_step")
        if channel_wise:
            mask_labels = {}
            for chan in self.channels:
                self.selector.add(
                    f"{chan}_{step_label}",
                    mask[chan]
                )
                mask_labels[chan] = [f"{chan}_{step_label}"]
        else:
            mask_labels = {chan: [step_label] for chan in self.channels}
            self.selector.add(step_label,mask)
        self.steps[step_label] = step(step_label, mask_labels,
                                    parent=self.steps[parent], metadata=metadata)

    def create_cutflow_histograms(self, events, last_step, weight_field="eventWeight"):
        """Create cutflow histograms for the given events and last step"""
        # nentries = {chan: [ak.num(events, axis=0)] for chan in self.channels}
        nevents = {chan: [ak.sum(events[weight_field], axis=0)] for chan in self.channels}
        nvariances = {chan: [ak.sum(events[weight_field]**2, axis=0)] for chan in self.channels}
        labels = {chan: ["Initial"] for chan in self.channels}
        for chan in self.channels:
            step_labels = last_step.mask_labels[chan]

            for step_label in step_labels:
                labels[chan].append(step_label)
                selected_events = events[self.selector.all(*labels[chan][1:])]
                # nentries[chan].append(ak.num(selected_events, axis=0))
                nevents[chan].append(ak.sum(selected_events[weight_field], axis=0))
                nvariances[chan].append(ak.sum(selected_events[weight_field]**2, axis=0))

        (nevents,) = dask.compute(nevents)
        (nvariances,) = dask.compute(nvariances)

        for chan in self.channels:
            if chan not in self.histograms:
                self.histograms[chan] = {}
            # self.histograms[chan]["nentries"] = hist.Hist(
            #     hist.axis.Variable([0,1],
            #                     name="nentries", label="nentries"),
            #     storage=hist.storage.Weight()
            # )
            # self.histograms[chan]["nentries"].fill(
            #     [0.5]*len(nentries[chan]),
            #     weight=nentries[chan]
            # )

            self.histograms[chan]["nevents"] = hist.Hist(
                hist.axis.StrCategory(labels[chan], name="label", label="labels"),
                storage=hist.storage.Weight()
            )
            self.histograms[chan]["nevents"].fill(
                label=labels[chan],
                weight=nevents[chan]
            )
            self.histograms[chan]["nevents"][...] = np.concatenate(
                (np.array(nevents[chan])[:, None], np.array(nvariances[chan])[:, None]), axis=1
            )

    def process(self, events):
        """ Main process """

        if self.cfg['isData'] == "True":
            # Golden JSON filtering for data
            events = apply_golden_json(events, self.cfg['era'])

        # Detector defects filtering per era
        events = detector_defects_mask(events, self.cfg['era'], self.cfg)

        # Get number of entries before selection
        self.cfg["nEntriesBeforeSelection"] = ak.num(events,axis=0)

        events = self.pre_selection(events)

        # Compute weights for MC
        if self.cfg['isData'] == "False":
            events = make_weights_fields(events, self.cfg['weights'], self.ban_weights)
        else:
            events["eventWeight"] = ak.ones_like(events.eventNumber)

        events = self.event_selection(events)

        # Store histograms selection
        # if self._make_selection_histograms:
        #     for chan in self.channels:
        #         if chan not in self.histograms:
        #             self.histograms[chan] = {}
        #         nminusone = self.selector.nminusone(*self.step_labels[chan])
        #         self.histograms[chan]["nminusone"], _ = nminusone.yieldhist()

        #         cutflow = self.selector.cutflow(*self.step_labels[chan])
        #         yield_outputs = cutflow.yieldhist()
        #         self.histograms[chan]["onecut"] = yield_outputs[0]
        #         self.histograms[chan]["cutflow"] = yield_outputs[1]

        if self.output_mode == "minitree":
            return {
                "minitree": self.minitree,
                "weightedEvents": self.weighted_events,
                "channels": list(self.channels.keys())
            }
        elif self.output_mode == "histogram":
            return {
                "histograms": self.histograms,
                "weightedEvents": self.weighted_events,
                "channels": list(self.channels.keys())
            }
        elif self.output_mode == "both":
            return {
                "minitree": self.minitree,
                "histograms": self.histograms,
                "weightedEvents": self.weighted_events,
                "channels": list(self.channels.keys())
            }
        else:
            raise ValueError(f"Unsupported output mode: {self.output_mode}")


    def event_selection(self, events):
        """User specified selection process to be implemented in subclasses"""
        return events

    def pre_selection(self, events):
        """User specified pre-selection process to be implemented in subclasses"""
        return events

    def postprocess(self, accumulator):
        pass
