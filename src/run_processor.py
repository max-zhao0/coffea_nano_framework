"""
    Loads user processor and runs it
"""
import importlib
import sys
import pathlib
import argparse
import json
import yaml
import uproot
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.util import save
import utils

def load_cfg(fw_dir, args):
    """Load configuration for the processor."""
    cfg = args.metadata
    cfg["data_dir"] = fw_dir + "/data"
    cfg["fw_dir"] = fw_dir
    # Load minitree configuration
    with open(fw_dir+"/config/selection/tree_structure.yml", "r", encoding="utf-8") as f:
        cfg["structure"] = yaml.safe_load(f)["tree"]

    try:
        with open(fw_dir+"/config/workingPoints/BTag.json", "r", encoding="utf-8") as f:
            cfg["btag"] = json.load(f)
    except FileNotFoundError:
        cfg["btag"] = {}
        print("BTag working points file not found, proceeding without btag config.")

    with open(fw_dir+"/config/selection/weights.yml", "r", encoding="utf-8") as f:
        cfg["weights"] = yaml.safe_load(f)["Weights"]

    with open(fw_dir+"/config/selection/HLT.yml", "r", encoding="utf-8") as f:
        _file = yaml.safe_load(f)
        try:
            cfg["HLT"] = _file["HLT"][cfg["era"]]
        except KeyError:
            cfg["HLT"] = _file["HLT"][cfg["era"][:4]]
    cfg["tag"] = args.output if args.output != "" else args.input.replace(".root", "")
    cfg["hist_tag"] = args.output_histos if args.output_histos != "" \
            else args.input.replace(".root", "")
    cfg['file'] = args.input

    return cfg

def load_processor(fw_config):
    """Dynamically load the user processor."""
    file_path = pathlib.Path(fw_config["selector_script"])

    # Create a module spec from the file
    spec = importlib.util.spec_from_file_location("user_module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_module"] = module
    spec.loader.exec_module(module)

    return module.Selector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make minitree in a slurm job (selection)")
    parser.add_argument("input", type=str, help="Input NTuple file")
    parser.add_argument("--output", type=str, help="Output minitree tag", default="")
    parser.add_argument("--output_histos", type=str, help="Output histograms tag", default="")
    parser.add_argument("--metadata", type=str, default="", help="Metadata file (default: empty)")
    return parser.parse_args()

def main() -> None:
    """Main function to run the user processor."""
    args = parse_args()
    args.metadata = args.metadata.split(",") if args.metadata else []

    args.metadata = {item.split(":")[0]: item.split(":")[1]
                    for item in args.metadata} if args.metadata != [] else {}
    fw_config = utils.parse_main_config()

    minitree_cfg = load_cfg(fw_config["fw_dir"], args)

    status_path = fw_config["fw_dir"] + "/selection_status/" + \
        args.input.split("/")[-1].replace(".root", "_status.out")
    minitree_cfg["status_file"] = open(status_path, "w", encoding="utf-8")
    minitree_cfg["status_file"].write(f"Processing file: {args.input}\n")

    try:
        # Load user processor
        print("Loading processor...")
        selector_class = load_processor(fw_config)

        events = NanoEventsFactory.from_root(
            {args.input: "NTuple"},
            schemaclass=NanoAODSchema,
            metadata={}
        ).events()

        selector = selector_class(minitree_cfg)
        output = selector.process(events)
        print("Processing events...")

        # print(output)
        # store outputs per channel
        for chan in output["channels"]:
            chan_file = minitree_cfg['tag'].replace('<chan>/',f'{chan}/{chan}_')
            with uproot.recreate(f"{chan_file}.root") as fout:
                print(f"Saving final minitree {chan}...")

                if output["weightedEvents"] is not None:
                    for key, histo in output["weightedEvents"].items():
                        print(f"Saving weightedEvents histogram: {key}")
                        fout[key] = histo

                for key, array in output["minitree"][chan].items():
                    print(f"Saving branch: {key}")
                    if not array:
                        print(f"WARNING: Branch {key} is empty. Skipping...")
                        continue
                    try:
                        fout[key] = array
                    except Exception as e:
                        print(f"ERROR: Could not save branch {key}. Error: {e}")
                        print(array)
                        raise e

                for key, array in output["minitree"].items():
                    if key in output["channels"]:
                        continue
                    print(f"Saving branch: {key}")
                    if not array:
                        print(f"WARNING: Branch {key} is empty. Skipping...")
                        continue

                    try:
                        fout[key] = array
                    except Exception as e:
                        print(f"ERROR: Could not save branch {key}. Error: {e}")
                        print(array)
                        raise e

            print(f"Saved final minitree: {chan_file}.root")
            minitree_cfg["status_file"].write(
                f"Saved final minitree for channel {chan}: {chan_file}.root\n")

        # Saving not channel wise histograms
        for histo_name, histo in output["histograms"].items():
            if histo_name in output["channels"]:
                continue
            print(f"Saving histogram: {histo_name}")
            histo_file = minitree_cfg['hist_tag'].replace('<chan>/', '')
            save(histo, f"{histo_file}_{histo_name}.coffea")
            minitree_cfg["status_file"].write(
                f"Saved histogram {histo_name}: {histo_file}_{histo_name}.coffea\n")

        # Saving channel wise histograms
        for chan in output["channels"]:
            histo_file = minitree_cfg['hist_tag'].replace('<chan>/', f'{chan}/{chan}_')
            for histo_name, histo in output["histograms"][chan].items():
                print(f"Saving histogram for channel {chan}: {histo_name}")
                save(histo, f"{histo_file}_{histo_name}.coffea")
                minitree_cfg["status_file"].write(
                    f"Saved histogram for channel {chan}: {histo_file}_{histo_name}.coffea\n")

        minitree_cfg["status_file"].write("SELECTION COMPLETED\n")
        minitree_cfg["status_file"].close()

    except Exception as e:
        # Print exception in status file
        minitree_cfg["status_file"].write("FAILED:\n")
        minitree_cfg["status_file"].write(str(e) + "\n")
        minitree_cfg["status_file"].close()
        raise e

if __name__ == "__main__":
    main()
