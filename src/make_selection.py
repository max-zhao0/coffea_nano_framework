"""
    Make object and event selection
"""
import os
import argparse
import yaml
import utils


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make minitree and event selection")
    parser.add_argument("--metadata", type=str, default="",
                        help="Metadata file (default: empty)")
    parser.add_argument("--era", type=str, default="",
                        help="Data-taking era (default: empty)")
    parser.add_argument("--channels", type=str, default="ee,emu,mumu",
                        help="Channels to be processed, comma-separated (default: ee,emu,mumu)")
    args = parser.parse_args()
    return args

def main():
    """Main function"""
    args = argparser()
    args.metadata = args.metadata.split(",") if args.metadata else []
    args.metadata = {item.split(":")[0]: item.split(":")[1]
                     for item in args.metadata} if args.metadata != [] else {}
    # TODO: Implement systematic handling
    fw_config, processes, _ = utils.initial_loading()

    channels = args.channels.split(",") if args.channels else ["ee", "emu", "mumu"]

    with open(fw_config["fw_dir"]+"/config/ntuples/datasets/Nominal.yml",
              "r", encoding='utf-8') as f:
        datasets = yaml.full_load(f)

    signals = [subproc for proc in fw_config["signals"] for subproc in processes[proc]]

    command_file = ""

    processed_files = []
    for process in datasets:
        print(f"Dataset: {process}\n")
        for era in datasets[process]:
            if args.era and era != args.era:
                continue
            print(f"Era: {era}\n")
            metadata = {"era": era,
                        "process": process,
                        "isData": "run" in process,
                        "isSignal": process in signals,
                        }
            ntuple_dir = fw_config["ntuples_dir"].replace("<era>",era) + "/Nominal"
            minitree_dir = fw_config["minitree_dir"].replace("<era>",era)
            control_hist_dir = fw_config["control_hist_dir"].replace("<era>",era)
            status_dir = fw_config["fw_dir"]+"/selection_status/"
            status_files = os.listdir(status_dir) if os.path.exists(status_dir) else []

            if not os.path.exists(minitree_dir):
                os.makedirs(minitree_dir)
                os.makedirs(minitree_dir+"/Nominal")
            if not os.path.exists(control_hist_dir):
                os.makedirs(control_hist_dir)
                os.makedirs(control_hist_dir+"/Nominal")

            minitree_dir = minitree_dir + "/Nominal"
            control_hist_dir = control_hist_dir + "/Nominal"

            if not os.path.exists(minitree_dir):
                os.makedirs(minitree_dir)
            if not os.path.exists(control_hist_dir):
                os.makedirs(control_hist_dir)

            filenames = os.listdir(ntuple_dir)
            for filename in filenames:
                if filename.endswith(".root") and f"{process}_" in filename and era in filename:
                    if filename.replace(".root", "_status.out") in status_files:
                        with open(f"{status_dir}/{filename.replace('.root', '_status.out')}",
                                "r", encoding='utf-8') as status_file:
                            status_lines = status_file.read()
                            if "SELECTION COMPLETED" in status_lines:
                                print(f"File {filename} already processed. Skipping...")
                                continue

                    for chan in channels:
                        if not os.path.exists(minitree_dir+f"/{chan}"):
                            os.makedirs(minitree_dir+f"/{chan}")
                        if not os.path.exists(control_hist_dir+f"/{chan}"):
                            os.makedirs(control_hist_dir+f"/{chan}")

                    if filename in processed_files:
                        raise ValueError(f"File matches twice {filename}")
                    output_minitree = os.path.join(
                        minitree_dir+"/<chan>",
                        filename.replace("_ntuples","").replace(".root","")
                        )
                    output_histos = os.path.join(
                        control_hist_dir+"/<chan>",
                        filename.replace("_ntuples","_histo").replace(".root","")
                        )
                    command_file += "python src/selection/run_processor.py "
                    command_file += f"'{os.path.join(ntuple_dir,filename)}' "
                    command_file += "--output "
                    command_file += f"'{output_minitree}' "
                    command_file += "--output_histos "
                    command_file += f"'{output_histos}' "
                    command_file += "--metadata "
                    for key, value in metadata.items():
                        command_file += f"{key}:{value},"
                    command_file = command_file[:-1] + " \n"
                    processed_files.append(filename)

    command_file_path = f"{fw_config['fw_dir']}/selection_commands_{args.era}.sh" if args.era\
                        else f"{fw_config['fw_dir']}/selection_commands.sh"
    with open(command_file_path, "w", encoding='utf-8') as cmd_file:
        cmd_file.write(command_file)

    print("Copying minitree configuration...")
    os.system(f"cp -r {fw_config['fw_dir']}/config {minitree_dir}/minitree_configuration")

if __name__ == "__main__":
    main()
