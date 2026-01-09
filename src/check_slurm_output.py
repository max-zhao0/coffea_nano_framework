"""
Script to check the output of Slurm jobs.
"""
import argparse
import os
import utils

def check_command_output(command, fw_config):
    """
    check the output of a command in the command list
    """
    command = command.strip()
    if not command:
        return False

    era = str(command.split("era:")[-1].split(",")[0].strip())
    minitree_dir = fw_config["minitree_dir"].replace("<era>", era[:4])
    output_file = command.split("--output ")[-1].split(" --")[0].strip()+".root"
    output_path = f"{minitree_dir}/{output_file}"
    if not output_file or not output_path:
        print(f"Invalid command: {command}")
        return False

    # Check if the output file in empty_files.txt
    # empty_files_path = f"{fw_config['fw_dir']}/empty_files.txt"
    # if os.path.exists(empty_files_path):
    #     with open(empty_files_path, "r", encoding='utf-8') as f:
    #         empty_files = f.read()
    #     if output_file in empty_files:
    #         print(f"Output file {output_file} is marked as empty. "
    #               f"Skipping command {command_idx+1}.")
    #         return True

    # Check if the output file exists
    if not os.path.exists(output_path):
        # print(f"Output file does not exist: {output_path}")
        # print(f"Command {command_idx+1} failed: {command}")
        return False

    # If all checks pass, return True
    return True

def check_logs_for_empty(fw_config):
    """
    Check the logs for empty output files and update the empty_files.txt record.
    """
    empty_files = []
    failed_logs = []
    log_dir = fw_config["fw_dir"] + "/HammercmsSlurmOut"
    slurm_dir = fw_config["fw_dir"] + "/HammercmsSlurmJobs"
    if not os.path.exists(log_dir):
        print(f"Log directory {log_dir} does not exist. Skipping empty file check.")
        return failed_logs

    empty_files_path = f"{fw_config['fw_dir']}/empty_files.txt"

    for log_file in os.listdir(log_dir):
        if not log_file.endswith(".out"):
            continue

        log_path = os.path.join(log_dir, log_file)
        with open(log_path, "r", encoding='utf-8') as f:
            content = f.read()
            lines = f.readlines()

        if "python src/" not in content:
            print(f"Log file {log_path} does not contain a valid command. Skipping.")
            failed_logs.append(log_path)

        if "CODE-EMPTY-FILE" in content:
            # CODE-EMPTY-FILE indicates the output file was empty and next to it is the output_name
            output_name = content.split("CODE-EMPTY-FILE")[-1].strip()
            empty_files.append(output_name)
            if not os.path.exists(empty_files_path):
                with open(empty_files_path, "w", encoding='utf-8') as ef:
                    ef.write(f"{output_name}\n")
            else:
                with open(empty_files_path, "a", encoding='utf-8') as ef:
                    ef.write(f"{output_name}\n")

        if "Saved final minitree: " not in content:
            print(content)
            try:
                command = lines[2]
                if "python src/" not in command:
                    print(command)
                    raise ValueError("Could not find command in log file.")
                failed_logs.append(command)
            except ValueError:
                print(f"Could not extract command from log file: {log_path}")
                print("Using log name as slurm")
                slurm_job_number = log_file.split("-")[1]
                slurm_job_name = f"SlurmJob_{slurm_job_number}"
                slurm_path = os.path.join(slurm_dir, f"{slurm_job_name}.sh")
                with open(slurm_path, "r", encoding='utf-8') as sf:
                    for line in sf.readlines():
                        if line.startswith("python src/"):
                            failed_logs.append(line.strip())
                            print("Found command in slurm file:", line.strip())
                            break

    if not os.path.exists(empty_files_path):
        with open(empty_files_path, "w", encoding='utf-8') as ef:
            ef.write("\n".join(empty_files) + "\n")
    else:
        with open(empty_files_path, "r", encoding='utf-8') as ef:
            existing_empty_files = ef.read()

        with open(empty_files_path, "a", encoding='utf-8') as ef:
            for ef_name in empty_files:
                if ef_name not in existing_empty_files:
                    ef.write(f"{ef_name}\n")

    return failed_logs

def main():
    """Main function to parse arguments and check command outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("commandlist")
    parser.add_argument("--cluster", type=str, default="Hammer",
                        help='Cluster to submit job to. Default is Hammer.')
    args = parser.parse_args()

    with open(args.commandlist, "r", encoding='utf-8') as commandlistfile:
        all_lines = commandlistfile.readlines()
    commands = [line.strip()
                for line in all_lines if line.strip() and not line.strip().startswith("#")]

    fw_config = utils.parse_main_config()

    failed_logs = check_logs_for_empty(fw_config)
    print(failed_logs)

    failed_commands = []
    for idx, command in enumerate(commands):
        # if not check_command_output(command, fw_config, idx):
        #     failed_commands.append((idx+1, command))
        if command in failed_logs:
            failed_commands.append((idx+1, command))

    if failed_commands:
        print("The following commands failed:")
        for idx, command in failed_commands:
            print(f"Command {idx}: {command}")

        # Create a sbatch script to re-run the failed commands

        fw_dir = fw_config["fw_dir"]
        old_submition_script = fw_dir + f"/Run{args.cluster}cmsSlurm_{args.commandlist}"
        if len(commands) > 4999:
            old_submition_script = fw_dir + f"/Run{args.cluster}cmsSlurm_selection_commands_0.sh"

        initial_txt = ""
        with open(old_submition_script, "r", encoding='utf-8') as f:
            for line in f.readlines():
                if "sbatch" in line:
                    break
                initial_txt += line
        with open(f"Run{args.cluster}cmsSlurm_selection_retryJobs.sh", "w", encoding='utf-8') as f:
            f.write(initial_txt)
            for idx, command in failed_commands:
                f.write(f"sbatch {args.cluster}cmsSlurmJobs/SlurmJob_{idx}.sh\n")
    else:
        print("All commands succeeded.")

if __name__ == "__main__":
    main()
