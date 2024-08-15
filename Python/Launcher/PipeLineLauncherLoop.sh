#!/bin/bash
#SBATCH --job-name=FullPipeLine
#SBATCH -p WS-CPU1,WS-CPU2,Serveurs-CPU  #MPI-CPU,5810,2080CPU,1080CPU
#SBATCH --output=output_sbatch/PipeLine.out
#SBATCH --error=output_sbatch/PipeLine.err

simu_number=$1

# Launch Gate_Launcher.sh
#job1_id=$(sbatch --export=ALL,simu_number=$simu_number /homes/ymellak/Direct3G_f/Python/Launcher/Gate_Launcher.sh | awk '{print $4}')
#echo "Launched Gate_Launcher.sh with Job ID $job1_id, simu_number=$simu_number"

# Launch Gate_Launcher.sh
job1_id=$(sbatch --export=ALL,simu_number=$simu_number /homes/ymellak/Direct3G_f/Python/Launcher/Gate_Launcher.sh | awk '{print $4}')
echo "Launched Gate_Launcher.sh with Job ID $job1_id, simu_number=$simu_number"

# Function to convert time to seconds
time_to_seconds() {
    local time_str=$1
    local colon_count=$(echo "$time_str" | tr -cd ':' | wc -c)
    
    if [ $colon_count -eq 1 ]; then
        # MM:SS format
        echo "$time_str" | awk -F: '{ print ($1 * 60) + $2 }'
    elif [ $colon_count -eq 2 ]; then
        # HH:MM:SS format
        echo "$time_str" | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }'
    else
        echo "Invalid time format: $time_str" >&2
        echo 0
    fi
}

# Check job status and runtime
max_runtime=600  # 18 minutes in seconds
while true; do
    # Check if the job is still running
    job_info=$(squeue -u ymellak | grep "^${job1_id}")
    if [ -z "$job_info" ]; then
        echo "Job $job1_id has completed."
        break
    fi

    # Get all sub-processes (array tasks) of the job
    sub_processes=$(squeue -u ymellak | awk -v job_id=$job1_id '$1 ~ job_id"_" {print $1}')

    for sub_process in $sub_processes; do
        # Extract the runtime for each sub-process
        runtime=$(squeue -u ymellak | awk -v sp=$sub_process '$1 == sp {print $6}')
        runtime_seconds=$(time_to_seconds "$runtime")

        # Check if the sub-process has exceeded the maximum runtime
        if [ $runtime_seconds -gt $max_runtime ]; then
            echo "Sub-process $sub_process of Job $job1_id has exceeded the maximum runtime of $max_runtime seconds. Cancelling this sub-process."
            scancel $sub_process
        fi
    done

    # Check if all sub-processes have been cancelled
    remaining_sub_processes=$(squeue -u ymellak | awk -v job_id=$job1_id '$1 ~ job_id"_" {print $1}' | wc -l)
    if [ $remaining_sub_processes -eq 0 ]; then
        echo "All sub-processes of Job $job1_id have been cancelled."
        break
    fi

    # Wait for 10 seconds before checking again
    sleep 10
done

# Continue with the rest of the pipeline, using --dependency only if job1_id is not CANCELLED
if [ "$job1_id" != "CANCELLED" ]; then
    dependency_option="--dependency=afterany:$job1_id"
else
    dependency_option=""
fi

# Launch BuildDetectors.sh
job2_id=$(sbatch --kill-on-invalid-dep=no $dependency_option --export=ALL,simu_number=$simu_number /homes/ymellak/Direct3G_f/Python/Launcher/BuildDetectors.sh | awk '{print $4}')
echo "Launched BuildDetectors.sh with Job ID $job2_id, dependent on Job ID $job1_id, simu_number=$simu_number"

# Monitor and handle Job2 similar to Job1

# Check job status and runtime for Job2
while true; do
    # Check if the job is still running
    job_info=$(squeue -u ymellak | grep "^${job2_id}")
    if [ -z "$job_info" ]; then
        echo "Job $job2_id has completed."
        break
    fi

    # Get all sub-processes (array tasks) of the job
    sub_processes=$(squeue -u ymellak | awk -v job_id=$job2_id '$1 ~ job_id"_" {print $1}')

    for sub_process in $sub_processes; do
        # Extract the runtime for each sub-process
        runtime=$(squeue -u ymellak | awk -v sp=$sub_process '$1 == sp {print $6}')
        runtime_seconds=$(time_to_seconds "$runtime")

        # Check if the sub-process has exceeded the maximum runtime
        if [ $runtime_seconds -gt $max_runtime ]; then
            echo "Sub-process $sub_process of Job $job2_id has exceeded the maximum runtime of $max_runtime seconds. Cancelling this sub-process."
            scancel $sub_process
        fi
    done

    # Check if all sub-processes have been cancelled
    remaining_sub_processes=$(squeue -u ymellak | awk -v job_id=$job2_id '$1 ~ job_id"_" {print $1}' | wc -l)
    if [ $remaining_sub_processes -eq 0 ]; then
        echo "All sub-processes of Job $job2_id have been cancelled."
        break
    fi

    # Wait for 10 seconds before checking again
    sleep 10
done

# Continue with the rest of the pipeline, using --dependency only if job1_id is not CANCELLED
if [ "$job2_id" != "CANCELLED" ]; then
    dependency_option="--dependency=afterany:$job2_id"
else
    dependency_option=""
fi

# Launch BuildEmissionSite.sh after BuildDetectors.sh completes
job3_id=$(sbatch --kill-on-invalid-dep=no $dependency_option --export=ALL,simu_number=$simu_number /homes/ymellak/Direct3G_f/Python/Launcher/BuildEmissionSite.sh | awk '{print $4}')
echo "Launched BuildEmissionSite.sh with Job ID $job3_id, dependent on Job ID $job2_id, simu_number=$simu_number"

# Launch MergeImages.sh with Workon=EmissionImages after BuildEmissionSite.sh completes
Workon="EmissionImages"
job4_id=$(sbatch --kill-on-invalid-dep=no --dependency=afterany:$job3_id --export=ALL,simu_number=$simu_number,Workon=$Workon /homes/ymellak/Direct3G_f/Python/Launcher/MergeImages.sh | awk '{print $4}')
echo "Launched MergeImages.sh with Job ID $job4_id, dependent on Job ID $job3_id, simu_number=$simu_number, Workon=$Workon"

# Launch Estimate_w_U.sh after MergeImages.sh completes
job5_id=$(sbatch --kill-on-invalid-dep=no --dependency=afterany:$job3_id --export=ALL,simu_number=$simu_number /homes/ymellak/Direct3G_f/Python/Launcher/Estimate_w_U.sh | awk '{print $4}')
echo "Launched Estimate_w_U.sh with Job ID $job5_id, dependent on Job ID $job4_id, simu_number=$simu_number"

# Launch BPImages.sh after Estimate_w_U.sh completes
job6_id=$(sbatch --kill-on-invalid-dep=no --dependency=afterany:$job5_id --export=ALL,simu_number=$simu_number /homes/ymellak/Direct3G_f/Python/Launcher/BPImages.sh | awk '{print $4}')
echo "Launched BPImages.sh with Job ID $job6_id, dependent on Job ID $job5_id, simu_number=$simu_number"

# Launch MergeImages.sh with Workon=BPImages after BPImages.sh completes
Workon="BPImages"
job7_id=$(sbatch --kill-on-invalid-dep=no --dependency=afterany:$job6_id --export=ALL,simu_number=$simu_number,Workon=$Workon /homes/ymellak/Direct3G_f/Python/Launcher/MergeImages.sh | awk '{print $4}')
echo "Launched MergeImages.sh with Job ID $job7_id, dependent on Job ID $job6_id, simu_number=$simu_number, Workon=$Workon"
