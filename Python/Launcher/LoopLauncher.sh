#!/bin/bash
#SBATCH --job-name=Checker
#SBATCH -p WS-CPU1,WS-CPU2,Serveurs-CPU
#SBATCH --output=output_sbatch/Checker.out
#SBATCH --error=output_sbatch/Checker.err

# Define the username
USER="ymellak"

# Loop through simu_number from 10 to 15
for simu_number in {999..999}
do
    while true
    do
        # Check if there are any running jobs other than this checker job
        # Use -h to remove the header, and count lines that don't match "Checker"
        #job_count=$(squeue -h -u $USER | grep -v "Checker" | wc -l)
        
        # on the specified hardware partitions
        job_count=$(squeue -h -u $USER -p WS-CPU1,WS-CPU2,Serveurs-CPU | grep -vE "Checker|Merge" | wc -l)
        
        # If no other jobs are running, break the loop and launch the next simulation
        if [ $job_count -eq 0 ]; then
            echo "Launching simulation for simu_number=$simu_number"
            bash /homes/ymellak/Direct3G_f/Python/Launcher/PipeLineLauncherLoop.sh $simu_number
            
            break
        
        #else
            #echo "Waiting for other jobs to complete. Current job count: $job_count"
        fi
        
        # Wait for some time before checking again
        sleep 1
    done
done
