#!/bin/bash
# Batch cancel SLURM jobs
# Usage: ./batch_cancel.sh [OPTIONS]

show_help() {
    cat << EOF
Batch cancel SLURM jobs

Usage:
    ./batch_cancel.sh [OPTIONS]

Options:
    -r START END        Cancel jobs in range (e.g., -r 12345 12350)
    -n PATTERN          Cancel jobs matching name pattern (e.g., -n shinobi)
    -c                  Cancel currently RUNNING jobs only
    -a                  Cancel ALL your pending/running jobs
    -s STATE            Cancel jobs in specific state (e.g., -s PENDING)
    -l                  List your jobs (no cancellation)
    -h                  Show this help message

Examples:
    ./batch_cancel.sh -r 34343744 34343768    # Cancel job range
    ./batch_cancel.sh -n shinobi              # Cancel all jobs with "shinobi" in name
    ./batch_cancel.sh -n run-level            # Cancel all run-level jobs
    ./batch_cancel.sh -c                      # Cancel currently running jobs
    ./batch_cancel.sh -s PENDING              # Cancel all pending jobs
    ./batch_cancel.sh -a                      # Cancel ALL your jobs
    ./batch_cancel.sh -l                      # Just list your jobs

EOF
}

# Default values
MODE=""
START=""
END=""
PATTERN=""
STATE=""

# Parse arguments
while getopts "r:n:s:calh" opt; do
    case $opt in
        r)
            MODE="range"
            START=$OPTARG
            END=${!OPTIND}
            OPTIND=$((OPTIND + 1))
            ;;
        n)
            MODE="pattern"
            PATTERN=$OPTARG
            ;;
        s)
            MODE="state"
            STATE=$OPTARG
            ;;
        c)
            MODE="active"
            ;;
        a)
            MODE="all"
            ;;
        l)
            MODE="list"
            ;;
        h)
            show_help
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            exit 1
            ;;
    esac
done

# Execute based on mode
case $MODE in
    list)
        echo "Your current jobs:"
        squeue -u $USER --format="%.10i %.40j %.8T %.10M %.6D"
        ;;

    range)
        if [[ -z "$START" || -z "$END" ]]; then
            echo "Error: Range requires START and END job IDs"
            echo "Usage: ./batch_cancel.sh -r START END"
            exit 1
        fi
        echo "Canceling jobs from $START to $END..."
        for ID in $(seq $START $END); do
            scancel $ID 2>/dev/null && echo "Cancelled: $ID" || echo "Failed/Not found: $ID"
        done
        ;;

    pattern)
        if [[ -z "$PATTERN" ]]; then
            echo "Error: Pattern mode requires a job name pattern"
            echo "Usage: ./batch_cancel.sh -n PATTERN"
            exit 1
        fi
        echo "Canceling jobs matching pattern: $PATTERN"
        JOB_IDS=$(squeue -u $USER --format="%i %j" | grep "$PATTERN" | awk '{print $1}')

        if [[ -z "$JOB_IDS" ]]; then
            echo "No jobs found matching pattern: $PATTERN"
            exit 0
        fi

        echo "Found jobs:"
        squeue -u $USER --format="%.10i %.40j %.8T" | grep "$PATTERN"
        echo ""
        read -p "Cancel these jobs? (y/N): " confirm

        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            for ID in $JOB_IDS; do
                scancel $ID && echo "Cancelled: $ID"
            done
        else
            echo "Cancelled. No jobs were terminated."
        fi
        ;;

    state)
        if [[ -z "$STATE" ]]; then
            echo "Error: State mode requires a state (PENDING, RUNNING, etc.)"
            echo "Usage: ./batch_cancel.sh -s STATE"
            exit 1
        fi
        echo "Canceling jobs in state: $STATE"
        JOB_IDS=$(squeue -u $USER --format="%i %T" | grep "$STATE" | awk '{print $1}')

        if [[ -z "$JOB_IDS" ]]; then
            echo "No jobs found in state: $STATE"
            exit 0
        fi

        echo "Found jobs:"
        squeue -u $USER --format="%.10i %.40j %.8T" --state=$STATE
        echo ""
        read -p "Cancel these jobs? (y/N): " confirm

        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            for ID in $JOB_IDS; do
                scancel $ID && echo "Cancelled: $ID"
            done
        else
            echo "Cancelled. No jobs were terminated."
        fi
        ;;

    active)
        echo "Canceling currently RUNNING jobs..."
        JOB_IDS=$(squeue -u $USER --format="%i %T" | grep "RUNNING" | awk '{print $1}')

        if [[ -z "$JOB_IDS" ]]; then
            echo "No currently running jobs found."
            exit 0
        fi

        echo "Found running jobs:"
        squeue -u $USER --format="%.10i %.40j %.8T %.10M %.6D" --state=RUNNING
        echo ""
        read -p "Cancel these running jobs? (y/N): " confirm

        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            for ID in $JOB_IDS; do
                scancel $ID && echo "Cancelled: $ID"
            done
        else
            echo "Cancelled. No jobs were terminated."
        fi
        ;;

    all)
        echo "WARNING: This will cancel ALL your pending/running jobs!"
        squeue -u $USER --format="%.10i %.40j %.8T"
        echo ""
        read -p "Are you absolutely sure? (yes/N): " confirm

        if [[ $confirm == "yes" ]]; then
            scancel -u $USER
            echo "All jobs cancelled."
        else
            echo "Cancelled. No jobs were terminated."
        fi
        ;;

    "")
        echo "No option specified. Use -h for help."
        echo ""
        echo "Quick commands:"
        echo "  -l              List your jobs"
        echo "  -n PATTERN      Cancel jobs by name"
        echo "  -r START END    Cancel job range"
        echo "  -h              Full help"
        exit 1
        ;;
esac
