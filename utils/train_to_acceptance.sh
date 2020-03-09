#!/bin/bash
# bash train_to_acceptance.sh ./runcards/train.yml ./runcards/short_sample.yml

# --- Exit gracefully --- #
exit_gracefully () {
    echo "Exiting..."
    echo "Total epochs completed: $epochs"
    echo "Total train time: $((train_time / 60)) mins"
}
files_not_found () {
    echo "Error: directory $run_id exists but no previous training data was found."
    echo "Please delete/rename $run_id before running this script again."
    exit
}
trap exit_gracefully EXIT
trap "echo ' Aborting!'; exit;" SIGINT

# --- Parameters to be set by user --- #
target_acceptance=0.99
n_sample=5

# --- env variables --- #
# Runcards from command line args
train_runcard=$1
sample_runcard=$2

run_id=$(grep "training_output:" $sample_runcard | awk '{print $2}')
log_file=$run_id/training_log.out
data_file=$run_id/training_data.out
lr_file=$run_id/learning_rate.out
tmp_file=$run_id/tmp

epochs_iter=$(grep "epochs:" $train_runcard | awk '{print $2}')

IFS="
"

# --- Function definitions --- #
# Write to data file
write_data () {
    loss=$(grep "Final loss:" "$tmp_file" | awk '{print $3}')

    arr=$(grep "Acceptance:" "$tmp_file" | awk '{print $2}')
    sm=$(echo $(for a in ${arr[@]}; do echo -n "$a+"; done; echo -n 0) | bc -l)
    acceptance=$(echo "$sm / $n_sample" | bc -l)
    sm=$(echo $(for a in ${arr[@]}; do echo -n "($a - $acceptance)^2+"; done; echo -n 0) | bc -l)
    std_acceptance=$(echo "sqrt($sm / ($n_sample-1))" | bc -l)

    arr=$(grep "Integrated autocorrelation time:" "$tmp_file" | awk '{print $4}')
    sm=$(echo $(for a in ${arr[@]}; do echo -n "$a+"; done; echo -n 0) | bc -l)
    tauint=$(echo "$sm / $n_sample" | bc -l)
    sm=$(echo $(for a in ${arr[@]}; do echo -n "($a-$tauint)^2+"; done; echo -n 0) | bc -l)
    std_tauint=$(echo "sqrt($sm / ($n_sample-1))" | bc -l)

    echo "$epochs $train_time $loss $acceptance $std_acceptance $tauint $std_tauint" >> "$data_file"
}

# Retrieve learning rate history and write to file
write_lr () {
    rows=$(grep "Epoch" "$tmp_file" | awk '{print $2 $10}')
    if [ ! -u "$rows" ]; then
        for i in ${rows[@]}; do
            ep=${i%:*}
            lr=${i#*:}

            fr=$(echo "1 - $ep / $epochs_iter" | bc -l)
            ep=$(echo "($epochs - $fr * $epochs_iter)" | bc -l)
            tt=$(echo "($train_time - $fr * ($end_time-$start_time))" | bc -l)
            echo "$ep $tt ${lr%?}" >> "$lr_file"
        done
    fi
}

write_log () {
    cat $tmp_file >> $log_file
}
write_delimiter () {
    echo "<<<<<<<<< E-P-O-C-H-S $((epochs-epochs_iter)) - $epochs >>>>>>>>>" >> $log_file
}

# --- Initialise from existing data or start from scratch --- #
if [ -d $run_id ]; then
    [ ! -f $run_id/training_data.out ] && files_not_found
    epochs=$( tail -1 "${run_id}/training_data.out" | awk '{print $1}' )
    train_time=$( tail -1 "${run_id}/training_data.out" | awk '{print $2}' )
    acceptance=$( tail -1 "${run_id}/training_data.out" | awk '{print $4}' )
    echo "
      #################################################################
      #####                         NEW RUN                       #####
      #################################################################
     " >> $log_file
else
    tmp_file=$run_id.tmp
    epochs=0
    acceptance=0
    # Run an iteration without attempting sampling
    start_time=$(date +%s)
    anvil-train $train_runcard -o $run_id > $tmp_file || exit
    end_time=$(date +%s)
    train_time=$((end_time-start_time))
    ((epochs+=epochs_iter))
    
    write_delimiter
    write_log
    write_lr
    
    rm $tmp_file
    tmp_file=$run_id/tmp
fi

# --- Loop until target acceptance achieved --- #
while (( $(echo "$acceptance < $target_acceptance" | bc -l) ))
do
    # Training
    start_time=$(date +%s)
    anvil-train $run_id -r -1 > $tmp_file || exit # overwrite
    end_time=$(date +%s)
    ((train_time+=(end_time-start_time) ))

    # Run sampling n_sample times
    for s in `seq 1 $n_sample`
    do
        anvil-sample $sample_runcard >> $tmp_file || exit
    done
    ((epochs+=epochs_iter))

    # Retrieve current state and write to log file
    write_delimiter
    write_log
    write_data
    write_lr

done

echo "Reached target acceptance"

