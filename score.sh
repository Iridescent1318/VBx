
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

for audio in $(ls /home/liaozty20/amicorpus)
do
    filename=$(echo "${audio}" | cut -f 1 -d '.')
    echo "Scoring Starts: "${filename}""
    # check if there is ground truth .rttm file
    REFDIR=/home/liaozty20/AMI-diarization-setup/only_words/rttms/test/${filename}.rttm
    SYSDIR=/home/liaozty20/VBx/ami_exp/${filename}.rttm
    if [ -f $REFDIR ]
    then
        # run dscore
        # forgiving
        echo "Forgiving mode for ${filename}"
        python dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.25 --ignore_overlaps
        # fair
        echo "Fair mode for ${filename}"
        python dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.25
        # full
        echo "Full mode for ${filename}"
        python dscore/score.py -r $REFDIR -s $SYSDIR --collar 0.0
    fi
    echo "Scoring Ends: "${filename}""
done