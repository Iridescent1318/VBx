export KALDI_ROOT='/home/liaozty20/kaldi'
ABS_DIR="/home/liaozty20"

mkdir -p ami_exp_dir
mkdir -p ami_xvec_dir

INSTRUCTION="diarization"
METHOD="AHC+VB"

echo "test from nohup"

exp_dir="${ABS_DIR}/ami_exp_dir" # output experiment directory
xvec_dir="${ABS_DIR}/ami_xvec_dir" # output xvectors directory
WAV_DIR="${ABS_DIR}/amicorpus" # wav files directory
FILE_LIST="${ABS_DIR}/AMI-diarization-setup/lists/test.meetings.txt" # txt list of files to process
LAB_DIR="${ABS_DIR}/AMI-diarization-setup/only_words/labs/test" # lab files directory with VAD segments
RTTM_DIR="${ABS_DIR}/AMI-diarization-setup/only_words/rttms/test" # reference rttm files directory
nohup ./AMI_run.sh $INSTRUCTION $METHOD $exp_dir $xvec_dir $WAV_DIR $FILE_LIST $LAB_DIR $RTTM_DIR > AMI_run_log.log 2>&1 &