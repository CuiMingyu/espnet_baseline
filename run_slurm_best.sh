set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"

asr_config=conf/mycui/train_slurm_best.yaml
inference_config=conf/example/decode/default.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors=""

bash /home/cuimingyu/project/espnet_transducer_streaming/asr_slurm_best.sh \
    --audio_format flac.ark \
    --lang en \
    --ngpu 4 \
    --nj 128 \
    --inference_nj 64 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" "$@" \
    --local_score_opts "--inference_config ${inference_config}" \
