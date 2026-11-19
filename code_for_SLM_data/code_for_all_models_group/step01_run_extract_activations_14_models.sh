#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${SLM_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ROOT="${SLM_CONDA_ROOT:-${HOME}/anaconda3/envs}"
cd "$PROJECT_ROOT"

MODEL_KEYS=(
  baichuan_audio_base
  baichuan_audio_instruct
  covo_audio_chat
  firered_audio
  freeze_omni
  glm_4_voice
  kimi_audio_7b_instruct
  minicpm_o_2_6
  minicpm_o_4_5
  qwen_audio
  qwen_audio_chat
  qwen2_audio
  qwen2_audio_instruct
  qwen2_5_omni
)

declare -A ENVIRONMENTS=(
  [baichuan_audio_base]="py310_SLM"
  [baichuan_audio_instruct]="py310_SLM"
  [covo_audio_chat]="py310_covo"
  [firered_audio]="py310_fireredaudio"
  [freeze_omni]="py310_SLM"
  [glm_4_voice]="py310_SLM"
  [kimi_audio_7b_instruct]="py310_kimi"
  [minicpm_o_2_6]="py310_SLM"
  [minicpm_o_4_5]="py310_SLM"
  [qwen_audio]="py310_SLM"
  [qwen_audio_chat]="py310_SLM"
  [qwen2_audio]="py310_SLM"
  [qwen2_audio_instruct]="py310_SLM"
  [qwen2_5_omni]="py310_SLM"
)

declare -A SCRIPTS=(
  [baichuan_audio_base]="code_for_Baichuan-Audio-Base/01_extract_activation/extract_Baichuan_Audio_Base_activations_revised.py"
  [baichuan_audio_instruct]="code_for_Baichuan-Audio-Instruct/01_extract_activation/extract_Baichuan_Audio_Instruct_activations_revised.py"
  [covo_audio_chat]="code_for_Covo-Audio-Chat/01_extract_activation/extract_Covo_Audio_Chat_activations_revised.py"
  [firered_audio]="code_for_FireRedAudio/01_extract_activation/extract_FireRedAudio_activations_revised.py"
  [freeze_omni]="code_for_Freeze-Omni/01_extract_activation/extract_Freeze_Omni_activations_revised.py"
  [glm_4_voice]="code_for_GLM-4-Voice/01_extract_activation/extract_GLM_4_Voice_activations_revised.py"
  [kimi_audio_7b_instruct]="code_for_Kimi-Audio-7B-Instruct/01_extract_activation/extract_Kimi_Audio_7B_Instruct_activations_revised.py"
  [minicpm_o_2_6]="code_for_MiniCPM-o 2.6/01_extract_activation/extract_MiniCPM_o_2_6_activations_revised.py"
  [minicpm_o_4_5]="code_for_MiniCPM-o 4.5/01_extract_activation/extract_MiniCPM_o_4_5_activations_revised.py"
  [qwen_audio]="code_for_Qwen-Audio/01_extract_activation/extract_Qwen_Audio_activations_revised.py"
  [qwen_audio_chat]="code_for_Qwen-Audio-Chat/01_extract_activation/extract_Qwen_Audio_Chat_activations_revised.py"
  [qwen2_audio]="code_for_Qwen2-Audio/01_extract_activation/extract_qwen2_audio_activations_revised.py"
  [qwen2_audio_instruct]="code_for_Qwen2-Audio-Instruct/01_extract_activation/extract_Qwen2_Audio_Instruct_activations_revised.py"
  [qwen2_5_omni]="code_for_Qwen2.5-Omni/01_extract_activation/extract_Qwen2_5_Omni_activations_revised.py"
)

run_one() {
  local key="$1"
  local environment="${ENVIRONMENTS[$key]:-}"
  local script="${SCRIPTS[$key]:-}"
  if [[ -z "$environment" || -z "$script" ]]; then
    echo "Unknown model key: $key" >&2
    echo "Valid keys: ${MODEL_KEYS[*]}" >&2
    return 2
  fi

  local python="$CONDA_ROOT/$environment/bin/python"
  local log="${script%/*}/${key}_extraction.log"
  [[ -x "$python" ]] || { echo "Missing Python: $python" >&2; return 3; }
  [[ -f "$script" ]] || { echo "Missing script: $PROJECT_ROOT/$script" >&2; return 4; }

  echo "[$(date '+%F %T')] START $key ($environment)"
  "$python" "$script" 2>&1 | tee "$log"
  echo "[$(date '+%F %T')] DONE  $key"
}

if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' "${MODEL_KEYS[@]}"
  exit 0
fi

if (( $# == 0 )); then
  targets=("${MODEL_KEYS[@]}")
else
  targets=("$@")
fi
for key in "${targets[@]}"; do
  run_one "$key"
done
