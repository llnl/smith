#!/usr/bin/env bash

set -uo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'EOF'
Usage: run_sphere_into_corner_response_matrix.sh

Environment variables:
  BUILD_DIR    Existing release build directory.
  OUTPUT_ROOT  Matrix output directory.
  RUN_FILTER   Comma-separated solver_preconditioner entries. Default: all eight.
  RUN_TIMEOUT  Per-entry timeout. Default: 1800s.
  NONLINEAR_MAX_ITERATIONS  Maximum nonlinear iterations per increment. Default: 40000.
EOF
}

if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && ("$1" == "--help" || "$1" == "-h") ]]; then
    usage
    exit 0
  fi
  echo "Unexpected command-line arguments" >&2
  usage >&2
  exit 2
fi

if [[ "$("${repo_dir}/skills/building/scripts/is_compute_node")" != "compute" ]]; then
  echo "This response matrix must run on a compute node." >&2
  exit 2
fi

build_dir="${BUILD_DIR:-${repo_dir}/build-rzwhippet-toss_4_x86_64_ib-llvm@19.1.3-release}"
executable="${build_dir}/examples/challenge_suite"
run_stamp=$(date +%Y%m%d_%H%M%S)
output_root="${OUTPUT_ROOT:-${build_dir}/paper_sphere_into_corner_response_${run_stamp}}"
run_filter="${RUN_FILTER:-}"
run_timeout="${RUN_TIMEOUT:-1800s}"
nonlinear_max_iterations="${NONLINEAR_MAX_ITERATIONS:-40000}"

if ! [[ "${nonlinear_max_iterations}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NONLINEAR_MAX_ITERATIONS must be a positive integer" >&2
  exit 2
fi

solvers=(nls petsc_cp tr tr_subspace)
preconditioners=(HypreAMG HypreJacobi)

solver_name() {
  case "$1" in
    nls) echo NewtonLineSearch ;;
    petsc_cp) echo PetscNewtonCriticalPoint ;;
    tr|tr_subspace) echo TrustRegion ;;
    *) return 1 ;;
  esac
}

subspace_option() {
  case "$1" in
    tr_subspace) echo 2 ;;
    *) echo 0 ;;
  esac
}

max_cg_iterations() {
  case "$1" in
    HypreAMG) echo 600 ;;
    HypreJacobi) echo 60000 ;;
    *) return 1 ;;
  esac
}

entry_selected() {
  local entry="$1"
  local requested_entry
  local -a requested

  if [[ -z "${run_filter}" ]]; then
    return 0
  fi

  IFS=',' read -r -a requested <<< "${run_filter}"
  for requested_entry in "${requested[@]}"; do
    if [[ "${requested_entry}" == "${entry}" ]]; then
      return 0
    fi
  done
  return 1
}

run_entry() {
  local solver="$1"
  local preconditioner="$2"
  local entry="${solver}_${preconditioner}"
  local run_dir="${output_root}/${entry}"
  local max_iterations
  local -a launcher
  local -a command
  local start_seconds
  local end_seconds
  local elapsed_seconds
  local status
  local outcome

  if ! entry_selected "${entry}"; then
    return
  fi

  max_iterations=$(max_cg_iterations "${preconditioner}")
  mkdir -p "${run_dir}"

  if [[ -n "${FLUX_URI:-}" ]]; then
    launcher=(flux run -n 8)
  else
    launcher=(flux start "--test-size=8" flux run -n 8)
  fi

  command=(
    "${launcher[@]}"
    "${executable}"
    "--case=07"
    "--size=small"
    "--mesh-scale=1"
    "--nonlinear-solver=$(solver_name "${solver}")"
    "--linear-solver=CG"
    "--preconditioner=${preconditioner}"
    "--trust-subspace-option=$(subspace_option "${solver}")"
    "--nonlinear-tol=1e-11"
    "--linear-tol=1e-14"
    "--nonlinear-max-iterations=${nonlinear_max_iterations}"
    "--max-cg-iterations=${max_iterations}"
    "--print-level=1"
    "--linear-print-level=0"
    --no-use-bsr-spmv
    --no-assemble-bsr
    --paraview
    --timings
  )

  printf '%q ' "${command[@]}" > "${run_dir}/command.txt"
  printf '\n' >> "${run_dir}/command.txt"
  echo "Running ${entry}"

  start_seconds=$(date +%s)
  set +e
  (
    cd "${run_dir}"
    /usr/bin/time -p timeout --signal=TERM "${run_timeout}" "${command[@]}" 2>&1 | tee run.log
    exit "${PIPESTATUS[0]}"
  )
  status=$?
  set -e
  end_seconds=$(date +%s)
  elapsed_seconds=$((end_seconds - start_seconds))

  if [[ ${status} -eq 0 ]]; then
    outcome=completed
  elif [[ ${status} -eq 124 ]]; then
    outcome=timeout
  else
    outcome=failed
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${solver}" "${preconditioner}" "${outcome}" "${status}" "${elapsed_seconds}" \
    >> "${output_root}/status.tsv"
}

if [[ ! -x "${executable}" ]]; then
  echo "Missing executable: ${executable}" >&2
  echo "Build it from the prescribed release build directory before running this script." >&2
  exit 2
fi

mkdir -p "${output_root}"
printf 'solver\tpreconditioner\toutcome\texit_code\telapsed_seconds\n' > "${output_root}/status.tsv"
cat > "${output_root}/effective_settings.txt" <<EOF
case=07/sphere_into_corner
mpi_ranks=8
element_order=1
time_steps=32
time_step_size=0.0625
final_time=2.0
patch_traction_rate=0.019
final_patch_traction=0.038
warm_start=off
nonlinear_tolerance=1e-11
linear_tolerance=1e-14
base_nonlinear_max_iterations=10000
effective_nonlinear_max_iterations=${nonlinear_max_iterations}
max_cg_iterations_hypre_amg=600
max_cg_iterations_hypre_jacobi=60000
use_bsr_spmv=off
assemble_bsr=off
paraview=on
EOF
stat -c $'executable=%n\nmtime=%y\nbytes=%s' "${executable}" > "${output_root}/executable.txt"
sha256sum "${executable}" >> "${output_root}/executable.txt"

for solver in "${solvers[@]}"; do
  for preconditioner in "${preconditioners[@]}"; do
    run_entry "${solver}" "${preconditioner}"
  done
done

mkdir -p "${build_dir}/paper_logs"
printf '%s\n' "${output_root}" > "${build_dir}/paper_logs/latest_sphere_into_corner_response_matrix.txt"
echo "Response matrix written to ${output_root}"
