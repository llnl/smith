#!/usr/bin/env bash

set -uo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'EOF'
Usage: run_euler_cylinder_response_matrix.sh

Environment variables:
  BUILD_DIR         Existing release build directory.
  OUTPUT_ROOT       Matrix output directory.
  CASE_FILTER       Comma-separated euler,cylinder selection. Default: both.
  EULER_TIMEOUT     Per-entry Euler timeout. Default: 3600s.
  CYLINDER_TIMEOUT  Per-entry cylinder timeout. Default: 1800s.
  CYLINDER_RANKS    MPI ranks per cylinder entry. Default: 16.
  CYLINDER_PARAVIEW Set to 1 to write paired cylinder field output. Default: 1.
  CYLINDER_EIGENPAIR
                    Set to 1 to compute the accepted-state lowest eigenpair. Default: 0.
  CYLINDER_EIGENVECTOR
                    Set to 1 to write the lowest eigenvector. Default: 0.
  RUN_FILTER        Comma-separated solver_preconditioner entries. Default: all eight.
  EIGENVALUE_COUNT  Number of requested cylinder final-state modes. Default: 1.
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
output_root="${OUTPUT_ROOT:-${build_dir}/paper_euler_cylinder_response_${run_stamp}}"
case_filter="${CASE_FILTER:-euler,cylinder}"
euler_timeout="${EULER_TIMEOUT:-3600s}"
cylinder_timeout="${CYLINDER_TIMEOUT:-1800s}"
cylinder_ranks="${CYLINDER_RANKS:-16}"
cylinder_paraview="${CYLINDER_PARAVIEW:-1}"
cylinder_eigenpair="${CYLINDER_EIGENPAIR:-0}"
cylinder_eigenvector="${CYLINDER_EIGENVECTOR:-0}"
run_filter="${RUN_FILTER:-}"
eigenvalue_count="${EIGENVALUE_COUNT:-1}"

if ! [[ "${eigenvalue_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EIGENVALUE_COUNT must be a positive integer" >&2
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

case_selected() {
  local requested
  IFS=',' read -r -a requested <<< "${case_filter}"
  for entry in "${requested[@]}"; do
    if [[ "$entry" == "$1" ]]; then return 0; fi
  done
  return 1
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
  local case_name="$1"
  local ranks="$2"
  local timeout_duration="$3"
  local solver="$4"
  local preconditioner="$5"
  shift 5

  local run_dir="${output_root}/${case_name}/${solver}_${preconditioner}"
  local max_iterations
  local -a launcher
  local -a command
  local status

  if ! entry_selected "${solver}_${preconditioner}"; then
    return
  fi

  max_iterations=$(max_cg_iterations "${preconditioner}")
  mkdir -p "${run_dir}"

  if [[ -n "${FLUX_URI:-}" ]]; then
    launcher=(flux run -n "${ranks}")
  else
    launcher=(flux start "--test-size=${ranks}" flux run -n "${ranks}")
  fi

  command=(
    "${launcher[@]}"
    "${executable}"
    "$@"
    "--nonlinear-solver=$(solver_name "${solver}")"
    "--linear-solver=CG"
    "--preconditioner=${preconditioner}"
    "--trust-subspace-option=$(subspace_option "${solver}")"
    "--nonlinear-tol=1e-11"
    "--linear-tol=1e-14"
    "--max-cg-iterations=${max_iterations}"
    "--print-level=1"
    "--linear-print-level=0"
    --no-use-bsr-spmv
    --no-assemble-bsr
    --timings
  )

  printf '%q ' "${command[@]}" > "${run_dir}/command.txt"
  printf '\n' >> "${run_dir}/command.txt"
  echo "Running ${case_name} ${solver} ${preconditioner}"

  set +e
  (
    cd "${run_dir}"
    /usr/bin/time -p timeout --signal=TERM "${timeout_duration}" "${command[@]}" 2>&1 | tee run.log
    exit "${PIPESTATUS[0]}"
  )
  status=$?
  set -e

  printf '%s\t%s\t%s\t%s\n' "${case_name}" "${solver}" "${preconditioner}" "${status}" >> "${output_root}/status.tsv"
}

if [[ ! -x "${executable}" ]]; then
  echo "Missing executable: ${executable}" >&2
  echo "Build it from the prescribed release build directory before running this script." >&2
  exit 2
fi

mkdir -p "${output_root}"
printf 'case\tsolver\tpreconditioner\texit_code\n' > "${output_root}/status.tsv"

if case_selected euler; then
  for solver in "${solvers[@]}"; do
    for preconditioner in "${preconditioners[@]}"; do
      run_entry euler 4 "${euler_timeout}" "${solver}" "${preconditioner}" \
        --case=01 \
        --order=2 \
        --size=small \
        --mesh-scale=1 \
        --euler-load=0.0022916666666666667 \
        --euler-refined-start-traction=0.00225 \
        --euler-selector-traction=-1e-10 \
        --euler-coarse-load-steps=30 \
        --euler-refined-load-steps=5 \
        --paraview
    done
  done
fi

if case_selected cylinder; then
  cylinder_diagnostic_args=()
  cylinder_output_args=(--paraview)
  if [[ "${cylinder_paraview}" == "0" ]]; then
    cylinder_output_args=(--no-paraview)
  elif [[ "${cylinder_paraview}" != "1" ]]; then
    echo "CYLINDER_PARAVIEW must be 0 or 1" >&2
    exit 2
  fi
  if [[ "${cylinder_eigenpair}" == "1" ]]; then
    cylinder_diagnostic_args+=(--final-state-eigenpair "--eigenvalue-count=${eigenvalue_count}")
    if [[ "${cylinder_eigenvector}" == "1" ]]; then
      cylinder_diagnostic_args+=(--write-eigenvector)
    fi
  elif [[ "${cylinder_eigenvector}" == "1" ]]; then
    echo "CYLINDER_EIGENVECTOR=1 requires CYLINDER_EIGENPAIR=1" >&2
    exit 2
  fi

  for solver in "${solvers[@]}"; do
    for preconditioner in "${preconditioners[@]}"; do
      run_entry cylinder "${cylinder_ranks}" "${cylinder_timeout}" "${solver}" "${preconditioner}" \
        --case=03 \
        --size=small \
        --mesh-scale=1 \
        "${cylinder_output_args[@]}" \
        "${cylinder_diagnostic_args[@]}"
    done
  done
fi

printf '%s\n' "${output_root}" > "${build_dir}/paper_logs/latest_euler_cylinder_response_matrix.txt"
echo "Response matrix written to ${output_root}"
