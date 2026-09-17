#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
validator="${repo_dir}/scripts/validate_contact_arch_response.py"

usage() {
  cat <<'EOF'
Usage: run_contact_arch_response_matrix.sh

Environment variables:
  BUILD_DIR    Existing release build directory.
  OUTPUT_ROOT  Matrix output directory.
  ARCHIVE_ROOT Compact archive parent. Default: buckling_solver_paper/results/contact_arch.
  RUN_FILTER   Comma-separated solver_preconditioner entries. Default: all eight.
  RUN_TIMEOUT  Per-entry timeout. Default: 1800s.
  NONLINEAR_MAX_ITERATIONS  Maximum nonlinear iterations per increment. Default: 10000.
  PRINT_LEVEL  Nonlinear solver output level. Default: 1.
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
output_root="${OUTPUT_ROOT:-${build_dir}/paper_contact_arch_response_${run_stamp}}"
archive_root="${ARCHIVE_ROOT:-/usr/workspace/tupek2/dev/buckling_solver_paper/results/contact_arch}"
archive_dir="${archive_root}/solver_response_${run_stamp}"
run_filter="${RUN_FILTER:-}"
run_timeout="${RUN_TIMEOUT:-1800s}"
nonlinear_max_iterations="${NONLINEAR_MAX_ITERATIONS:-10000}"
print_level="${PRINT_LEVEL:-1}"

if ! [[ "${nonlinear_max_iterations}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NONLINEAR_MAX_ITERATIONS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${print_level}" =~ ^[0-9]+$ ]]; then
  echo "PRINT_LEVEL must be a nonnegative integer" >&2
  exit 2
fi

solvers=(petsc_cp tr tr_subspace nls)
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
    launcher=(flux start --test-size=8 flux run -n 8)
  fi

  command=(
    "${launcher[@]}"
    "${executable}"
    --case=06
    --size=small
    --mesh-scale=1
    --nonlinear-solver="$(solver_name "${solver}")"
    --linear-solver=CG
    --preconditioner="${preconditioner}"
    --trust-subspace-option="$(subspace_option "${solver}")"
    --nonlinear-tol=1e-11
    --linear-tol=1e-14
    --nonlinear-max-iterations="${nonlinear_max_iterations}"
    --max-cg-iterations="${max_iterations}"
    --print-level="${print_level}"
    --linear-print-level=0
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
if [[ ! -x "${validator}" ]]; then
  echo "Missing validator: ${validator}" >&2
  exit 2
fi
set +e
make -q -C "${build_dir}" challenge_suite/fast >/dev/null 2>&1
build_query_status=$?
set -e
if [[ ${build_query_status} -eq 1 ]]; then
  echo "The challenge_suite target is out of date." >&2
  echo "Build it from the prescribed release build directory before running this script." >&2
  exit 2
elif [[ ${build_query_status} -ne 0 ]]; then
  echo "Could not verify the challenge_suite target." >&2
  exit 2
fi

mkdir -p "${output_root}" "${archive_dir}"
printf 'solver\tpreconditioner\toutcome\texit_code\telapsed_seconds\n' > "${output_root}/status.tsv"
cat > "${output_root}/effective_settings.txt" <<EOF
case=06/contact_arch
mpi_ranks=8
element_order=1
time_steps=50
time_step_size=0.0032
final_time=0.16
warm_start=off
nonlinear_tolerance=1e-11
linear_tolerance=1e-14
base_nonlinear_max_iterations=10000
effective_nonlinear_max_iterations=${nonlinear_max_iterations}
nonlinear_print_level=${print_level}
max_cg_iterations_hypre_amg=600
max_cg_iterations_hypre_jacobi=60000
use_bsr_spmv=off
assemble_bsr=off
paraview=on
entry_timeout=${run_timeout}
EOF
cat > "${output_root}/parameters.tsv" <<EOF
parameter\tvalue
case\t06/contact_arch
mesh\tdata/meshes/half_circle_arch_3d.g
element_order\t1
mpi_ranks\t8
time_steps\t50
final_time\t0.16
support_inset_scale\t0.05
plane_travel_scale\t0.2
contact_penalty\t20
contact_regularization\t0.001
plane_clearance\t0.0001
density\t1
youngs_modulus\t10
poisson_ratio\t0.33
nonlinear_tolerance\t1e-11
linear_tolerance\t1e-14
nonlinear_max_iterations\t${nonlinear_max_iterations}
nonlinear_print_level\t${print_level}
max_cg_iterations_hypre_amg\t600
max_cg_iterations_hypre_jacobi\t60000
warm_start\toff
use_bsr_spmv\toff
assemble_bsr\toff
paraview\ton
entry_timeout\t${run_timeout}
EOF
stat -c $'executable=%n\nmtime=%y\nbytes=%s' "${executable}" > "${output_root}/executable.txt"
sha256sum "${executable}" >> "${output_root}/executable.txt"
{
  printf 'commit=%s\n' "$(git -C "${repo_dir}" rev-parse HEAD)"
  printf 'describe=%s\n' "$(git -C "${repo_dir}" describe --always --dirty --tags 2>/dev/null || true)"
  printf 'repository=%s\n' "${repo_dir}"
} > "${output_root}/source_state.txt"
git -C "${repo_dir}" status --short > "${output_root}/repository_status.txt"
git -C "${repo_dir}" diff --binary HEAD > "${output_root}/source.patch"
git -C "${repo_dir}" submodule status --recursive > "${output_root}/submodules.txt"
{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'uname=%s\n' "$(uname -a)"
  flux --version 2>&1 | sed 's/^/flux_version=/'
} > "${output_root}/environment.txt"

for solver in "${solvers[@]}"; do
  for preconditioner in "${preconditioners[@]}"; do
    run_entry "${solver}" "${preconditioner}"
  done
done

validation_args=()
if [[ -z "${run_filter}" ]]; then
  validation_args=(--require-all)
fi
set +e
"${validator}" "${output_root}" "${validation_args[@]}"
validation_status=$?
set -e

printf '%s\n' "${output_root}" > "${archive_dir}/source_run.txt"
printf '%s\n' "${archive_dir}" > "${output_root}/compact_archive.txt"
for file in compact_archive.txt effective_settings.txt environment.txt executable.txt parameters.tsv repository_status.txt \
  source.patch source_state.txt status.tsv submodules.txt validation.tsv; do
  if [[ -f "${output_root}/${file}" ]]; then
    cp -a "${output_root}/${file}" "${archive_dir}/"
  fi
done
for solver in "${solvers[@]}"; do
  for preconditioner in "${preconditioners[@]}"; do
    entry="${solver}_${preconditioner}"
    if [[ ! -d "${output_root}/${entry}" ]]; then
      continue
    fi
    mkdir -p "${archive_dir}/${entry}"
    for file in command.txt run.log paper_contact_arch_fast_load_displacement.csv; do
      if [[ -f "${output_root}/${entry}/${file}" ]]; then
        cp -a "${output_root}/${entry}/${file}" "${archive_dir}/${entry}/"
      fi
    done
  done
done
cp -a "${BASH_SOURCE[0]}" "${validator}" "${archive_dir}/"

mkdir -p "${build_dir}/paper_logs"
printf '%s\n' "${output_root}" > "${build_dir}/paper_logs/latest_contact_arch_response_matrix.txt"
printf '%s\n' "${archive_dir}" > "${archive_root}/latest_solver_response.txt"
echo "Authoritative matrix: ${output_root}"
echo "Compact archive: ${archive_dir}"
if [[ ${validation_status} -ne 0 ]]; then
  exit 1
fi
