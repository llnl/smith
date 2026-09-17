#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
validator="${repo_dir}/scripts/validate_shallow_arch_response.py"

usage() {
  cat <<'EOF'
Usage: run_shallow_arch_response_matrix.sh

Environment variables:
  BUILD_DIR    Existing release build directory.
  OUTPUT_ROOT  Matrix output directory.
  ARCHIVE_ROOT Compact archive parent. Default: buckling_solver_paper/results/shallow_arch.
  RUN_FILTER   Comma-separated solver_preconditioner entries. Default: all eight.
  RUN_TIMEOUT  Per-entry timeout. Default: 3600s.
  SHALLOW_ARCH_PRECOMPRESSION       Horizontal end-shortening. Default: 0.02.
  SHALLOW_ARCH_PRECOMPRESSION_STEPS Precompression increments. Default: 10.
  SHALLOW_ARCH_LOAD_STEPS           Vertical load increments. Default: 200.
  SHALLOW_ARCH_LOAD_MAGNITUDE       Final vertical traction magnitude. Default: 0.015.
  SHALLOW_ARCH_PARAVIEW             Set to 1 for per-step fields. Default: 1.
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
output_root="${OUTPUT_ROOT:-${build_dir}/paper_shallow_arch_response_${run_stamp}}"
archive_root="${ARCHIVE_ROOT:-/usr/workspace/tupek2/dev/buckling_solver_paper/results/shallow_arch}"
archive_dir="${archive_root}/solver_response_${run_stamp}"
run_filter="${RUN_FILTER:-}"
run_timeout="${RUN_TIMEOUT:-3600s}"
shallow_arch_precompression="${SHALLOW_ARCH_PRECOMPRESSION:-0.02}"
shallow_arch_precompression_steps="${SHALLOW_ARCH_PRECOMPRESSION_STEPS:-10}"
shallow_arch_load_steps="${SHALLOW_ARCH_LOAD_STEPS:-200}"
shallow_arch_load_magnitude="${SHALLOW_ARCH_LOAD_MAGNITUDE:-0.015}"
shallow_arch_paraview="${SHALLOW_ARCH_PARAVIEW:-1}"

if [[ "${shallow_arch_paraview}" == "1" ]]; then
  shallow_arch_output_option=--paraview
elif [[ "${shallow_arch_paraview}" == "0" ]]; then
  shallow_arch_output_option=--no-paraview
else
  echo "SHALLOW_ARCH_PARAVIEW must be 0 or 1" >&2
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
  local requested

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
    launcher=(flux run -n 16)
  else
    launcher=(flux start --test-size=16 flux run -n 16)
  fi

  command=(
    "${launcher[@]}"
    "${executable}"
    --case=02
    --size=small
    --mesh-scale=1
    --nonlinear-solver="$(solver_name "${solver}")"
    --linear-solver=CG
    --preconditioner="${preconditioner}"
    --trust-subspace-option="$(subspace_option "${solver}")"
    --nonlinear-tol=1e-11
    --linear-tol=1e-14
    "--shallow-arch-precompression=${shallow_arch_precompression}"
    "--shallow-arch-precompression-steps=${shallow_arch_precompression_steps}"
    "--shallow-arch-load-steps=${shallow_arch_load_steps}"
    "--shallow-arch-load-magnitude=${shallow_arch_load_magnitude}"
    --max-cg-iterations="${max_iterations}"
    --print-level=1
    --linear-print-level=0
    --no-use-bsr-spmv
    --no-assemble-bsr
    --use-warm-start
    "${shallow_arch_output_option}"
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
  elif grep -q "nonlinear solve failed" "${run_dir}/run.log"; then
    outcome=nonlinear_failure
  else
    outcome=process_failure
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
(cd "${build_dir}" && make -q examples/challenge_suite) >/dev/null 2>&1
build_query_status=$?
set -e
if [[ ${build_query_status} -eq 1 ]]; then
  echo "The challenge_suite target is out of date." >&2
  echo "Build it first with: cd ${build_dir} && make -j40" >&2
  exit 2
elif [[ ${build_query_status} -ne 0 ]]; then
  echo "Could not verify the challenge_suite build state in ${build_dir}." >&2
  exit 2
fi
if [[ -e "${output_root}" ]]; then
  echo "Output directory already exists: ${output_root}" >&2
  exit 2
fi
if [[ -e "${archive_dir}" ]]; then
  echo "Archive directory already exists: ${archive_dir}" >&2
  exit 2
fi

mkdir -p "${output_root}" "${archive_dir}"
printf 'solver\tpreconditioner\toutcome\texit_code\telapsed_seconds\n' > "${output_root}/status.tsv"
stat -c $'executable=%n\nmtime=%y\nbytes=%s' "${executable}" > "${output_root}/executable.txt"
sha256sum "${executable}" >> "${output_root}/executable.txt"

cat > "${output_root}/parameters.tsv" <<EOF
parameter	value	rationale
case	02/shallow_arch	Force-controlled snap-through after horizontal precompression
mesh	generated shallow arch	16 by 3 base mesh with four additional uniform refinements
global_elements	12288	Fixed generated-mesh resolution
displacement_order	1	Linear displacement interpolation
mpi_ranks	16	Standard production layout
horizontal_precompression	${shallow_arch_precompression}	Right support moves left while the left support remains fixed
precompression_steps	${shallow_arch_precompression_steps}	The horizontal end-shortening is applied before vertical loading
vertical_load_steps	${shallow_arch_load_steps}	Resolves the force-controlled transition
total_steps	$((shallow_arch_precompression_steps + shallow_arch_load_steps))	Precompression followed by vertical loading
vertical_traction_magnitude	${shallow_arch_load_magnitude}	Final downward top-surface traction magnitude
loading	sequential	Hold precompression fixed while ramping uniform downward traction on the full top boundary
initialization	warm start	Each increment starts from the prior accepted state to follow the physical response path
nonlinear_tolerance	1e-11	Standardized tight production tolerance
linear_tolerance	1e-14	Standardized tight production tolerance
linear_solver	CG	Common matrix comparison solver
amg_iteration_cap	600	Standardized AMG cap
jacobi_iteration_cap	60000	Standardized Jacobi cap
bsr_spmv	off	Explicitly disabled for every entry
assembled_bsr	off	Explicitly disabled for every entry
field_output	${shallow_arch_paraview}	One writes every accepted state; zero retains only the response history
outcome_policy	record nonlinear failure	A clean solver failure with paired partial history and fields is retained as benchmark evidence
EOF

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
for file in compact_archive.txt environment.txt executable.txt parameters.tsv repository_status.txt source.patch source_state.txt \
  status.tsv submodules.txt validation.tsv; do
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
    for file in command.txt run.log paper_shallow_arch_fast_load_displacement.csv; do
      if [[ -f "${output_root}/${entry}/${file}" ]]; then
        cp -a "${output_root}/${entry}/${file}" "${archive_dir}/${entry}/"
      fi
    done
  done
done
cp -a "${BASH_SOURCE[0]}" "${validator}" "${archive_dir}/"

mkdir -p "${build_dir}/paper_logs"
printf '%s\n' "${output_root}" > "${build_dir}/paper_logs/latest_shallow_arch_response_matrix.txt"
printf '%s\n' "${archive_dir}" > "${archive_root}/latest_solver_response.txt"
echo "Authoritative matrix: ${output_root}"
echo "Compact archive: ${archive_dir}"
if [[ ${validation_status} -ne 0 ]]; then
  exit 1
fi
