#!/usr/bin/env bash

set -uo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'EOF'
Usage: run_circ_in_circ_response_matrix.sh

Environment variables:
  BUILD_DIR    Existing release build directory.
  OUTPUT_ROOT  Matrix output directory.
  RUN_FILTER   Comma-separated solver_preconditioner entries. Default: all eight.
  RUN_TIMEOUT  Per-entry timeout. Default: 1200s.
  The executable must be rebuilt after hard-coding Functional::Q = 4.
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
output_root="${OUTPUT_ROOT:-${build_dir}/paper_circ_in_circ_q4x4_response_${run_stamp}}"
run_filter="${RUN_FILTER:-}"
run_timeout="${RUN_TIMEOUT:-1200s}"

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
    --case=08
    --size=small
    --mesh-scale=1
    --nonlinear-solver="$(solver_name "${solver}")"
    --linear-solver=CG
    --preconditioner="${preconditioner}"
    --trust-subspace-option="$(subspace_option "${solver}")"
    --nonlinear-tol=1e-11
    --linear-tol=1e-14
    --max-cg-iterations="${max_iterations}"
    --print-level=1
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
if ! grep -Eq '^[[:space:]]*static constexpr auto Q = 4;' \
  "${repo_dir}/src/smith/numerics/functional/functional.hpp"; then
  echo "Circle-in-circle requires Functional::Q = 4 in the source." >&2
  exit 2
fi

set +e
(cd "${build_dir}" && make -q examples/challenge_suite) >/dev/null 2>&1
build_query_status=$?
set -e
if [[ ${build_query_status} -eq 1 ]]; then
  echo "The challenge_suite target is out of date; rebuild before running." >&2
  exit 2
elif [[ ${build_query_status} -ne 0 ]]; then
  echo "Could not verify the challenge_suite build state in ${build_dir}." >&2
  exit 2
fi

mkdir -p "${output_root}"
printf 'solver\tpreconditioner\toutcome\texit_code\telapsed_seconds\n' > "${output_root}/status.tsv"
stat -c $'executable=%n\nmtime=%y\nbytes=%s' "${executable}" > "${output_root}/executable.txt"
sha256sum "${executable}" >> "${output_root}/executable.txt"
{
  printf 'commit=%s\n' "$(git -C "${repo_dir}" rev-parse HEAD)"
  printf 'describe=%s\n' "$(git -C "${repo_dir}" describe --always --dirty)"
  printf 'repository=%s\n' "${repo_dir}"
  printf 'captured_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'build_directory=%s\n' "${build_dir}"
  printf 'build_command=cd %s && make -j40\n' "${build_dir}"
  printf 'quadrature_override=src/smith/numerics/functional/functional.hpp: Functional::Q = 4\n'
  printf 'quadrature_points_per_coordinate_direction=4\n'
  printf 'quadrature_points_per_quadrilateral=16\n'
  printf 'scope=global Functional override; executable reserved for the circ_in_circ matrix\n'
} > "${output_root}/source_state.txt"
git -C "${repo_dir}" status --short > "${output_root}/source_status.txt"
git -C "${repo_dir}" diff --binary HEAD -- > "${output_root}/source.patch"
cp "${repo_dir}/scripts/run_circ_in_circ_response_matrix.sh" "${output_root}/run_circ_in_circ_response_matrix.sh"
cp "${repo_dir}/scripts/validate_circ_in_circ_response.py" "${output_root}/validate_circ_in_circ_response.py"
cat > "${output_root}/parameters.tsv" <<'EOF'
parameter	value	rationale
case	05/circ_in_circ	Paper case 05 maps to challenge-suite case 08
mesh	data/meshes/circ_in_circ.g	First-order quadrilateral material-contrast mesh
displacement_order	1	Case definition
mpi_ranks	8	Matched solver comparison layout
initialization	cold start	Every nonlinear increment starts without a warm start
load_steps	50	Uniform increments over time 0 to 1
quadrature_points_per_direction	4	Temporarily hard-coded Functional::Q value
quadrature_points_per_quadrilateral	16	Tensor-product quadrature on each quadrilateral
quadrature_override_scope	global	Executable is reserved for this matrix
nonlinear_tolerance	1e-11	Production comparison tolerance
linear_tolerance	1e-14	Production comparison tolerance
linear_solver	CG	Common matrix comparison solver
amg_iteration_cap	600	Production AMG cap
jacobi_iteration_cap	60000	Production Jacobi cap
bsr_spmv	off	Explicitly disabled for every entry
assembled_bsr	off	Explicitly disabled for every entry
field_output	on	Initial state and every accepted increment
EOF

for solver in "${solvers[@]}"; do
  for preconditioner in "${preconditioners[@]}"; do
    run_entry "${solver}" "${preconditioner}"
  done
done

python3 "${repo_dir}/scripts/validate_circ_in_circ_response.py" \
  --input "${output_root}" \
  --output "${output_root}/validation.tsv"

mkdir -p "${build_dir}/paper_logs"
printf '%s\n' "${output_root}" > "${build_dir}/paper_logs/latest_circ_in_circ_response_matrix.txt"
echo "Response matrix written to ${output_root}"
