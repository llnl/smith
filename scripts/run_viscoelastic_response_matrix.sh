#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
validator="${repo_dir}/scripts/validate_viscoelastic_response.py"

usage() {
  cat <<'EOF'
Usage: run_viscoelastic_response_matrix.sh

Environment variables:
  BUILD_DIR    Existing release build directory.
  OUTPUT_ROOT  Matrix output directory.
  ARCHIVE_ROOT Compact archive parent. Default: buckling_solver_paper/results/viscoelastic_buckling.
  RUN_FILTER   Comma-separated solver_preconditioner entries. Default: all eight.
  RUN_TIMEOUT  Per-entry timeout. Default: 1800s.
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
output_root="${OUTPUT_ROOT:-${build_dir}/paper_viscoelastic_response_${run_stamp}}"
archive_root="${ARCHIVE_ROOT:-/usr/workspace/tupek2/dev/buckling_solver_paper/results/viscoelastic_buckling}"
archive_dir="${archive_root}/solver_response_${run_stamp}"
run_filter="${RUN_FILTER:-}"
run_timeout="${RUN_TIMEOUT:-1800s}"

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
    launcher=(flux start "--test-size=16" flux run -n 16)
  fi

  command=(
    "${launcher[@]}"
    "${executable}"
    "--case=04"
    "--size=small"
    "--mesh-scale=1"
    "--nonlinear-solver=$(solver_name "${solver}")"
    "--linear-solver=CG"
    "--preconditioner=${preconditioner}"
    "--trust-subspace-option=$(subspace_option "${solver}")"
    "--nonlinear-tol=1e-11"
    "--linear-tol=1e-14"
    "--max-cg-iterations=${max_iterations}"
    --no-warm-start
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
  echo "Build it first with: cd ${build_dir} && make -j70 challenge_suite" >&2
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

cat > "${output_root}/parameters.tsv" <<'EOF'
parameter	value	rationale
case	04/viscoelastic_buckling	Finite-deformation viscoelastic loading, unloading, and recovery
mesh	data/meshes/cap_hemisphere.g	1097-element hemispherical-cap mesh
displacement_order	2	Quadratic displacement interpolation
mpi_ranks	16	Standard production layout
time_steps	80	Uniform increments over time 0 to 24
time_step_size	0.3	Fixed transient increment
loading	force controlled	Top-patch resultant reaches 0.6 before removal at time 6
initialization	cold start	Every nonlinear step starts from the zero increment
nonlinear_tolerance	1e-11	Standardized tight production tolerance
linear_tolerance	1e-14	Standardized tight production tolerance
linear_solver	CG	Common matrix comparison solver
amg_iteration_cap	600	Standardized AMG cap
jacobi_iteration_cap	60000	Standardized Jacobi cap
bsr_spmv	off	Explicitly disabled for every entry
assembled_bsr	off	Explicitly disabled for every entry
field_output	on	One ParaView cycle for the initial state and every accepted step
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
    for file in command.txt run.log paper_viscoelastic_buckling_fast_force_displacement.csv; do
      if [[ -f "${output_root}/${entry}/${file}" ]]; then
        cp -a "${output_root}/${entry}/${file}" "${archive_dir}/${entry}/"
      fi
    done
  done
done
cp -a "${BASH_SOURCE[0]}" "${validator}" "${archive_dir}/"

mkdir -p "${build_dir}/paper_logs"
printf '%s\n' "${output_root}" > "${build_dir}/paper_logs/latest_viscoelastic_response_matrix.txt"
echo "Authoritative matrix: ${output_root}"
echo "Compact archive: ${archive_dir}"
if [[ ${validation_status} -ne 0 ]]; then
  exit 1
fi
