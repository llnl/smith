#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
validator="${repo_dir}/scripts/validate_euler_refined_verification.py"

usage() {
  cat <<'EOF'
Usage: run_euler_refined_verification.sh

Runs the one-level-refined Euler analytic-verification case and creates both an
authoritative run directory and a compact provenance archive.

Environment variables:
  BUILD_DIR     Existing prescribed release build directory.
  OUTPUT_ROOT   Authoritative run directory. Default: timestamped directory in BUILD_DIR.
  ARCHIVE_ROOT  Compact archive parent. Default: buckling_solver_paper/results/euler.
  TIMEOUT       Run timeout. Default: 7200s.
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
  echo "The refined Euler verification must run on a compute node." >&2
  exit 2
fi

build_dir="${BUILD_DIR:-${repo_dir}/build-rzwhippet-toss_4_x86_64_ib-llvm@19.1.3-release}"
executable="${build_dir}/examples/challenge_suite"
run_stamp=$(date +%Y%m%d_%H%M%S)
output_root="${OUTPUT_ROOT:-${build_dir}/paper_euler_refined_verification_${run_stamp}}"
archive_root="${ARCHIVE_ROOT:-/usr/workspace/tupek2/dev/buckling_solver_paper/results/euler}"
archive_dir="${archive_root}/refined_verification_${run_stamp}"
timeout_duration="${TIMEOUT:-7200s}"

if [[ ! -x "${executable}" ]]; then
  echo "Missing executable: ${executable}" >&2
  echo "Build it with the prescribed release-build command before running this script." >&2
  exit 2
fi
set +e
make -q -C "${build_dir}" challenge_suite/fast >/dev/null 2>&1
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
if [[ ! -x "${validator}" ]]; then
  echo "Missing validator: ${validator}" >&2
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

if [[ -n "${FLUX_URI:-}" ]]; then
  launcher=(flux run -n 16)
else
  launcher=(flux start "--test-size=16" flux run -n 16)
fi

command=(
  "${launcher[@]}"
  "${executable}"
  "--case=01"
  "--order=2"
  "--size=small"
  "--mesh-scale=1"
  "--euler-extra-refinement=1"
  "--euler-load=0.0022916666666666667"
  "--euler-refined-start-force=0.0027"
  "--euler-selector-traction=-1e-10"
  "--euler-coarse-load-steps=30"
  "--euler-refined-load-steps=5"
  "--nonlinear-solver=TrustRegion"
  "--linear-solver=CG"
  "--preconditioner=HypreAMG"
  "--trust-subspace-option=0"
  "--nonlinear-tol=1e-11"
  "--linear-tol=8e-12"
  "--max-cg-iterations=600"
  --use-warm-start
  --no-use-bsr-spmv
  --no-assemble-bsr
  --no-paraview
  "--print-level=1"
  "--linear-print-level=0"
  --timings
)

cat > "${output_root}/parameters.tsv" <<'EOF'
parameter	value	rationale
case	01/euler	Three-dimensional Euler cantilever benchmark
purpose	analytic-verification	Bracket the nonlinear branch switch around the classical Euler load
displacement_order	2	Quadratic interpolation for bending response
base_global_elements	1400	Structured 4 x 7 x 50 hexahedral mesh
extra_uniform_refinements	1	Produces 11200 global elements
mpi_ranks	16	Historical refined-run layout
nonlinear_solver	TrustRegion	Historical verification solver
linear_solver	CG	Historical verification linear solver
preconditioner	HypreAMG	Historical verification preconditioner
warm_start	on	Euler default and historical effective behavior
nonlinear_tolerance	1e-11	Verification tolerance
linear_tolerance	8e-12	Historical verification exception; not a timing-comparison setting
maximum_cg_iterations	600	AMG iteration cap
final_top_traction	0.0022916666666666667	Top area 1.2 gives final resultant 0.00275
lateral_selector_traction	-1e-10	Selects a deterministic weak-axis branch
coarse_load_steps	30	Advances from zero to resultant 0.00270
refined_load_steps	5	Advances from resultant 0.00270 to 0.00275
refined_force_spacing	1e-5	Sets the final critical-load bracket resolution
bsr_spmv	off	Hypre AMG forces this path off; stated explicitly for provenance
assembled_bsr	off	Hypre AMG uses the non-BSR assembled path
field_output	off	The result is a response-history verification, not a field-comparison run
EOF

printf '%q ' "${command[@]}" > "${output_root}/command.txt"
printf '\n' >> "${output_root}/command.txt"
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

echo "Running refined Euler verification in ${output_root}"
start_seconds=$(date +%s)
set +e
(
  cd "${output_root}"
  /usr/bin/time -p timeout --signal=TERM "${timeout_duration}" "${command[@]}" 2>&1 | tee run.log
  exit "${PIPESTATUS[0]}"
)
run_status=$?
set -e
end_seconds=$(date +%s)
elapsed_seconds=$((end_seconds - start_seconds))

if [[ ${run_status} -eq 0 ]]; then
  outcome=completed
elif [[ ${run_status} -eq 124 ]]; then
  outcome=timeout
else
  outcome=failed
fi

printf 'outcome\texit_code\telapsed_seconds\tvalidation_exit_code\n%s\t%s\t%s\t\n' \
  "${outcome}" "${run_status}" "${elapsed_seconds}" > "${output_root}/status.tsv"

set +e
"${validator}" "${output_root}" --outcome "${outcome}"
validation_status=$?
set -e
if [[ ${run_status} -eq 0 && ${validation_status} -ne 0 ]]; then
  outcome=invalid
fi
printf 'outcome\texit_code\telapsed_seconds\tvalidation_exit_code\n%s\t%s\t%s\t%s\n' \
  "${outcome}" "${run_status}" "${elapsed_seconds}" "${validation_status}" > "${output_root}/status.tsv"

printf '%s\n' "${output_root}" > "${archive_dir}/source_run.txt"
printf '%s\n' "${archive_dir}" > "${output_root}/compact_archive.txt"
for file in command.txt compact_archive.txt environment.txt executable.txt parameters.tsv repository_status.txt run.log \
  source.patch source_state.txt status.tsv submodules.txt validation.tsv paper_euler_fast_load_displacement.csv; do
  if [[ -f "${output_root}/${file}" ]]; then
    cp -a "${output_root}/${file}" "${archive_dir}/"
  fi
done
cp -a "${BASH_SOURCE[0]}" "${validator}" "${archive_dir}/"

mkdir -p "${build_dir}/paper_logs"
printf '%s\n' "${output_root}" > "${build_dir}/paper_logs/latest_euler_refined_verification.txt"
echo "Authoritative run: ${output_root}"
echo "Compact archive: ${archive_dir}"

if [[ ${run_status} -ne 0 ]]; then
  exit "${run_status}"
fi
if [[ ${validation_status} -ne 0 ]]; then
  exit 1
fi
