#!/usr/bin/env bash
# cleanup_gitlab_builds: remove old GitLab CI build directories.

set -euo pipefail

WORK_DIR_LINK="${HOME}/.jacamar-ci"
WORK_DIR="$(realpath "$WORK_DIR_LINK")"

if [[ ! -d "$WORK_DIR" ]]; then
    echo "Error: WORK_DIR is not a directory: $WORK_DIR_LINK -> $WORK_DIR" >&2
    exit 1
fi

LAST_MODIFY_DAYS=4
DRY_RUN=1
QUIET=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [-d DAYS] [-f] [-q] [-h]

Options:
  -d DAYS   Delete directories older than DAYS days by mtime. Default: 4
  -f        Actually delete directories. Default is dry-run.
  -q        Quiet mode
  -h        Show this help
EOF
}

while getopts ":d:fqh" opt; do
    case "$opt" in
        d) LAST_MODIFY_DAYS="$OPTARG" ;;
        f) DRY_RUN=0 ;;
        q) QUIET=1 ;;
        h) usage; exit 0 ;;
        :)
            echo "Error: -$OPTARG requires an argument" >&2
            usage >&2
            exit 2
            ;;
        \?)
            echo "Error: unknown option -$OPTARG" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$LAST_MODIFY_DAYS" =~ ^[0-9]+$ ]]; then
    echo "Error: DAYS must be a non-negative integer" >&2
    exit 2
fi

if [[ ! -d "$WORK_DIR" ]]; then
    echo "Error: WORK_DIR does not exist or is not a directory: $WORK_DIR" >&2
    exit 1
fi

space_cleared=0
dirs_seen=0
dirs_deleted=0

while IFS= read -r -d '' dir; do
    # Safety check: never delete anything outside the expected tree.
    if [[ "$dir" != "$WORK_DIR"/* ]]; then
        echo "Skipping suspicious path outside WORK_DIR: $dir" >&2
        continue
    fi

    dir_size=$(du -sk -- "$dir" | awk '{print $1}')
    dirs_seen=$((dirs_seen + 1))

    if [[ "$QUIET" -eq 0 ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "Would delete: $dir  ${dir_size} KiB"
        else
            echo "Deleting: $dir  ${dir_size} KiB"
        fi
    fi

    if [[ "$DRY_RUN" -eq 0 ]]; then
        rm -rf -- "$dir"
        space_cleared=$((space_cleared + dir_size))
        dirs_deleted=$((dirs_deleted + 1))
    fi
done < <(find "$WORK_DIR" -mindepth 4 -maxdepth 4 -type d -mtime +"${LAST_MODIFY_DAYS}" -print0)

if [[ "$QUIET" -eq 0 ]]; then
    echo
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "Dry run complete. $dirs_seen directories matched."
        echo "Run with -f to actually delete them."
    else
        echo "Deleted $dirs_deleted directories."
        echo "$space_cleared KiB cleared."
    fi
fi
