#!/bin/bash

echo "DO_INTEGRATION_TESTS=${DO_INTEGRATION_TESTS}"
echo "EXTRA_BUILD_OPTIONS=${EXTRA_BUILD_OPTIONS}"
echo "EXTRA_CMAKE_OPTIONS=${EXTRA_CMAKE_OPTIONS}"
echo "HOST_CONFIG=${HOST_CONFIG}"

# EXTRA_CMAKE_OPTIONS needs quotes wrapped around it, since it may contain spaces, and we want them all to be a part of
# one large string. (e.g. "-DSMITH_ENABLE_CODEVELOP=ON -DENABLE_DOCS=OFF")
# EXTRA_BUILD_OPTIONS does not need quotes because - while it also may contain spaces, are seperate build_src.py
# arguments (e.g. --skip-install --jobs=8).

# Build source, run unit tests, and test install examples
python3 scripts/llnl/build_src.py -v \
    --host-config=${HOST_CONFIG} \
    --extra-cmake-options="${EXTRA_CMAKE_OPTIONS}" \
    ${EXTRA_BUILD_OPTIONS}
if [ $? -ne 0 ]; then { echo "ERROR: build_src.py failed." ; exit 1; } fi

# Run integration tests
if [[ "$DO_INTEGRATION_TESTS" == "yes" ]] ; then
    cd *build_and_test_*/build-*
    ./ats.sh
    if [ $? -ne 0 ]; then 
        # Print each failing test's logs
        cd *.*.logs
        FAILING_TEST_NUMBERS=$(awk '/#[0-9]* FAIL/ {print $1}' ats.log | cut -c 2-)
        for num in $FAILING_TEST_NUMBERS ; do
            LOG_FILENAMES=(*$num*.log *$num*.log.err)
            for f in ${LOG_FILENAMES[@]} ; do
                if [ -s $f ] ; then
                    echo "======== START $f START ========"
                    cat $f
                    echo "======== END $f END ========"
                fi
            done
        done

        echo "ERROR: ATS failed."
        exit 1
    fi
fi
