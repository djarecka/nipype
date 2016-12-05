#!/bin/bash
set -e
set -x
set -u

PYTHON_VERSION=$( python -c "import sys; print('{}{}'.format(sys.version_info[0], sys.version_info[1]))" )

# Create necessary directories
mkdir -p /scratch/pytest /scratch/crashfiles /scratch/logs/py${PYTHON_VERSION}

# Create a nipype config file
mkdir -p /root/.nipype
echo '[logging]' > /root/.nipype/nipype.cfg
echo 'log_to_file = true' >> /root/.nipype/nipype.cfg
echo "log_directory = /scratch/logs/py${PYTHON_VERSION}" >> /root/.nipype/nipype.cfg

# Enable profile_runtime tests only for python 2.7
if [[ "${PYTHON_VERSION}" -lt "30" ]]; then
    echo '[execution]' >> /root/.nipype/nipype.cfg
    echo 'profile_runtime = true' >> /root/.nipype/nipype.cfg
fi

# Run tests using pytest
# NOTE_dj: this has to be improved, help needed.
# NOTE_dj: pip install can be probably moved to dockerfiles
# NOTE_dj: not sure about --xunit-file part (not using with pytest for now)
cd /root/src/nipype/
make clean
conda install pytest
pip install pytest-raisesregexp
conda install pytest-cov
conda install click
py.test --doctest-modules --cov-report xml:/scratch/coverage_py${PYTHON_VERSION}.xml --cov=nipype nipype
#nosetests -s nipype -c /root/src/nipype/.noserc --xunit-file="/scratch/nosetests_py${PYTHON_VERSION}.xml" --cover-xml-file="/scratch/coverage_py${PYTHON_VERSION}.xml"

# Workaround: run here the profiler tests in python 3
# NOTE_dj: not sure why this is different for python 3 and 2, is it ok that only python3 part is included in coverage?
# NOTE_dj: I'm not sure about --xunit-file
if [[ "${PYTHON_VERSION}" -ge "30" ]]; then
    echo '[execution]' >> /root/.nipype/nipype.cfg
    echo 'profile_runtime = true' >> /root/.nipype/nipype.cfg
    py.test --cov-report xml:/scratch/coverage_py${PYTHON_VERSION}_profiler.xml --cov=nipype nipype/interfaces/tests/test_runtime_profiler.py #--xunit-file="/scratch/nosetests_py${PYTHON_VERSION}_profiler.xml" --cover-xml-file="/scratch/coverage_py${PYTHON_VERSION}_profiler.xml"
    py.test --cov-report xml:/scratch/coverage_py${PYTHON_VERSION}_multiproc.xml --cov=nipype nipype/pipeline/plugins/tests/test_multiproc*.py #--xunit-file="/scratch/nosetests_py${PYTHON_VERSION}_multiproc.xml" --cover-xml-file="/scratch/coverage_py${PYTHON_VERSION}_multiproc.xml"
fi

# Copy crashfiles to scratch
for i in $(find /root/src/nipype/ -name "crash-*" ); do cp $i /scratch/crashfiles/; done
chmod 777 -R /scratch/*
