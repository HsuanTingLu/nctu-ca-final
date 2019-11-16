#!/bin/bash -l

if [ $# -ne 2 ]; then
    echo "not enough arguments for build-script"
    exit 126
fi

echo ::debug step=update::Upgrading environment
echo -e "\n\nUpdating packages...\n"
apt-get update
apt-get install -y build-essential cmake

echo ::debug step=build::Start building
echo -e "\n\nStart building...\n"
cd $1 && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=$2
make $1

# outputs: https://help.github.com/en/actions/automating-your-workflow-with-github-actions/development-tools-for-github-actions#set-an-output-parameter-set-output
