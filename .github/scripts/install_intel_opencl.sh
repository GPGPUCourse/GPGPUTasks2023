#!/bin/bash
#See https://github.com/KhronosGroup/SyclParallelSTL/blob/03367798d81fe396cd8ebf5492e51ad6a495467d/.travis/install_intel_opencl.sh

set -ev

apt update

###########################
# Get Intel OpenCL Runtime
###########################

# https://software.intel.com/en-us/articles/opencl-drivers#latest_CPU_runtime
PACKAGE_URL=https://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/15532/l_opencl_p_18.1.0.015.tgz
PACKAGE_NAME=l_opencl_p_18.1.0.015

wget -q ${PACKAGE_URL} -O /tmp/opencl_runtime.tgz
tar -xzf /tmp/opencl_runtime.tgz -C /tmp
sed 's/decline/accept/g' -i /tmp/${PACKAGE_NAME}/silent.cfg
apt-get install -yq cpio
/tmp/${PACKAGE_NAME}/install.sh -s /tmp/${PACKAGE_NAME}/silent.cfg

###################################
# Check OpenCL runtime with clinfo
###################################

apt install -yq ocl-icd-opencl-dev ocl-icd-dev opencl-headers

git clone https://github.com/Oblomov/clinfo
cd clinfo
make -j4
./clinfo
cd ..
