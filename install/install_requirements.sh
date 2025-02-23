#!/usr/bin/env bash

function installWithoutConda {
  echo "Install without conda"

  echo "Install requierments with pip"
  python -m pip install --no-cache-dir -r install/pip/no_conda.txt

}  

function installWithConda {
  echo "Install with conda"
  
  if (! command -v "conda" &> /dev/null ) then
    echo "Try to install basic Python requirements without Miniconda\n"
    installWithoutConda
  else
    conda install --yes -c conda-forge \
      beautifulsoup4 \
      jupyter_core \
      jupyterlab \
      keras \
      Keras-Preprocessing \
      matplotlib-base \
      nodejs \
      Pillow \
      pandas \
      py-cpuinfo \
      pyopengl \
      pytables  \
      scikit-image \
      scikit-learn \
      scipy \
      seaborn \
      selenium \
      statsmodels \
      sympy \
      tensorflow
  fi

}

# Install all required libraries t
function installWithPip {
  echo "Install with pip"

  echo "Prepare pip"
  python -m pip install --upgrade pip    
  python -m pip install setuptools

  echo "Install requierments with pip"
  python -m pip install --no-cache-dir -r install/pip/requirements.txt

}

# Detect OS
unameOut="$(uname -s)"
os="${unameOut:0:7}"
case "${os}" in
    Linux*)     
      installWithConda
      installWithPip
    ;;
    # MacOS
    Darwin*)
      installWithConda
      installWithPip
    ;;
    # Git Bash
    MINGW*)     
      installWithoutConda
      installWithPip
    ;;
    *)          
      installWithoutConda
      installWithPip
esac