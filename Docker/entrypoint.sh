#!/bin/bash
echo "/kasper/entrypoint.sh: Setting up environment"

if [[ -n "${kasper_dir_code}" ]]; then
  ln -s "/home/parrot/$kasper_dir_code" /kasper/code
fi

if [[ -n "${kasper_dir_build}" && -d /kasper/build ]]; then
  mkdir -p /home/parrot/${kasper_dir_build}
  if ! [ "$(ls -A /home/parrot/$kasper_dir_build)" ]; then
    echo "Preparing custom build folder"
    mv /kasper/build/* /home/parrot/$kasper_dir_build
  fi
  rm -r /kasper/build
  ln -s "/home/parrot/$kasper_dir_build" /kasper/build
fi

if [[ -n "${kasper_dir_install}" ]]; then
  mkdir -p /home/parrot/${kasper_dir_install}
  if ! [ "$(ls -A /home/parrot/$kasper_dir_install)" ]; then
    echo "Preparing custom install folder"
    mv /kasper/install/* /home/parrot/$kasper_dir_install
  fi
  rm -r /kasper/install
  ln -s "/home/parrot/$kasper_dir_install" /kasper/install
fi

source /kasper/install/bin/kasperenv.sh /kasper/install

exec "$@"
