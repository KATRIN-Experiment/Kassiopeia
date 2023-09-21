#!/bin/bash
echo "/kassiopeia/entrypoint.sh: Setting up environment"

if [[ -n "${kassiopeia_dir_code}" ]]; then
  ln -s "/home/parrot/$kassiopeia_dir_code" /kassiopeia/code
fi

if [[ -n "${kassiopeia_dir_build}" && -d /kassiopeia/build ]]; then
  mkdir -p /home/parrot/${kassiopeia_dir_build}
  if ! [ "$(ls -A /home/parrot/$kassiopeia_dir_build)" ]; then
    echo "Preparing custom build folder"
    mv /kassiopeia/build/* /home/parrot/$kassiopeia_dir_build
  fi
  rm -r /kassiopeia/build
  ln -s "/home/parrot/$kassiopeia_dir_build" /kassiopeia/build
fi

if [[ -n "${kassiopeia_dir_install}" ]]; then
  mkdir -p /home/parrot/${kassiopeia_dir_install}
  if ! [ "$(ls -A /home/parrot/$kassiopeia_dir_install)" ]; then
    echo "Preparing custom install folder"
    mv /kassiopeia/install/* /home/parrot/$kassiopeia_dir_install
  fi
  rm -r /kassiopeia/install
  ln -s "/home/parrot/$kassiopeia_dir_install" /kassiopeia/install
fi

source /kassiopeia/install/bin/kasperenv.sh /kassiopeia/install

exec "$@"
