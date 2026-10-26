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

source /kassiopeia/install/bin/kasperenv.sh /kassiopeia/install

if [[ -n "${kassiopeia_dir_install}" ]]; then
  source /kassiopeia/install/bin/create_kasper_user_directory.sh /home/parrot/${kassiopeia_dir_install}
fi

rm -rf /tmp/kassiopeia_tmp_userdir # Ensure /tmp/kassiopeia_tmp_userdir does not exist
if [[ ! -w $KASPERSYS ]]; then
  echo "/kassiopeia/install is not writable. Creating user directory in /tmp/kassiopeia_tmp_userdir instead."
  source /kassiopeia/install/bin/create_kasper_user_directory.sh /tmp/kassiopeia_tmp_userdir
fi

exec "$@"
