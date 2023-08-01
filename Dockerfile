# --- runtime-base ---
FROM fedora:38 as runtime-base

LABEL description="Runtime base container"
LABEL maintainer="jan.behrens@kit.edu"

COPY Docker/packages.runtime packages
RUN dnf update -y \
 && dnf install -y $(cat packages) \
 && rm /packages

# --- build-base ---
FROM runtime-base as build-base

LABEL description="Build base container"

COPY Docker/packages.build packages
RUN dnf update -y \
 && dnf install -y $(cat packages) \
 && rm /packages

# --- build ---
FROM build-base as build

LABEL description="Build container"

COPY . /usr/src/kasper
RUN cd /usr/src/kasper && \
    mkdir -p build && \
    pushd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_STANDARD=14 \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_UNIT_TESTS=ON \
          -DKASPER_USE_ROOT=ON \
          -DKASPER_USE_VTK=ON \
          -DKASPER_USE_TBB=OFF \
          -DKEMField_USE_OPENCL=OFF \
       .. && \
    make -j $((($(nproc)+1)/2)) && \
    make install && \
    popd

# --- runtime ---
FROM runtime-base as runtime

LABEL description="Run container"

COPY --from=build /usr/local /usr/local
RUN echo "/usr/local/lib64" > /etc/ld.so.conf.d/local-x86_64.conf \
 && ldconfig
RUN strip --remove-section=.note.ABI-tag /usr/lib64/libQt5Core.so.5

WORKDIR /usr/local

RUN echo '#!/bin/bash'                         >  /usr/local/bin/entrypoint.sh && \
    echo '. /usr/local/bin/kasperenv.sh `pwd`' >> /usr/local/bin/entrypoint.sh && \
    echo 'exec "$@"'                           >> /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["/usr/local/bin/entrypoint.sh","Kassiopeia"]
