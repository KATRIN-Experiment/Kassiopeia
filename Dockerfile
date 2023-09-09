# Global args
ARG KASSIOPEIA_UID="1000"
ARG KASSIOPEIA_USER="parrot"
ARG KASSIOPEIA_GID="1000"
ARG KASSIOPEIA_GROUP="kassiopeia"

ARG KASSIOPEIA_GIT_BRANCH=""
ARG KASSIOPEIA_GIT_COMMIT=""

ARG KASSIOPEIA_CPUS=""

# --- runtime-base ---
# NOTE: Fedora 36 is used because of this issue: https://gitlab.freedesktop.org/mesa/mesa/-/issues/9442
#       If our IT can circumvent this with a JupyterHub VM emulation change, we may ignore this issue in the future.
# NOTE: For Fedora 37 & 38, uncomment marked lines below
# NOTE: For Fedora 39, remove marked lines below
FROM fedora:36 as runtime-base
ARG KASSIOPEIA_UID
ARG KASSIOPEIA_USER
ARG KASSIOPEIA_GID
ARG KASSIOPEIA_GROUP

LABEL description="Runtime base container"

# # TODO REMOVE FOR FEDORA 39
# RUN dnf update -y \
#  && dnf install -y --setopt=install_weak_deps=False dnf-plugins-core \
#  && dnf clean all
# RUN dnf copr enable thofmann/log4xx-1.x -y
# # END TODO

COPY Docker/packages.runtime packages
RUN dnf update -y \
 && dnf install -y --setopt=install_weak_deps=False $(cat packages) \
 && rm /packages \
 && dnf clean all

# Setting user
# Compare:
# * https://github.com/jupyter/docker-stacks/blob/master/base-notebook/Dockerfile
# * https://docs.docker.com/develop/develop-images/dockerfile_best-practices/

RUN groupadd -g $KASSIOPEIA_GID $KASSIOPEIA_GROUP && useradd --no-log-init -r --create-home -g $KASSIOPEIA_GROUP -u $KASSIOPEIA_UID $KASSIOPEIA_USER \
 && mkdir /kassiopeia \
 && chown $KASSIOPEIA_USER:$KASSIOPEIA_GROUP /kassiopeia

# For backwards compatibility:
RUN ln -s /kassiopeia /home/$KASSIOPEIA_USER/kassiopeia

# Local directory for Python packages
# /kassiopeia/install is created by setup.sh before Python packages are installed
ENV PYTHONUSERBASE=/kassiopeia/install/python
# ---

# --- build-base ---
FROM runtime-base as build-base

LABEL description="Build base container"

COPY Docker/packages.build packages
RUN dnf update -y \
 && dnf install -y --setopt=install_weak_deps=False $(cat packages) \
 && rm /packages \
 && dnf clean all
# ---

# --- build ---
FROM build-base as build
ARG KASSIOPEIA_USER
ARG KASSIOPEIA_GROUP

ARG KASSIOPEIA_GIT_BRANCH
ARG KASSIOPEIA_GIT_COMMIT

ARG KASSIOPEIA_CPUS

LABEL description="Build container"

USER $KASSIOPEIA_USER

COPY --chown=$KASSIOPEIA_USER:$KASSIOPEIA_GROUP . /kassiopeia/code
RUN KASSIOPEIA_BUILD_TYPE="RelWithDebInfo" \
        KASSIOPEIA_INSTALL_PREFIX="/kassiopeia/install" \
        KASSIOPEIA_BUILD_PREFIX="/kassiopeia/build" \
        KASSIOPEIA_MAKECMD="ninja" \
        KASSIOPEIA_CUSTOM_CMAKE_ARGS="-GNinja" \
        /kassiopeia/code/setup.sh && \
    mkdir /kassiopeia/install/log/build && \
    cp /kassiopeia/build/.ninja_log /kassiopeia/install/log/build/ && \
    rm -r /kassiopeia/build && \
    rm -r /kassiopeia/code

COPY --chown=$KASSIOPEIA_USER:$KASSIOPEIA_GROUP Docker/entrypoint.sh /kassiopeia/

# Update /kassiopeia permissions to everyone
USER root
RUN chmod --recursive a=u /kassiopeia
USER $KASSIOPEIA_USER

WORKDIR /kassiopeia

ENTRYPOINT ["/kassiopeia/entrypoint.sh"]
# ---

# --- minimal ---
FROM runtime-base as minimal
ARG KASSIOPEIA_USER

LABEL description="Minimal run container"

USER root

RUN mkdir /kassiopeia/install \
 && chown $KASSIOPEIA_USER:$KASSIOPEIA_GROUP /kassiopeia/install

COPY --from=build /kassiopeia/entrypoint.sh /kassiopeia/entrypoint.sh
COPY --from=build /kassiopeia/install /kassiopeia/install
 
RUN echo /kassiopeia/install/lib64 > /etc/ld.so.conf.d/local-x86_64.conf \
 && ldconfig

# Update /kassiopeia permissions to everyone
RUN chmod --recursive a=u /kassiopeia

USER $KASSIOPEIA_USER

WORKDIR /home/$KASSIOPEIA_USER

ENTRYPOINT ["/kassiopeia/entrypoint.sh"]

CMD ["bash"]
# ---

# --- full-base ---
FROM build-base as full-base

LABEL description="Full base container"

ENV JUPYTER_CONFIG_DIR=/kassiopeia/install/python/.jupyter

COPY Docker/packages.full packages
RUN dnf update -y \
 && dnf install -y --setopt=install_weak_deps=False $(cat packages) \
 && rm /packages \
 && dnf clean all
RUN pip3 install --no-cache-dir jupyterlab \
 && pip3 install --no-cache-dir jupyter-server-proxy \
 && pip3 install --no-cache-dir jupyterhub \
 && pip3 install --no-cache-dir ipympl \
 && pip3 install --no-cache-dir iminuit

# Ensure if LDAP is used on a JupyterHub, user names are correctly resolved
# Corresponding packages: nslcd, libnss-ldapd
RUN sed -e 's/^passwd:\(.*\)/passwd:\1 ldap/' -e 's/^group:\(.*\)/group:\1 ldap/' -i /etc/nsswitch.conf
# ---

# --- full ---
# Include build files to enable faster development
FROM full-base as full
ARG KASSIOPEIA_USER

LABEL description="Full run container"

USER root

RUN mkdir /kassiopeia/install \
 && chown $KASSIOPEIA_USER:$KASSIOPEIA_GROUP /kassiopeia/install

COPY --from=build /kassiopeia/entrypoint.sh /kassiopeia/entrypoint.sh
COPY --from=build /kassiopeia/install /kassiopeia/install

RUN echo /kassiopeia/install/lib64 > /etc/ld.so.conf.d/local-x86_64.conf \
 && ldconfig \
 && echo "source /usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> /home/$KASSIOPEIA_USER/.zshrc

# Always show Kasper information when opening a terminal
RUN echo "source /kassiopeia/install/bin/kassiopeiaenv.sh" >> /etc/profile

# JupyterLab VNC desktop environment:
#  Adapted from renku-vnc by SwissDataScienceCenter
#    https://github.com/SwissDataScienceCenter/renku-vnc/tree/c458e1cefc017be657f7068605ad72c7ce91d78d/xvnc4
#  License: Apache License 2.0
#    https://github.com/SwissDataScienceCenter/renku-vnc/blob/5304c95e77b1ef3a71f224bc43582c7dd52b5dc8/LICENSE
# Fix vnc.html
# Fix vnc_lite.html
# Resize to browser window
# Make vnc_lite default
RUN sed -i -e "s,'websockify',window.location.pathname.slice(1),g" /usr/share/novnc/app/ui.js \
    && sed -i -e "s,'websockify',window.location.pathname.slice(1),g" /usr/share/novnc/vnc_lite.html \
    && sed -i -e "s/rfb.scaleViewport = readQueryVariable('scale', false);/rfb.scaleViewport = readQueryVariable('scale', false);rfb.resizeSession = true;/g" /usr/share/novnc/vnc_lite.html \
    && sed -i -e 's,<div id="sendCtrlAltDelButton">Send CtrlAltDel</div>,<div id="sendCtrlAltDelButton" hidden>Send CtrlAltDel</div><div onClick="window.location.reload(true);" style="position: fixed;top: 0px;right: 0px;border: 1px outset;padding: 5px 5px 4px 5px;cursor: pointer;">Reload</div>,g' /usr/share/novnc/vnc_lite.html \
    && ln -fs /usr/share/novnc/vnc_lite.html /usr/share/novnc/index.html
COPY --chown=root:root Docker/startvnc /

USER $KASSIOPEIA_USER

# Configure VNC desktop
RUN jupyter lab --generate-config \
    && echo "c.ServerProxy.servers = {\
    'vnc': {\
        'command': ['/startvnc', '{port}'],\
        'timeout' : 10,\
        'absolute_url': False,\
        'new_browser_tab': False,\
        'launcher_entry' :  {\
            'enabled': True,\
            'title': 'VNC (Desktop)'\
        }\
    }\
}" >> $JUPYTER_CONFIG_DIR/jupyter_lab_config.py
# Fix DISPLAY so applications can also use it outside the desktop environment
ENV DISPLAY=:20

# Update /kassiopeia permissions to everyone (needed e.g. in some JupyterHub environments)
USER root
RUN chmod --recursive a=u /kassiopeia
USER $KASSIOPEIA_USER

WORKDIR /home/$KASSIOPEIA_USER

ENTRYPOINT ["/kassiopeia/entrypoint.sh"]

CMD ["jupyter", "lab", "--port=44444", "--ip=0.0.0.0", "--ServerApp.custom_display_url='http://localhost:44444/'"]
# ---
