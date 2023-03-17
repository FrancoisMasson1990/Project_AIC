FROM quay.io/opendatahub-contrib/workbench-images:base-c9s-py39_2023b_latest

USER 0

RUN yum install -y yum-utils && \
    yum-config-manager --enable crb && \
    dnf install -y https://download.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    dnf install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-9.noarch.rpm && \
    INSTALL_PKGS="ffmpeg libXext libSM python3-tkinter" && \
    yum install -y --setopt=tsflags=nodocs $INSTALL_PKGS && \
    yum -y clean all --enablerepo='*'

# Copying AIC and packages files
COPY --chown=1001:0 aic /opt/app-root/src/Project_AIC/aic
COPY --chown=1001:0 configs /opt/app-root/src/Project_AIC/configs
COPY --chown=1001:0 scripts /opt/app-root/src/Project_AIC/scripts
COPY --chown=1001:0 setup.py README.md Pipfile.lock /opt/app-root/src/Project_AIC/

USER 1001

#WORKDIR /opt/app-root/bin/
WORKDIR /opt/app-root/src/Project_AIC

# Install packages and cleanup
# (all commands are chained to minimize layer size)
RUN echo "Installing softwares and packages" && \
    # Install Python packages \
    micropipenv install && \
    rm -f ./Pipfile.lock && \
    pip install miscnn==1.4.0 --no-deps && \
    pip install -e . && \
    chmod -R g+w /opt/app-root/lib/python3.9/site-packages && \
    fix-permissions /opt/app-root -P

EXPOSE 8080

ENV DISPLAY=:0

WORKDIR /opt/app-root/src/Project_AIC/scripts/web

CMD ["python", "app.py"]