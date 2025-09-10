# Final, robust Dockerfile using TigerVNC and pre-generated password
FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV USER=root

# --- Main Installation Layer ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget software-properties-common ca-certificates && \
    install -d -m 0755 /etc/apt/keyrings && \
    wget -q https://packages.mozilla.org/apt/repo-signing-key.gpg -O- | tee /etc/apt/keyrings/packages.mozilla.org.asc > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] https://packages.mozilla.org/apt mozilla main" | tee /etc/apt/sources.list.d/mozilla.list > /dev/null && \
    printf 'Package: *\nPin: origin packages.mozilla.org\nPin-Priority: 1000\n' | tee /etc/apt/preferences.d/mozilla && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    firefox-esr \
    python3-pip \
    libopencv-dev python3-opencv \
    python3-grpcio \
    wine64 \
    xdotool \
    dbus-x11 \
    libegl1-mesa \
    libpci3 \
    tigervnc-standalone-server \
    tigervnc-common \
    tigervnc-tools \
    xfonts-base \
    xfce4 \
    xfce4-goodies \
    x11-apps \
    x11-xserver-utils \
    pm-utils \
    curl \
    jq \
    && \
    rm -rf /var/lib/apt/lists/*

# Set font path in tigervncserver script
RUN sed -i 's|^$fontPath =.*|$fontPath = "/usr/share/fonts/X11/misc,/usr/share/fonts/X11/75dpi,/usr/share/fonts/X11/100dpi";|' /usr/bin/tigervncserver

# Install python packages
RUN pip3 install pywinauto opencv-python grpcio tritonclient[all] mss numpy

# APT OCR deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Python libs used by the task runner
RUN pip3 install --no-cache-dir pyyaml pytesseract rapidfuzz requests pynvml

COPY src /scripts

# --- Set VNC password ---
# The original RUN command that called vncpasswd has been REMOVED.
# Instead, we create the directory, COPY the pre-made file, and set permissions.
RUN mkdir -p /root/.vnc
COPY passwd /root/.vnc/passwd
RUN chmod 600 /root/.vnc/passwd

# Create a robust xstartup script for XFCE
RUN printf '#!/bin/sh\nunset SESSION_MANAGER\nunset DBUS_SESSION_BUS_ADDRESS\nexport DISPLAY=:1\n[ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources\nsleep 2\npython3 /scripts/agent.py &\nexec xfce4-session\n' > ~/.vnc/xstartup && \
    chmod +x ~/.vnc/xstartup

# Expose the VNC port
EXPOSE 5901

# Final, robust CMD using TigerVNC's correct syntax
CMD ["/bin/bash", "-c", "tigervncserver :1 -geometry 1024x768 -depth 24 -xstartup /root/.vnc/xstartup -localhost no; sleep 2; tail -f /root/.vnc/*.log"]

