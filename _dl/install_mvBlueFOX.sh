#!/bin/bash
DEF_DIRECTORY=/opt/mvIMPACT_Acquire
DEF_DATA_DIRECTORY=${MVIMPACT_ACQUIRE_DATA_DIR:-/opt/mvIMPACT_Acquire/data}
PRODUCT=mvBlueFOX
API=mvIMPACT_acquire
TARNAME=mvBlueFOX
PACKAGENAME=mvBlueFOX
USE_DEFAULTS=NO
UNATTENDED_INSTALLATION=NO
MINIMAL_INSTALLATION=NO
OS_NAME="unknown"
OS_VERSION="unknown"
OS_CODENAME="unknown"
KERNEL_VERSION="unknown"
REPOSITORY_ACCESS_AVAILABLE=FALSE

APT_GET_EXTRA_PARAMS=""
YUM_EXTRA_PARAMS=""
ZYPPER_EXTRA_PARAMS=""

# Define a variable for the ErrorCount and WarningCount
ERROR_NUMBER=0
WARNING_NUMBER=0

# Define variables for colorized bash output
# Foreground
red=$(tput setaf 1)
yellow=$(tput setaf 3)
green=$(tput setaf 10)
blue=$(tput setaf 12)
bold=$(tput bold)
# Background
greyBG=$(tput setaf 7)
reset=$(tput sgr0)

UDEVADM=$(which udevadm)

# Define the users real name if possible, to prevent accidental mvIA root ownership if script is invoked with sudo
if [ "$(which logname)" == "" ]; then
    USER=$(whoami)
else
    if [ "$(logname 2>&1 | grep -c logname:)" == "1" ]; then
        USER=$(whoami)
    else
        USER=$(logname)
    fi
fi

# If user is root, then sudo shouldn't be used
if [ "$USER" == "root" ]; then
    SUDO=
else
    SUDO=$(which sudo)
fi

# Determine OS Name, version and Kernel version
function check_distro_and_version()
{
    if [ -f /etc/fedora-release ]; then
        OS_NAME='Fedora'
        OS_VERSION=$(cat /etc/redhat-release | sed s/.*release\ // | sed s/\ .*//)
    elif [ -f /etc/redhat-release ]; then
        OS_NAME='RedHat'
        OS_VERSION=$(cat /etc/redhat-release | sed s/.*release\ // | sed s/\ .*//)
    elif [ -f /etc/centos-release ]; then
        OS_NAME='CentOS'
        OS_VERSION=$(cat /etc/centos-release | sed s/.*release\ // | sed s/\ .*//)
    elif [ -f /etc/SuSE-release ]; then
        OS_NAME='SuSE'
        OS_VERSION=$(cat /etc/SuSE-release | tr "\n" ' ' | sed s/.*=\ //)
    elif [ -f /etc/SUSE-brand ]; then
        OS_NAME='SuSE'
        OS_VERSION=$(cat /etc/SUSE-brand | grep VERSION | awk '{ print $3 }'$)
    elif [ -f /etc/mandrake-release ]; then
        OS_NAME='Mandrake'
        OS_VERSION=$(cat /etc/mandrake-release | sed s/.*release\ // | sed s/\ .*//)
    elif [ -x /usr/bin/lsb_release ]; then
        OS_NAME="$(lsb_release -is)" #Ubuntu
        OS_VERSION="$(lsb_release -rs)"
        OS_CODENAME="$(lsb_release -cs)"
    elif [ -f /etc/debian_version ]; then
        OS_NAME="Debian"
        OS_VERSION="$(cat /etc/debian_version)"
    fi
    KERNEL_VERSION=$(uname -r)
    if [[ -v JETSON_KERNEL ]] && [[ -z "$JETSON_KERNEL" ]]; then
        JETSON_KERNEL=$(uname -r | grep tegra)
    fi
}

function checkAPTGETRepositoryAccess()
{
    hosts=$(apt-cache policy | grep -Eo 'http://[^ >]+' | sed -e 's/^http:\/\///g' -e 's/\/.*//' | sort -u)
    host=($hosts)
    HOSTS_TO_BE_PINGED_COUNT=${#host[@]}
    for entry in "${host[@]}"; do
        if [[ $entry != *"microsoft"* ]] && [[ $entry != *"ppa"* ]]; then
            ping -c2 "$entry" >/dev/null 2>&1
            if [ $? -eq 0 ]; then
                ((SUCCESSFULPINGED = SUCCESSFULPINGED + 1))
            fi
        else
            ((HOSTS_TO_BE_PINGED_COUNT = HOSTS_TO_BE_PINGED_COUNT - 1))
        fi
    done
    status=$((SUCCESSFULPINGED - HOSTS_TO_BE_PINGED_COUNT))
    return $status
}

function checkYUMRepositoryAccess()
{
    hosts=$(yum repolist -v | grep Repo-baseurl | awk '{print $3}' | sed -e 's/^http:\/\///g' -e 's/^https:\/\///g' -e 's/^rsync:\/\///g' -e 's/\/.*//' | sort -u)
    host=($hosts)
    HOSTS_TO_BE_PINGED_COUNT=${#host[@]}
    for entry in "${host[@]}"; do
        if [[ $entry != *"microsoft"* ]] && [[ $entry != *"metadata"* ]]; then
            ping -c2 "$entry" >/dev/null 2>&1
            if [ $? -eq 0 ]; then
                ((SUCCESSFULPINGED = SUCCESSFULPINGED + 1))
            fi
        else
            ((HOSTS_TO_BE_PINGED_COUNT = HOSTS_TO_BE_PINGED_COUNT - 1))
        fi
    done
    status=$((SUCCESSFULPINGED - HOSTS_TO_BE_PINGED_COUNT))
    return $status
}

function checkRepositoryAccess()
{
    echo
    echo "Checking repository access. This might take a moment... "
    if [ "$OS_NAME" == "Ubuntu" ] || [ "$OS_NAME" == "Debian" ] || [ "$OS_NAME" == "Linuxmint" ]; then
        if which apt-get >/dev/null 2>&1; then
            checkAPTGETRepositoryAccess
            if [ $? == 0 ]; then
                echo 'Success: apt-get has repository access.'
            else
                return 1
            fi
        fi
    elif [ "$OS_NAME" == "SuSE" ] || [ "$OS_NAME" == "openSUSE" ]; then
        if $SUDO which zypper >/dev/null 2>&1; then
            $SUDO zypper refresh
            if [ $? == 0 ]; then
                echo 'Success: zypper has repository access.'
            else
                return 1
            fi
        fi
    elif [ "$OS_NAME" == "RedHat" ] || [ "$OS_NAME" == "Fedora" ] || [ "$OS_NAME" == "CentOS" ]; then
        if $SUDO which yum >/dev/null 2>&1; then
            checkYUMRepositoryAccess
            if [ $? == 0 ]; then
                echo 'Success: yum has repository access.'
            else
                return 1
            fi
        fi
    fi
    return 0
}

function couldNotInstallMessage()
{
    printf "Could not install the selected package(s) >%s<. Please try to install it manually" "$1"
    ((WARNING_NUMBER = WARNING_NUMBER + 1))
}

function couldNotFindPkgManagerMessage()
{
    printf "Could not find the package manager of the system; please install >%s< manually." "$1"
    ((WARNING_NUMBER = WARNING_NUMBER + 1))
}

function installPackages()
{
    pkgs=("$@")
    # package manager extra parameters
    APT_GET_EXTRA_PARAMS=" -y"
    YUM_EXTRA_PARAMS=" -y"
    ZYPPER_EXTRA_PARAMS="--non-interactive"

    echo "Installing ${pkgs[@]}..."
    if [ "$OS_NAME" == "Ubuntu" ] || [ "$OS_NAME" == "Debian" ] || [ "$OS_NAME" == "Linuxmint" ]; then
        if which apt-get >/dev/null 2>&1; then
            $SUDO apt-get $APT_GET_EXTRA_PARAMS -q install ${pkgs[@]}
            if [ $? != 0 ]; then
                couldNotInstallMessage ${pkgs[@]}
            fi
        else
            couldNotFindPkgManagerMessage ${pkgs[@]}
        fi
    elif [ "$OS_NAME" == "SuSE" ] || [ "$OS_NAME" == "openSUSE" ]; then
        if $SUDO which zypper >/dev/null 2>&1; then
            $SUDO zypper $ZYPPER_EXTRA_PARAMS install ${pkgs[@]}
            if [ $? != 0 ]; then
                couldNotInstallMessage ${pkgs[@]}
            fi
        else
            couldNotFindPkgManagerMessage ${pkgs[@]}
        fi
    elif [ "$OS_NAME" == "RedHat" ] || [ "$OS_NAME" == "Fedora" ] || [ "$OS_NAME" == "CentOS" ]; then
        if $SUDO which yum >/dev/null 2>&1; then
            echo "${pkgs[@]}"
            $SUDO yum $YUM_EXTRA_PARAMS install ${pkgs[@]}
            if [ $? != 0 ]; then
                couldNotInstallMessage ${pkgs[@]}
            fi
        else
            couldNotFindPkgManagerMessage ${pkgs[@]}
        fi
    fi
}

function createSoftlink()
{
    if [ ! -e "$1/$2" ]; then
        echo "Error: File ""$1"/"$2"" does not exist, softlink cannot be created! "
        exit 1
    fi
    if [ -e "$1/$3" ]; then
        rm -rf "$1/$3" >/dev/null 2>&1
    fi
    if ! [ -L "$1/$3" ]; then
        ln -fs "$2" "$1/$3" >/dev/null 2>&1
        if ! [ -L "$1/$3" ]; then
            $SUDO ln -fs "$2" "$1/$3" >/dev/null 2>&1
            if ! [ -L "$1/$3" ]; then
                echo "Error: Could not create softlink $1/$3, even with sudo!"
                exit 1
            fi
        fi
    fi
}

# Print out ASCII-Art Logo.
clear
echo ""
echo ""
echo ""
echo ""
echo "                                ===      ===     MMM~,    M                     "
echo "                                 ==+     ==      M   .M   M                     "
echo "                                 .==    .=+      M    M.  M   M    MM   ~MMM    "
echo "                                  ==+   ==.      MMMMM.   M   M    MM  M:   M   "
echo "             ..                    .== ,==       M   =M   M   M    MM +MMMMMMM  "
echo "   MMMM   DMMMMMM      MMMMMM       =====        M    MM  M   M    MM 7M        "
echo "   MMMM MMMMMMMMMMM :MMMMMMMMMM      ====        M    M+  M   MM  DMM  MM   ,   "
echo "   MMMMMMMMMMMMMMMMMMMMMMMMMMMMM                 IMM+''   M    .M:       =MI    "
echo "   MMMMMMM   .MMMMMMMM    MMMMMM                                                "
echo "   MMMMM.      MMMMMM      MMMMM                                                "
echo "   MMMMM       MMMMM       MMMMM                 MMMMMM    MMMMM    MM.   M     "
echo "   MMMMM       MMMMM       MMMMM                 M       MM    .MM  .M: .M      "
echo "   MMMMM       MMMMM       MMMMM                 M      .M       M~  .M~M       "
echo "   MMMMM       MMMMM       MMMMM                 MMMMMM MM       MD   +MM       "
echo "   MMMMM       MMMMM       MMMMM                 M       M       M    M MM      "
echo "   MMMMM       MMMMM       MMMMM                 M       MM     MM  :M. .M8     "
echo "   MMMMM       MMMMM       MMMMM                 M        .MMMMM    M     MD    "
echo "   MMMMM       MMMMM       MMMMM                                                "

echo ""
echo "=============================================================================="
sleep 1

# Analyze the command line arguments and react accordingly
FORCE_REPOSITORY_USAGE=NO
PATH_EXPECTED=NO
SHOW_HELP=NO
while [[ $# -gt 0 ]]; do
    if [ "$1" == "-f" ] || [ "$1" == "--force" ]; then
        FORCE_REPOSITORY_USAGE=YES
    elif [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        SHOW_HELP=YES
        break
    elif [[ ("$1" == "-u" || "$1" == "--unattended") && "$PATH_EXPECTED" == "NO" ]]; then
        UNATTENDED_INSTALLATION=YES
    elif [[ ("$1" == "-m" || "$1" == "--minimal") && "$PATH_EXPECTED" == "NO" ]]; then
        MINIMAL_INSTALLATION=YES
    elif [[ ("$1" == "-p" || "$1" == "--path") && "$PATH_EXPECTED" == "NO" ]]; then
        if [ "$2" == "" ]; then
            echo
            echo "WARNING: Path option used with no defined path, will use: $DEF_DIRECTORY directory"
            echo
            SHOW_HELP=YES
            break
        else
            PATH_EXPECTED=YES
        fi
    elif [ "$PATH_EXPECTED" == "YES" ]; then
        DEF_DIRECTORY=$1
        PATH_EXPECTED=NO
    else
        echo 'Please check your syntax and try again!'
        SHOW_HELP=YES
    fi
    shift
done

if [ "$SHOW_HELP" == "YES" ]; then
    echo
    echo 'Installation script for the '$PRODUCT' driver.'
    echo
    echo "Default installation path: "$DEF_DIRECTORY
    echo "Usage:                     ./${0##*/} [OPTION] ... "
    echo "Example:                   ./${0##*/} -p /myPath -u"
    echo
    echo "Arguments:"
    echo "-f --force                 Force the installation script to use the package manager of the system to install"
    echo "                           required packages if the installation script reports inaccessible repositories."
    echo "                           e.g. a package source on local media."
    echo "-h --help                  Display this help."
    echo "-p --path                  Set the directory where the files shall be installed."
    echo "-u --unattended            Unattended installation with default settings. By using"
    echo "                           this parameter you explicitly accept the EULA."
    echo "-m --minimal               Minimal installation. No tools or samples will be built, and"
    echo "                           no automatic configuration and/or optimizations will be done."
    echo "                           By using this parameter you explicitly accept the EULA."
    echo
    echo "Note:"
    echo "                           It is possible as well to combine unattended and minimal installation."
    echo "                           Combines the use of default settings from the 'minimal' installation with"
    echo "                           not building tools and samples."
    echo
    exit 1
fi

if [ "$UNATTENDED_INSTALLATION" == "YES" ]; then
    echo
    echo "Unattended installation requested, no user interaction will be required and the"
    echo "default settings will be used."
    echo
    USE_DEFAULTS=YES
fi

if [ "$MINIMAL_INSTALLATION" == "YES" ]; then
    echo
    echo "Minimal installation requested, no user interaction will be required, no tools or samples"
    echo "will be built and no automatic configurations or optimizations will be done."
    echo
    USE_DEFAULTS=YES
fi

# Get some details about the system
check_distro_and_version

# Get the targets platform and if it is called "i686" we know it is a x86 system, else it s x86_64
TARGET=$(uname -m)
if [ "$TARGET" == "i686" ]; then
    TARGET="x86"
fi

# Get the source directory (the directory where the files for the installation are) and cd to it
# (The script file must be in the same directory as the source TGZ) !!!
if which dirname >/dev/null; then
    SCRIPTSOURCEDIR="$(dirname $(realpath "$0"))"
fi
if [ "$SCRIPTSOURCEDIR" != "$PWD" ]; then
    if [ "$SCRIPTSOURCEDIR" == "" ] || [ "$SCRIPTSOURCEDIR" == "." ]; then
        SCRIPTSOURCEDIR="$PWD"
    fi
    cd "$SCRIPTSOURCEDIR" || exit
fi

# Set variables for GenICam and mvIMPACT_acquire for later use
if grep -q '/etc/ld.so.conf.d/' /etc/ld.so.conf; then
    ACQUIRE_LDSOCONF_FILE=/etc/ld.so.conf.d/acquire.conf
else
    ACQUIRE_LDSOCONF_FILE=/etc/ld.so.conf
fi

# Make sure the environment variables are set at the next boot as well
if grep -q '/etc/profile.d/' /etc/profile; then
    ACQUIRE_EXPORT_FILE=/etc/profile.d/acquire.sh
else
    ACQUIRE_EXPORT_FILE=/etc/profile
fi

# Get driver name, version, file. In case of multiple *.tgz files in the folder select the newest version.
if [ "$(ls | grep -c $PACKAGENAME'.*\.tgz')" != "0" ]; then
    TARNAME="$(ls "$PACKAGENAME"*.tgz | tail -1 | sed -e s/\\.tgz//)"
    TARFILE="$(ls "$PACKAGENAME"*.tgz | tail -1)"
    VERSION="$(ls "$PACKAGENAME"*.tgz | tail -1 | sed -e s/\\"$PACKAGENAME"// | sed -e s/\\-"$TARGET"// | sed -e s/\\_ABI2-// | sed -e s/\\.tgz//)"
    ACT=$API-$TARGET-$VERSION
    ACT2=$ACT
fi

# Check if tar-file is correct for the system architecture
if [ "$TARGET" == "x86_64" ]; then
    if [ "$(echo $TARNAME | grep -c x86_ABI2)" != "0" ]; then
        echo "-----------------------------------------------------------------------------------"
        echo "  ABORTING: Attempt to install 32-bit drivers in a 64-bit machine!  "
        echo "-----------------------------------------------------------------------------------"
        exit
    fi
fi
if [ "$TARGET" == "x86" ]; then
    if [ "$(echo $TARNAME | grep -c x86_64_ABI2)" != "0" ]; then
        echo "-----------------------------------------------------------------------------------"
        echo "  ABORTING: Attempt to install 64-bit drivers in a 32-bit machine!  "
        echo "-----------------------------------------------------------------------------------"
        exit
    fi
fi

# A quick check whether the Version has a correct format (due to other files being in the same directory..?)
if [ "$(echo "$VERSION" | grep -c '^[0-9]\{1,2\}\.[0-9]\{1,2\}\.[0-9]\{1,2\}')" == "0" ]; then
    echo "-----------------------------------------------------------------------------------"
    echo "${red}  ABORTING: Script could not determine a valid $PRODUCT *.tgz file!  "
    echo "${reset}-----------------------------------------------------------------------------------"
    echo "  This script could not extract a valid version number from the *.tgz file"
    echo "  This script determined $TARFILE as the file containing the installation data."
    echo "  It is recommended that only this script and the correct *.tgz file reside in this directory."
    echo "  Please remove all other files and try again."
    exit 1
fi

# A quick check whether the user has been determined
if [ "$USER" == "" ]; then
    echo "-----------------------------------------------------------------------------------"
    echo "${red}  ABORTING: Script could not determine a valid user!  ${reset}"
    echo "-----------------------------------------------------------------------------------"
    echo "  This script could not determine the user of this shell"
    echo "  Please make sure this is a valid login shell!"
    exit 1
fi

YES_NO=
# Ask whether to use the defaults or proceed with an interactive installation
if [ "$UNATTENDED_INSTALLATION" == "NO" ] && [ "$MINIMAL_INSTALLATION" == "NO" ]; then
    echo
    echo "Would you like this installation to run in unattended mode?"
    echo "Using this mode you explicitly agree to the EULA(End User License Agreement)!"
    echo "No user interaction will be required, and the default settings will be used!"
    echo "Hit 'n' + <Enter> for 'no', or just <Enter> for 'yes'."
    read YES_NO
else
    YES_NO=""
fi
if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
    USE_DEFAULTS=NO
else
    USE_DEFAULTS=YES
fi

YES_NO=
# Check if package manager has access to repositories
if [ "$MINIMAL_INSTALLATION" == "NO" ]; then
    checkRepositoryAccess
    if [ $? == 0 ]; then
        REPOSITORY_ACCESS_AVAILABLE="YES"
    else
        echo "-----------------------------------------------------------------------------------"
        echo "${yellow}  WARNING: There seems to be no repository access!${reset}                "
        echo "-----------------------------------------------------------------------------------"
        echo "  This script could not get access to the system's package repositories."
        echo "  If additional packages are requiered during installation e.g.: g++,"
        echo "  wxWidgets, expat, FLTK, ... some features might not be available."
        echo
        echo "  In certain cases this might be a result of specific firewall settings or local   "
        echo "  package repositores. It is possible to let the script try to use those package   "
        echo "  repositories even if the access could not be verified.                           "
        echo
        echo "  Would you like this installation to force the usage of the package manager to    "
        echo "  install some additional required packages?"
        echo "  Hit 'y' + <Enter> for 'yes', or just <Enter> for 'no'."

        if [ "$USE_DEFAULTS" == "NO" ]; then
            read YES_NO
        else
            YES_NO=""
        fi
        if [ "$YES_NO" == "y" ] || [ "$YES_NO" == "Y" ]; then
            FORCE_REPOSITORY_USAGE="YES"
        fi
    fi
fi

YES_NO=
# Here we will ask the user if we shall start the installation process
echo
echo "-----------------------------------------------------------------------------------"
echo "${bold}Host System:${reset}"
echo "-----------------------------------------------------------------------------------"
echo
echo "${bold}OS:                             ${reset}"$OS_NAME
echo "${bold}OS Version:                     ${reset}""$OS_VERSION"
echo "${bold}OS Codename:                    ${reset}""$OS_CODENAME"
echo "${bold}Kernel:                         ${reset}""$KERNEL_VERSION"
echo "${bold}Platform:                       ${reset}""$TARGET"
echo
echo "-----------------------------------------------------------------------------------"
echo "${bold}Configuration:${reset}"
echo "-----------------------------------------------------------------------------------"
echo
echo "${bold}Installation for user:            ${reset}""$USER"
echo "${bold}Installation directory:           ${reset}"$DEF_DIRECTORY
echo "${bold}Data directory:                   ${reset}""$DEF_DATA_DIRECTORY"
echo "${bold}Source directory:                 ${reset}"$(echo "$SCRIPTSOURCEDIR" | sed -e 's/\/\.//')
echo "${bold}Version:                          ${reset}""$VERSION"
echo "${bold}TAR-File:                         ${reset}""$TARFILE"
echo
echo "${bold}ldconfig:"
echo "${bold}mvIMPACT_acquire:                 ${reset}"$ACQUIRE_LDSOCONF_FILE
echo
echo "${bold}Exports:"
echo "${bold}mvIMPACT_acquire:                 ${reset}"$ACQUIRE_EXPORT_FILE

echo
echo "-----------------------------------------------------------------------------------"
echo
echo "Do you want to continue (default is 'yes')?"
echo "Hit 'n' + <Enter> for 'no', or just <Enter> for 'yes'."
if [ "$USE_DEFAULTS" == "NO" ]; then
    read YES_NO
else
    YES_NO=""
fi
echo

# If the user is choosing no, we will abort the installation, else we will start the process.
if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
    echo "Quit!"
    exit
fi

# End User License Agreement
YES_NO="r"
while [ "$YES_NO" == "r" ] || [ "$YES_NO" == "R" ]; do
    echo
    echo "Do you accept the End User License Agreement (default is 'yes')?"
    echo "Hit 'n' + <Enter> for 'no', 'r' + <Enter> to read the EULA or "
    echo "just <Enter> for 'yes'."
    if [ "$USE_DEFAULTS" == "NO" ]; then
        read YES_NO
        if [ "$YES_NO" == "r" ] || [ "$YES_NO" == "R" ]; then
            if [ "x$(which more)" != "x" ]; then
                EULA_SHOW_COMMAND="more -d"
            else
                EULA_SHOW_COMMAND="cat"
            fi
            tar -xzf "$TARFILE" -C /tmp mvIMPACT_acquire-"$VERSION".tar && tar -xf /tmp/mvIMPACT_acquire-"$VERSION".tar -C /tmp mvIMPACT_acquire-"$VERSION"/doc/EULA.txt --strip-components=2 && rm /tmp/mvIMPACT_acquire-"$VERSION".tar && $EULA_SHOW_COMMAND /tmp/EULA.txt && rm /tmp/EULA.txt && sleep 1
            # clear the stdin buffer in case user spammed the Enter key
            while read -r -t 0; do read -r; done
        fi
    else
        YES_NO=""
    fi
done

# If the user is choosing no, we will abort the installation, else we continue.
if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
    echo "Quit!"
    exit
fi

echo
echo "-----------------------------------------------------------------------------------"
echo "${bold}BY INSTALLING THIS SOFTWARE YOU HAVE AGREED TO THE EULA(END USER LICENSE AGREEMENT)${reset}"
echo "-----------------------------------------------------------------------------------"
echo

# First of all ask whether to dispose of the old mvIMPACT Acquire installation
if [ "$MVIMPACT_ACQUIRE_DIR" != "" ]; then
    echo "Existing installation detected at: $MVIMPACT_ACQUIRE_DIR"
    echo "Do you want to keep this installation (default is 'yes')?"
    echo "If you select no, mvIMPACT Acquire will be removed for ALL installed products!"
    echo "Hit 'n' + <Enter> for 'no', or just <Enter> for 'yes'."
    if [ "$USE_DEFAULTS" == "NO" ]; then
        read YES_NO
    else
        YES_NO=""
    fi
    echo
    if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
        $SUDO rm -f /usr/bin/mvDeviceConfigure >/dev/null 2>&1
        $SUDO rm -f /usr/bin/mvIPConfigure >/dev/null 2>&1
        $SUDO rm -f /usr/bin/wxPropView >/dev/null 2>&1
        $SUDO rm -f /etc/ld.so.conf.d/acquire.conf >/dev/null 2>&1
        $SUDO rm -f /etc/ld.so.conf.d/genicam.conf >/dev/null 2>&1
        $SUDO rm -f /etc/profile.d/acquire.sh >/dev/null 2>&1
        $SUDO rm -f /etc/profile.d/genicam.sh >/dev/null 2>&1
        $SUDO rm -f /etc/udev/rules.d/51-mvbf.rules >/dev/null 2>&1
        $SUDO rm -f /etc/udev/rules.d/52-U3V.rules >/dev/null 2>&1
        $SUDO rm -f /etc/udev/rules.d/52-mvbf3.rules >/dev/null 2>&1
        $SUDO rm -f /etc/udev/rules.d/51-udev-pcie.rules >/dev/null 2>&1
        $SUDO rm -f /usr/local/bin/make_device.sh 2>&1
        $SUDO rm -f /usr/local/bin/device_namer.sh 2>&1
        $SUDO rm -f /etc/sysctl.d/62-buffers-performance.conf >/dev/null 2>&1
        $SUDO rm -f /etc/security/limits.d/acquire.conf >/dev/null 2>&1
        $SUDO rm -rf /etc/matrix-vision >/dev/null >/dev/null 2>&1
        $SUDO rm -rf "$MVIMPACT_ACQUIRE_DIR" >/dev/null 2>&1
        if [ $? == 0 ]; then
            echo "Previous mvIMPACT Acquire Installation ($MVIMPACT_ACQUIRE_DIR) removed successfully!"
        else
            echo "Error removing previous mvIMPACT Acquire Installation ($MVIMPACT_ACQUIRE_DIR)!"
            echo "$?"
        fi
        if [ "$INIT_SYSTEM" == "systemd" ] && [ -f "/etc/systemd/system/resize_usbfs_buffersize.service" ]; then
            $SUDO systemctl stop resize_usbfs_buffersize.service
            $SUDO systemctl disable resize_usbfs_buffersize.service
            $SUDO rm -f /etc/systemd/system/resize_usbfs_buffersize.service
            if [ $? == 0 ]; then
                echo "usbcore.usbfs_memory_mb systemd service (/etc/systemd/system/resize_usbfs_buffersize.service) disabled and removed successfully!"
            else
                echo "Error removing usbcore.usbfs_memory_mb systemd service (/etc/systemd/system/resize_usbfs_buffersize.service)!"
                echo "$?"
            fi
        fi
    else
        echo "Previous mvIMPACT Acquire Installation ($MVIMPACT_ACQUIRE_DIR) NOT removed!"
    fi
fi

# Create the *.conf files if the system is supporting ld.so.conf.d
if grep -q '/etc/ld.so.conf.d/' /etc/ld.so.conf; then
    $SUDO rm -f $ACQUIRE_LDSOCONF_FILE
    $SUDO touch $ACQUIRE_LDSOCONF_FILE
fi

# Create the export files if the system is supporting profile.d
if grep -q '/etc/profile.d/' /etc/profile; then
    $SUDO rm -f $ACQUIRE_EXPORT_FILE
    $SUDO touch $ACQUIRE_EXPORT_FILE
fi

# Check if the destination directory exist, else create it
if ! [ -d $DEF_DIRECTORY ]; then
    # the destination directory does not yet exist
    # first try to create it as a normal user
    mkdir -p $DEF_DIRECTORY >/dev/null 2>&1
    if ! [ -d $DEF_DIRECTORY ]; then
        # that didn't work
        # now try it as superuser
        $SUDO mkdir -p $DEF_DIRECTORY
    fi
    if ! [ -d $DEF_DIRECTORY ]; then
        echo 'ERROR: Could not create target directory' $DEF_DIRECTORY '.'
        echo 'Problem:'$?
        echo 'Maybe you specified a partition that was mounted read only?'
        echo
        exit 1
    fi
else
    echo 'Installation directory already exists.'
fi

# in case the directory already existed BUT it belongs to other user
$SUDO chown -R "$USER": $DEF_DIRECTORY

# Check the actual tarfile
if ! [ -r "$TARFILE" ]; then
    echo 'ERROR: could not read' "$TARFILE".
    echo
    exit 1
fi

# needed at compile time (used during development, but not shipped with the final program)
ACT=$API-$VERSION.tar

# Now unpack the tarfile into /tmp
cd /tmp
tar xfz "$SCRIPTSOURCEDIR/$TARFILE"

# Now check if we can unpack the tar file with the device independent stuff
# this is entirely optional
if [ -r /tmp/$ACT2 ]; then
    cd /tmp
    #tar xvf /tmp/$ACT
    cp -r $ACT2/* $DEF_DIRECTORY
else
    echo
    echo "ERROR: Could not read: /tmp/"$ACT2
    exit
fi

# Set the necessary exports and library paths
cd $DEF_DIRECTORY || exit
if grep -q 'MVIMPACT_ACQUIRE_DIR=' "$ACQUIRE_EXPORT_FILE"; then
    echo 'MVIMPACT_ACQUIRE_DIR already defined in' "$ACQUIRE_EXPORT_FILE".
else
    $SUDO sh -c "echo 'export MVIMPACT_ACQUIRE_DIR=$DEF_DIRECTORY' >> $ACQUIRE_EXPORT_FILE"
fi

if grep -q "$DEF_DIRECTORY/lib/$TARGET" $ACQUIRE_LDSOCONF_FILE; then
    echo "$DEF_DIRECTORY/lib/$TARGET already defined in" $ACQUIRE_LDSOCONF_FILE.
else
    $SUDO sh -c "echo '$DEF_DIRECTORY/lib/$TARGET' >> $ACQUIRE_LDSOCONF_FILE"
fi
if grep -q "$DEF_DIRECTORY/Toolkits/expat/bin/$TARGET/lib" $ACQUIRE_LDSOCONF_FILE; then
    echo "$DEF_DIRECTORY/Toolkits/expat/bin/$TARGET/lib already defined in" $ACQUIRE_LDSOCONF_FILE.
else
    $SUDO sh -c "echo '$DEF_DIRECTORY/Toolkits/expat/bin/$TARGET/lib' >> $ACQUIRE_LDSOCONF_FILE"
fi

# This variable must be exported, or else wxPropView-related make problems can arise
export MVIMPACT_ACQUIRE_DIR=$DEF_DIRECTORY

# Clean up /tmp
rm -r -f /tmp/$ACT2 /tmp/$API-$VERSION

# create softlinks for the Toolkits libraries
createSoftlink $DEF_DIRECTORY/Toolkits/expat/bin/$TARGET/lib libexpat.so.0.5.0 libexpat.so.0
createSoftlink $DEF_DIRECTORY/Toolkits/expat/bin/$TARGET/lib libexpat.so.0 libexpat.so
createSoftlink $DEF_DIRECTORY/Toolkits/FreeImage3160/bin/Release/FreeImage/$TARGET libfreeimage-3.16.0.so libfreeimage.so.3
createSoftlink $DEF_DIRECTORY/Toolkits/FreeImage3160/bin/Release/FreeImage/$TARGET libfreeimage.so.3 libfreeimage.so

# Update the library cache with ldconfig
$SUDO /sbin/ldconfig

if [ "$MINIMAL_INSTALLATION" == "NO" ]; then
    if ([ "$REPOSITORY_ACCESS_AVAILABLE" == "YES" ] || [ "$FORCE_REPOSITORY_USAGE" == "YES" ]) || [ "$USE_DEFAULTS" == "YES" ]; then
        ADDITIONAL_PACKAGE_TO_BE_INSTALLED=""

        # Install needed libraries and compiler
        DISTRO_NOT_SUPPORTED="Your Linux distribution is currently not officially supported by this installation script. Please install required package >%s< manually."

        # Check if we have g++
        if ! which g++ >/dev/null 2>&1; then
            ADDITIONAL_PACKAGE_TO_BE_INSTALLED="g++"
            echo
            echo "A native g++ compiler is required to build the sample applications."
            echo "Currently no g++ compiler is detected on the system. Do you want to install it?"
            echo "Hit 'n' + <Enter> for 'no', or just <Enter> for 'yes'."
            if [ "$USE_DEFAULTS" == "NO" ]; then
                read YES_NO
            else
                YES_NO=""
            fi
            echo
            if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
                printf "The >%s< compiler will not be installed." "$ADDITIONAL_PACKAGE_TO_BE_INSTALLED"
            else
                if [ "$OS_NAME" == "Ubuntu" ] || [ "$OS_NAME" == "Debian" ] || [ "$OS_NAME" == "Linuxmint" ]; then
                    REQUIRED_PACKAGES=(g++)
                    $SUDO apt-get update
                    installPackages "${REQUIRED_PACKAGES[@]}"
                elif [ "$OS_NAME" == "SuSE" ] || [ "$OS_NAME" == "openSUSE" ]; then
                    REQUIRED_PACKAGES=(gcc-c++ make)
                    installPackages "${REQUIRED_PACKAGES[@]}"
                elif [ "$OS_NAME" == "RedHat" ] || [ "$OS_NAME" == "Fedora" ] || [ "$OS_NAME" == "CentOS" ]; then
                    REQUIRED_PACKAGES=(gcc gcc-c++)
                    installPackages "${REQUIRED_PACKAGES[@]}"
                else
                    printf "$DISTRO_NOT_SUPPORTED" "${REQUIRED_PACKAGES[@]}"
                    ((WARNING_NUMBER = WARNING_NUMBER + 1))
                fi
            fi
        fi

        # Check if we have expat-devel
        if [ "$(whereis libexpat 2>&1)" == "libexpat:" ]; then
            echo
            ADDITIONAL_PACKAGE_TO_BE_INSTALLED="libexpat"
            echo "libexpat is required to parse the XML files."
            echo "Currently it is not present on the system. Do you want to install it?"
            echo "Hit 'n' + <Enter> for 'no', or just <Enter> for 'yes'."
            if [ "$USE_DEFAULTS" == "NO" ]; then
                read YES_NO
            else
                YES_NO=""
            fi
            echo
            if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
                printf ">%s< will not be installed." "$ADDITIONAL_PACKAGE_TO_BE_INSTALLED"
            else
                if [ "$OS_NAME" == "Ubuntu" ] || [ "$OS_NAME" == "Debian" ] || [ "$OS_NAME" == "Linuxmint" ]; then
                    REQUIRED_PACKAGES=(libexpat1-dev)
                    installPackages "${REQUIRED_PACKAGES[@]}"
                elif [ "$OS_NAME" == "SuSE" ] || [ "$OS_NAME" == "openSUSE" ]; then
                    REQUIRED_PACKAGES=(expat libexpat-devel)
                    installPackages "${REQUIRED_PACKAGES[@]}"
                elif [ "$OS_NAME" == "RedHat" ] || [ "$OS_NAME" == "Fedora" ] || [ "$OS_NAME" == "CentOS" ]; then
                    REQUIRED_PACKAGES=(expat-devel)
                    installPackages "${REQUIRED_PACKAGES[@]}"
                else
                    echo "$DISTRO_NOT_SUPPORTED $ADDITIONAL_PACKAGE_TO_BE_INSTALLED"
                    ((WARNING_NUMBER = WARNING_NUMBER + 1))
                fi
            fi
        fi

        INPUT_REQUEST="Do you want to install >%s< (default is 'yes')?\nHit 'n' + <Enter> for 'no', or just <Enter> for 'yes'.\n"

        # Do we want to install wxWidgets?
        if [ "$(wx-config --release 2>&1 | grep -c "^3.")" != "1" ] ||
            [ "$(wx-config --libs 2>&1 | grep -c "webview")" != "1" ] ||
            [ "$(wx-config --selected-config 2>&1 | grep -c "gtk3")" == "0" ]; then
            echo
            ADDITIONAL_PACKAGE_TO_BE_INSTALLED="wxWidgets"
            printf "$INPUT_REQUEST" "$ADDITIONAL_PACKAGE_TO_BE_INSTALLED"
            echo "This is highly recommended, as without wxWidgets, you cannot build wxPropView."
            echo
            if [ "$USE_DEFAULTS" == "NO" ]; then
                read YES_NO
            else
                YES_NO=""
            fi
            if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
                echo "Not installing $ADDITIONAL_PACKAGE_TO_BE_INSTALLED"
            else
                if [ "$OS_NAME" == "Ubuntu" ] || [ "$OS_NAME" == "Debian" ] || [ "$OS_NAME" == "Linuxmint" ]; then
                    REQUIRED_PACKAGES=(libwxgtk-webview3.0-gtk3-* libwxgtk3.0-gtk3-* wx3.0-headers build-essential libgtk2.0-dev)
                    installPackages "${REQUIRED_PACKAGES[@]}"
                    if [ "$(update-alternatives --list wx-config | grep -c "gtk3")" == "1" ]; then
                        echo "Using GTK3 wxWidgets as default during this installation."
                        $SUDO update-alternatives --set wx-config $(update-alternatives --list wx-config | grep gtk3)
                        WXCONFIG_RESET_NEEDED="YES"
                    fi
                elif [ "$OS_NAME" == "SuSE" ] || [ "$OS_NAME" == "openSUSE" ]; then
                    REQUIRED_PACKAGES=(wxWidgets-3_2-devel)
                    installPackages "${REQUIRED_PACKAGES[@]}"
                elif [ "$OS_NAME" == "RedHat" ] || [ "$OS_NAME" == "Fedora" ] || [ "$OS_NAME" == "CentOS" ]; then
                    REQUIRED_PACKAGES=(wxGTK3 wxGTK3-devel)
                    installPackages "${REQUIRED_PACKAGES[@]}"
                else
                    echo "$DISTRO_NOT_SUPPORTED $ADDITIONAL_PACKAGE_TO_BE_INSTALLED"
                    ((WARNING_NUMBER = WARNING_NUMBER + 1))
                fi
            fi
        fi

        # Do we want to install FLTK?
        if ! which fltk-config >/dev/null 2>&1; then
            ADDITIONAL_PACKAGE_TO_BE_INSTALLED="FLTK"
            echo
            printf "$INPUT_REQUEST" "$ADDITIONAL_PACKAGE_TO_BE_INSTALLED"
            echo "This is only required if you want to build the 'ContinuousCaptureFLTK' sample."
            echo
            if [ "$USE_DEFAULTS" == "NO" ]; then
                read YES_NO
            else
                YES_NO=""
            fi
            if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
                echo "Not installing $ADDITIONAL_PACKAGE_TO_BE_INSTALLED"
            else
                if [ "$OS_NAME" == "Ubuntu" ] || [ "$OS_NAME" == "Debian" ] || [ "$OS_NAME" == "Linuxmint" ]; then
                    OS_VERSION="20.04"
                    OS_VERSION_TOKEN=(${OS_VERSION//./ })
                    if [ "$OS_NAME" == "Ubuntu" ] && [ "${OS_VERSION_TOKEN[0]}" -ge "20" ]; then
                        REQUIRED_PACKAGES=(libfltk1.3-dev)
                    else
                        REQUIRED_PACKAGES=(libgl1-mesa-dev libfltk1.1-dev)
                    fi
                    installPackages "${REQUIRED_PACKAGES[@]}"
                elif [ "$OS_NAME" == "SuSE" ] || [ "$OS_NAME" == "openSUSE" ]; then
                    REQUIRED_PACKAGES=(Mesa-devel fltk fltk-devel)
                    installPackages "${REQUIRED_PACKAGES[@]}"
                elif [ "$OS_NAME" == "RedHat" ] || [ "$OS_NAME" == "Fedora" ] || [ "$OS_NAME" == "CentOS" ]; then
                    REQUIRED_PACKAGES=("fltk" "fltk-devel")
                    installPackages "${REQUIRED_PACKAGES[@]}"
                else
                    echo "$DISTRO_NOT_SUPPORTED $ADDITIONAL_PACKAGE_TO_BE_INSTALLED"
                    ((WARNING_NUMBER = WARNING_NUMBER + 1))
                fi
            fi
        fi

        # In case we are compiling on a SuSE flavour, remove mvDeviceConfigure because wxWidgets is no longer compatible
        if [ "$OS_NAME" == "SuSE" ] || [ "$OS_NAME" == "openSUSE" ]; then
            if [ -d $DEF_DIRECTORY/apps/mvDeviceConfigure ] && [ -r $DEF_DIRECTORY/apps/mvDeviceConfigure/Makefile ]; then
                $SUDO rm -rf $DEF_DIRECTORY/apps/mvDeviceConfigure >/dev/null 2>&1
            fi
            if [ "$MINIMAL_INSTALLATION" == "NO" ]; then
                echo "The mvDeviceConfigure application will not be built because your distribution does not supply all the necessary wxWidget libraries."
            fi
        fi

        if [ "$MINIMAL_INSTALLATION" == "NO" ]; then
            echo
            echo "Do you want the tools and samples to be built (default is 'yes')?"
            echo "Hit 'n' + <Enter> for 'no', or just <Enter> for 'yes'."
            if [ "$USE_DEFAULTS" == "NO" ]; then
                read YES_NO
            else
                YES_NO=""
            fi
            if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
                echo
                echo "The tools and samples were not built."
                echo "To build them manually run 'make native' in $DEF_DIRECTORY"
            else
                cd $DEF_DIRECTORY || exit
                $SUDO chown -R "$USER": $DEF_DIRECTORY
                make $TARGET

                if [ $? -ne 0 ]; then
                    ((WARNING_NUMBER = WARNING_NUMBER + 1))
                fi

                # Shall the MV Tools be linked in /usr/bin?
                if [ "$GEV_SUPPORT" == "TRUE" ]; then
                    if [ "$OS_NAME" == "SuSE" ] || [ "$OS_NAME" == "openSUSE" ]; then
                        GUI_APPS="wxPropView and mvIPConfigure"
                    else
                        GUI_APPS="wxPropView, mvIPConfigure and mvDeviceConfigure"
                    fi
                else
                    if [ "$OS_NAME" == "SuSE" ] || [ "$OS_NAME" == "openSUSE" ]; then
                        GUI_APPS="wxPropView and mvIPConfigure"
                    else
                        GUI_APPS="wxPropView and mvDeviceConfigure"
                    fi
                fi
                echo "Do you want to set a link to /usr/bin for $GUI_APPS (default is 'yes')?"
                echo "Hit 'n' + <Enter> for 'no', or just <Enter> for 'yes'."
                if [ "$USE_DEFAULTS" == "NO" ]; then
                    read YES_NO
                else
                    YES_NO=""
                fi
                if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
                    echo "Will not set any new link to /usr/bin."
                else
                    if [ -r /usr/bin ]; then
                        # Set wxPropView
                        if [ -r "$DEF_DIRECTORY"/apps/mvPropView/"$TARGET"/wxPropView ]; then
                            $SUDO rm -f /usr/bin/wxPropView
                            $SUDO ln -s "$DEF_DIRECTORY"/apps/mvPropView/"$TARGET"/wxPropView /usr/bin/wxPropView
                        fi
                        # Set mvDeviceConfigure
                        if [ -r "$DEF_DIRECTORY"/apps/mvDeviceConfigure/"$TARGET"/mvDeviceConfigure ]; then
                            $SUDO rm -f /usr/bin/mvDeviceConfigure
                            $SUDO ln -s "$DEF_DIRECTORY"/apps/mvDeviceConfigure/"$TARGET"/mvDeviceConfigure /usr/bin/mvDeviceConfigure
                        fi
                    fi

                fi
            fi

            # Should wxPropView check weekly for updates?
            echo "Do you want wxPropView to check for updates weekly(default is 'yes')?"
            echo "Hit 'n' + <Enter> for 'no', or just <Enter> for 'yes'."
            if [ "$USE_DEFAULTS" == "NO" ]; then
                read YES_NO
            else
                YES_NO=""
            fi
            if [ ! -e ~/.wxPropView ]; then
                touch ~/.wxPropView
            fi
            if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
                if [ "$(grep -c AutoCheckForUpdatesWeekly ~/.wxPropView)" -ne "0" ]; then
                    Tweakline=$(($(grep -n "AutoCheckForUpdatesWeekly" ~/.wxPropView | cut -d: -f1))) && sed -i "$Tweakline s/.*/AutoCheckForUpdatesWeekly=0/" ~/.wxPropView
                else
                    echo "AutoCheckForUpdatesWeekly=0" >>~/.wxPropView
                fi
            else
                if [ "$(grep -c AutoCheckForUpdatesWeekly ~/.wxPropView)" -ne "0" ]; then
                    Tweakline=$(($(grep -n "AutoCheckForUpdatesWeekly" ~/.wxPropView | cut -d: -f1))) && sed -i "$Tweakline s/.*/AutoCheckForUpdatesWeekly=1/" ~/.wxPropView
                else
                    echo "[MainFrame/Help]" >>~/.wxPropView
                    echo "AutoCheckForUpdatesWeekly=1" >>~/.wxPropView
                fi
            fi
        fi
    fi
fi

echo
echo "Do you want to copy 51-mvbf.rules to /etc/udev/rules.d for non-root user support (default is 'yes')?"
echo "Hit 'n' + <Enter> for 'no', or just <Enter> for 'yes'."
if [ "$USE_DEFAULTS" == "NO" ]; then
    read YES_NO
else
    YES_NO=""
fi
if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
    echo
    echo 'To grant non-root user support,'
    echo 'copy 51-mvbf.rules the file to /etc/udev/rules.d'
    echo
else
    $SUDO cp -f $DEF_DIRECTORY/Scripts/51-mvbf.rules /etc/udev/rules.d
fi
echo

# reload the udev rules after change.
$SUDO $UDEVADM control --reload-rules

# Update the library cache again.
$SUDO /sbin/ldconfig

# check if plugdev group exists and the user is member of it
if [ "$(grep -c ^plugdev: /etc/group)" == "0" ]; then
    echo "Group 'plugdev' doesn't exist, this is necessary to use USB devices as a normal user!"
    echo "Do you want to create it and add current user to 'plugdev' (default is 'yes')?"
    echo "Hit 'n' + <Enter> for 'no', or just <Enter> for 'yes'."
    if [ "$USE_DEFAULTS" == "NO" ]; then
        read YES_NO
    else
        YES_NO=""
    fi
    if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
        echo
        echo "'plugdev' will not be created and you can't run the device as non-root user!"
        echo "If you want non-root user support, you will need to create 'plugdev'"
        echo "and add these users to this group."
        ((WARNING_NUMBER = WARNING_NUMBER + 1))
    else
        $SUDO /usr/sbin/groupadd -g 46 plugdev
        $SUDO /usr/sbin/usermod -a -G plugdev $USER
        echo "Group 'plugdev' created and user '"$USER"' added to it."
    fi
else
    if [ "$(groups | grep -c plugdev)" == "0" ]; then
        echo "Group 'plugdev' exists, however user '"$USER"' is not a member, which is necessary to"
        echo "use USB devices. Do you want to add user '"$USER"' to 'plugdev' (default is 'yes')?"
        echo "Hit 'n' + <Enter> for 'no', or just <Enter> for 'yes'."
        if [ "$USE_DEFAULTS" == "NO" ]; then
            read YES_NO
        else
            YES_NO=""
        fi
        if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
            echo
            echo "If you want to use USB devices you have to manually add user '"$USER"' to the plugdev group."
            ((WARNING_NUMBER = WARNING_NUMBER + 1))
        else
            $SUDO /usr/sbin/usermod -a -G plugdev $USER
            echo "User '"$USER"' added to 'plugdev' group."
        fi
    fi
fi
echo

# create the logs directory and set MVIMPACT_ACQUIRE_DATA_DIR.
if ! [ -d "$DEF_DATA_DIRECTORY"/logs ]; then
    mkdir -p "$DEF_DATA_DIRECTORY"/logs >/dev/null 2>&1
    if ! [ -d "$DEF_DATA_DIRECTORY"/logs ]; then
        # that didn't work, now try it as superuser
        $SUDO mkdir -p "$DEF_DATA_DIRECTORY"/logs >/dev/null 2>&1
    fi
fi

if [ -d "$DEF_DATA_DIRECTORY"/logs ]; then
    # exec (1) and write (2) are needed to create new files and reading (4) should be allowed anyway
    # therefore the permissions are set to 777, so any user is able to read and write logs
    $SUDO chmod 777 "$DEF_DATA_DIRECTORY"/logs
    mv $DEF_DIRECTORY/apps/mvDebugFlags.mvd "$DEF_DATA_DIRECTORY"/logs >/dev/null 2>&1
    if ! [ -r "$DEF_DATA_DIRECTORY"/logs/mvDebugFlags.mvd ]; then
        $SUDO mv $DEF_DIRECTORY/apps/mvDebugFlags.mvd "$DEF_DATA_DIRECTORY"/logs >/dev/null 2>&1
    fi
    if grep -q 'MVIMPACT_ACQUIRE_DATA_DIR=' "$ACQUIRE_EXPORT_FILE"; then
        echo 'MVIMPACT_ACQUIRE_DATA_DIR already defined in' "$ACQUIRE_EXPORT_FILE".
    else
        $SUDO sh -c "echo 'export MVIMPACT_ACQUIRE_DATA_DIR=$DEF_DATA_DIRECTORY' >> $ACQUIRE_EXPORT_FILE"
    fi
else
    echo "ERROR: Could not create " "$DEF_DATA_DIRECTORY"/logs " directory."
    echo 'Problem:'$?
    echo 'Maybe you specified a partition that was mounted read only?'
    echo
    exit 1
fi

# Configure the /etc/security/limits.d/acquire.conf file to be able to set thread priorities
if [ -d /etc/security/limits.d ]; then
    if [[ ! -f /etc/security/limits.d/acquire.conf || "$(grep -c '@plugdev            -       nice            -20' /etc/security/limits.d/acquire.conf)" == "0" ]]; then
        echo '@plugdev            -       nice            -20' | sudo tee -a /etc/security/limits.d/acquire.conf >/dev/null
    fi
    if [ "$(grep -c '@plugdev            -       rtprio          99' /etc/security/limits.d/acquire.conf)" == "0" ]; then
        echo '@plugdev            -       rtprio          99' | sudo tee -a /etc/security/limits.d/acquire.conf >/dev/null
    fi
else
    echo 'INFO: Directory /etc/security/limits.d is missing, mvIMPACT Acquire will not'
    echo 'be able to set thread priorities correctly. Incomplete frames may occur!!!'
    ((WARNING_NUMBER = WARNING_NUMBER + 1))
fi

# remove all example application sources in case of minimal installation
if [ "$MINIMAL_INSTALLATION" == "YES" ]; then
    $SUDO rm -rf $DEF_DIRECTORY/apps >/dev/null 2>&1
fi

# resetting wxWidgets configuration to the auto default
if [ $WXCONFIG_RESET_NEEDED ]; then
    echo "Resetting wxWidgets configuration to the auto default."
    $SUDO update-alternatives --auto wx-config
fi

echo
if [ "$ERROR_NUMBER" == 0 ] && [ "$WARNING_NUMBER" == 0 ]; then
    echo "-----------------------------------------------------------------------------------"
    echo "${green}${bold}                           Installation successful!${reset}         "
    echo "-----------------------------------------------------------------------------------"
elif [ "$ERROR_NUMBER" == 0 ] && [ "$WARNING_NUMBER" != 0 ]; then
    echo "-----------------------------------------------------------------------------------"
    echo "${yellow}${bold}                           Installation successful!${reset}        "
    echo "-----------------------------------------------------------------------------------"
    echo "                                                                                   "
    echo "  Some warnings have been issued during the installation. Typically the driver     "
    echo "  will work, but some functionality is missing e.g. some sample applications       "
    echo "  which could not be built because of missing dependencies.                        "

    echo "                                                                                   "
    echo "  Please refer to the output of the script for further details.                    "
    echo "-----------------------------------------------------------------------------------"
else
    echo "-----------------------------------------------------------------------------------"
    echo "${red}${bold}                        Installation NOT successful!${reset}          "
    echo "-----------------------------------------------------------------------------------"
    echo "                                                                                   "
    echo "  Please provide the full output of this installation script to the MATRIX VISION  "
    echo "  support department if the error messages shown during the installation procedure "
    echo "  don't help you to get the driver package installed correctly!                    "
    echo "-----------------------------------------------------------------------------------"
fi
echo
echo "Do you want to reboot now (default is 'yes')?"
echo "Hit 'n' + <Enter> for 'no', or just <Enter> for 'yes'."
if [ "$USE_DEFAULTS" == "NO" ]; then
    read YES_NO
else
    YES_NO="n"
fi
if [ "$YES_NO" == "n" ] || [ "$YES_NO" == "N" ]; then
    echo "You need to reboot manually to complete the installation."
else
    $SUDO shutdown -r now
fi
