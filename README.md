# Cpp Bindings Homography Demo
This repository contains a demo application showcasing the use of PyBind11 to incorporate a C++ module into a Python project. The application takes an input image of a chess board and applies a homography transformation to perspectively _undistort_ the chessboard, leading to an output image that shows the chess board from directly above.


The constant ```UNDISTORT_METHOD``` at the top of the ```python/homography.py``` script can be set to either ```"python"``` or ```"cpp"``` to switch between transform algorithms implement directly within the python script or in an external C++ module. Runtimes for these transforms are printed in the console for comparison between the two undistortion methods.


## Starting the Docker Container
1. Login to lab PC: ECE-Gast, Password: Labor6004
2. Open Project in VS Code by Clicking "File -> Open Folder...". Navigate to the "home" directory and select the folder "cpp-bindings". Click "Open".
3. Click the two small arrows in the bottom left corner. This will open a dropdown menu.
4. Select "**Reopen in Container**".\
VS Code will reopen the window within a Docker Container. This might take a few seconds.
5. Check if the Container is opened properly by checking if it says "**Dev Container:ipcv**" next to the two arrows in the bottom left corner.



## Setup on Personal Device

To use the container on your personal device, you'll have to go through these installation steps once. After that you should be able to start the container as described above.

### Windows
The docker container is meant to run on a Linux System. When running it on Windows you will not be able to access cameras from within the container. However the normal exercises should run without issues.

1. **Setting up VS Code:** \
    You can download VS Code from [here.](https://code.visualstudio.com/download) \
    To run the docker container the **"Dev Containers" Extension** needs to be installed.

2. **Installing Docker Desktop:** \
    Download the installer from [here](https://docs.docker.com/desktop/install/windows-install/).\
    During Installation, make sure that the Option "Use WSL 2 instead of Hyper-V" is enabled. \
    You'll have to open up Docker Desktop every time before launching the Container.

3. **Cloning the repository:** \
    If you have never used git before you can install it from [here](https://git-scm.com/downloads). Go through with the default installation settings.\
    Once installed, open an Explorer window and navigate to the folder where you want these files be saved. Right Click and click on "Open Git Bash here". Enter the following command to download the repo:
    ```
    git clone https://gitlab.ost.ch/ipcv-teaching-material/ipcv-extra-material/cpp-bindings.git
    ```
4. **Change .devcontainer.json:** \
    Open the Project in VS Code and open up the ".devcontainer.json" file. Change the line 
    ```json consolele": "docker-compose-windows.yml",
    ```
    From here on, continue with step 2 of [Starting the Container](#starting-the-docker-container)
    


### Ubuntu
On a Linux system you should be able to execute the Webcam examples if you have an external Webcam connected. We cannot guarantee support for built-in cameras. Additionally, the Exposure/Gain sliders might not work for your camera, since some cameras require specific drivers to adjust these settings.

1. **Installing Docker Engine:** \
    Install Docker Engine with this [guide](https://docs.docker.com/engine/install/ubuntu/#installation-methods). If you want to access cameras from within the container, make sure to **only install Docker Engine, not Docker Desktop entirely**.

2. **Setting up VS Code:** \
    VS Code can be installed via the Ubuntu Software App or by entering the following in a terminal:
    ```
    sudo snap install --classic code
    ```
    To run the docker container the **"Dev Containers" Extension** needs to be installed. \
    To get a better overview of your installed Docker images and running containers we recommend installing the "Docker" extension.

3. **Cloning the Repository** \
    To clone the repository, open a terminal in the folder you would like to clone to and execute:
    ```
    git clone https://gitlab.ost.ch/ipcv-teaching-material/ipcv-extra-material/cpp-bindings.git
    ```
    
## Updating the repo
Once everything is installed you can work with the Docker Container by following the steps in [Starting the Docker Container](#starting-the-docker-container).
If you want to update your local copy of this GitLab repository with the new exercises for this week, navigate to the 'ipcv-1' folder and open a terminal by right-clicking and selecting "Open Git Bash here" for Windows or "Open in Terminal" for Ubuntu. Execute:
```
git pull
```
