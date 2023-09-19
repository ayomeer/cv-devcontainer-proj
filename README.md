# cv-devcontainer-proj
This main repository constitutes a project skeleton for computer vision projects using the cv-devcontainer Docker container as a fully featured development environment for mixed language computer vision prototyping, i.e. integrating C++ modules into Python. 

The Dockerfile defining this Docker image is also included for customizing and building from scratch. 
However, building the DevContainer image takes a considerable amount of time due to OpenCV being built from source during image building so you'll also find the built Docker image on https://hub.docker.com/r/ayomeer/cv-devcontainer-image for download via Docker. The image tagged 'cvcuda' provides an extended development environment for working with CUDA, otherwise the 'latest' image should be your default.

The other branches show various example projects that were built ontop of what is made available in this main branch.