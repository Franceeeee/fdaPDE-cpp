<div align="center"> <h1> fdaPDE </h1>

<h5> Physics-Informed Spatial and Functional Data Analysis </h5> </div>

![test-linux-gcc](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-cpp/test-linux-gcc.yml?branch=stable&label=test-linux-gcc)
![test-linux-clang](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-cpp/test-linux-clang.yml?branch=stable&label=test-linux-clang)
![test-macos-clang](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-cpp/test-macos-clang.yml?branch=stable&label=test-macos-clang)

fdaPDE is a C++ library for the analysis of spatial and functional data observed over complex multidimensional domains, featuring a Partial Differential Equation regularization. 

It is built on top of the [fdaPDE Core Library](https://github.com/fdaPDE/fdaPDE-core).

This project was developed by MSc Mathematical Engineering students Francesco Maria Mancinelli (10700393) and Giulia Ortolani(********) under the supervision of professor Laura M. Sangalli, professor Eleonora Arnone, doctor Aldo Clemente and doctor Alessandro Palummo.

Contribution features the whole implemenataion of the fANOVA class, as well as validation tests. The code structure of the library is presented in the image below:

![alt text](http://url/to/img.png)
oppure
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

## Accessing the Source Code

To access the source code of the implementation, you can clone the repository from the `develop` branch:

```bash
git clone -r https://github.com/Franceeeee/fdaPDE-cpp.git -b develop
cd fdaPDE-cpp/simulations
```

## Setting Up the Environment

### Using Docker

A pre-configured Docker image is provided to ensure that all dependencies are correctly set up in a clean environment.

1. **Pull the pre-built Docker image:**

   ```bash
   docker pull aldoclemente/fdapde-docker
   ```

2. **Run an interactive Docker container:**

   ```bash
   docker run -it aldoclemente/fdapde-docker /bin/bash
   ```

   This will start a terminal session inside the Docker container with the environment ready for use.

## Compiling the Simulation Executable

After cloning the repository and setting up the environment, compile the simulation executable using the following command:

```bash
g++ -o simulation_1_iter simulation_1_iter.cpp -I../ -I../fdaPDE/core/ -I/usr/include/eigen3 -O2 -std=c++20 -g -march=native -DFDAPDE_NO_DEBUG
```

This will compile the `simulation_base.cpp` file and create an executable named `simulation_base`. The command includes the necessary directories and compilation flags for optimization and debugging.

## Running the Simulation

To run the compiled simulation and generate results, execute the following command:

```bash
./simulation_base
```

By following the steps outlined, users can successfully verify the correct installation of the library.



## Documentation
Documentation can be found on our [documentation site](https://fdapde.github.io/)
