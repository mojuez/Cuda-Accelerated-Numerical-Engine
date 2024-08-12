#include<iostream>
#include<cstdlib>
#include<math.h>
#include<string>
#include<iostream>
#include<fstream>
#include<chrono>
#include<random>


#include <cstddef>
#include <cstdio>
#include <vector>
#include <cmath>
#include <complex>

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#include<thrust/reduce.h>
#include<thrust/device_ptr.h>
#include<thrust/execution_policy.h>

#include <curand.h>
#include <curand_kernel.h>

// check function
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

// specify parameters
int Nx, Ny, Nz; // system dimension
int Norient; // number of orientations
// coefficient
float h_alpha, h_beta, h_gamma; // latent heat
float tempcoeff; // coefficient for the global energy associated to temperature
float L0; // L0 in the TDGL equation
float L; //add number related to L0

float Ttrans; // transition temperature
float Tstart, Tend; // start temperature; end temperature; step temperature
float kxx, kyy, kzz; //gradient coefficient
// simulation settings
float dx, dt;
int Nstep, Noutput;
float NcoolingRate;
float kb = 1.3806452e-23;

//Nucleation parameters
float h_gamma_star1, h_gamma_star2; //surface energy
float h_N; //coefficient for nucleation rate

//Growth
float h_Gm1, h_Gm2; //coefficient for the L = L0 * exp(-d_Gm / kb * T)

//NEW MODULE PARAMTERS for sphere
int h_thickness;
int h_radius;


// specify device parameters
__constant__ int d_Nx, d_Ny, d_Nz;
__constant__ int d_Norient;
//__constant__ float d_alpha, d_beta, d_gamma;
//__constant__ float d_L0;
__device__ float d_alpha, d_beta, d_gamma;

__constant__ float d_L0;
//__constant__ float d_L;
__constant__ float d_Gm1, d_Gm2; //coefficient for the L = L0 * exp(-d_Gm / kb * T)

__constant__ float d_Ttrans;
__constant__ float d_kxx, d_kyy, d_kzz;
//__constant__ float d_dx, d_dt;
__constant__ float d_dx;
__device__ float d_dt;
__constant__ float d_kb;

//__constant__ float d_gamma_star;
//__constant__ float d_N;


// CPU functions
// read parameters
void readparam() {
    std::ifstream inputfile;
    inputfile.open("param.in");
    // error message if the file is not opened
    if (inputfile.is_open() == false) {
        std::cout << "param.in cannot be opened" << std::endl;
        exit(1);
    }
    // read data line by line; ignore to skip comments
    inputfile >> Nx >> Ny >> Nz; inputfile.ignore(1000, '\n');
    inputfile >> Norient; inputfile.ignore(1000, '\n');
    inputfile >> h_thickness >> h_radius; inputfile.ignore(1000, '\n');
    inputfile >> h_alpha >> h_beta >> h_gamma; inputfile.ignore(1000, '\n');
    inputfile >> tempcoeff; inputfile.ignore(1000, '\n');
    inputfile >> L0; inputfile.ignore(1000, '\n');
    //inputfile >> L; inputfile.ignore(1000, '\n');
    inputfile >> h_Gm1 >> h_Gm2; inputfile.ignore(1000, '\n');
    inputfile >> Ttrans; inputfile.ignore(1000, '\n');
    inputfile >> Tstart >> Tend; inputfile.ignore(1000, '\n');
    inputfile >> kxx >> kyy >> kzz; inputfile.ignore(1000, '\n');
    inputfile >> dx; inputfile.ignore(1000, '\n');
    inputfile >> dt; inputfile.ignore(1000, '\n');
    inputfile >> Nstep >> Noutput >> NcoolingRate; inputfile.ignore(1000, '\n');
    //nucelation parameters
    inputfile >> h_gamma_star1 >> h_gamma_star2; inputfile.ignore(1000, '\n');
    inputfile >> h_N; inputfile.ignore(1000, '\n');
    inputfile.close();
}

//4D index to 1D index
__host__ int hconvert4Dindex(int i, int j, int k, int n) {
    int index1D;
    index1D = i * Ny * Nz * Norient + j * Nz * Norient + k * Norient + n;
    return index1D;
}
__device__ int dconvert4Dindex(int i, int j, int k, int n) {
    int index1D;
    index1D = i * d_Ny * d_Nz * d_Norient + j * d_Nz * d_Norient + k * d_Norient + n;
    return index1D;
}
//3D index to 1D index
__host__ int convert3Dindex(int i, int j, int k) {
    int index1D;
    index1D = i * Ny * Nz + j * Nz + k;
    return index1D;
}

__device__ int dconvert3Dindex(int i, int j, int k) {
    int index1D;
    index1D = i * d_Ny * d_Nz + j * d_Nz + k;
    return index1D;
}

// initialization of eta
__host__ void initialize(float* eta) {
    srand(30);
    for (int i = 0; i < Nx * Ny * Nz * Norient; i++) {
        eta[i] = ((float)rand() / RAND_MAX) * 0.002 - 0.001;
    }
}

// get poshi (indicator of grain or grain boundary) based on eqn. (6)
__host__ float outputgrainvol(float* eta, float* grainvol, int step) {
    int indexeta, indexgrainvol;
    float avegrainvol = 0;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                // get corre
                indexgrainvol = convert3Dindex(i, j, k);
                grainvol[indexgrainvol] = 0;
                for (int n = 0; n < Norient; n++) {
                    indexeta = hconvert4Dindex(i, j, k, n);
                    grainvol[indexgrainvol] += pow(eta[indexeta], 2);
                }
                // calcualate the average grainvol
                avegrainvol += grainvol[indexgrainvol];
            }
        }
    }
    avegrainvol = avegrainvol / (Nx * Ny * Nz);
    // output the grainvol
    std::string filename = "grainvol" + std::to_string(step) + ".txt";
    std::ofstream outputfile;
    outputfile.open(filename);
    outputfile << "x" << " " << "y" << " " << "z" << " " << "grainvol" << std::endl;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                indexgrainvol = convert3Dindex(i, j, k);
                outputfile << i << " " << j << " " << k << " " << grainvol[indexgrainvol] << std::endl;
            }
        }
    }
    outputfile.close();

    return avegrainvol;

}

__host__ void output_eta_index(float* eta, int step) {
    int index_eta;

    std::string filename = "eta_index" + std::to_string(step) + ".txt";
    std::ofstream outputfile;
    outputfile.open(filename);
    //outputfile << "x" << " " << "y" << " " << "z" << " " << "eta" << std::endl;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                outputfile << i << " " << j << " " << k << " ";
                float sum_eta_pow2 = 0;
                int max_index = 0;
                float max_eta = 0;
                for (int n = 0; n < Norient; n++) {
                    index_eta = hconvert4Dindex(i, j, k, n);
                    sum_eta_pow2 += pow(eta[index_eta], 2);
                    if (abs(eta[index_eta]) > max_eta) {
                        max_eta = abs(eta[index_eta]);
                        max_index = n;
                    }
                }
                if (sum_eta_pow2 < 0.8) {
                    outputfile << 0 << std::endl;
                }
                else {
                    outputfile << max_index + 1 << std::endl;
                }
            }
        }
    }
}


//read the eta file as input
__host__ void read_eta_file(float* eta, int step) {
    //int index_eta;
    std::ifstream inputfile;
    inputfile.open("eta" + std::to_string(step) + ".txt");
    // error message if the file is not opened
    if (inputfile.is_open() == false) {
        std::cout << "eta.txt cannot be opened" << std::endl;
        exit(1);
    }
    else if (inputfile.is_open() == true) {
        std::string line;
        int m = 0;
        while (std::getline(inputfile, line)) {
            std::stringstream linestream(line);
            std::string data;
            int n = 0;
            while (std::getline(linestream, data, ' ')) {
                if (n >= 3 && n < Norient + 3) {
                    //std::cout << data << std::endl;
                    eta[m] = std::stof(data);
                    //std::cout << eta[m] << std::endl;
                    m++;
                }
                n++;
            }

        }
    }
    inputfile.close();

}


__host__ void output_eta(float* eta, int step) {
    int index_eta;
    std::string filename = "eta" + std::to_string(step) + ".txt";
    std::ofstream outputfile;
    outputfile.open(filename);
    //outputfile << "x" << " " << "y" << " " << "z" << " " << "eta" << std::endl;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                outputfile << i << " " << j << " " << k << " ";
                for (int n = 0; n < Norient; n++) {
                    index_eta = hconvert4Dindex(i, j, k, n);
                    outputfile << eta[index_eta] << " ";
                }
                outputfile << std::endl;
            }
        }
    }
}



__global__ void calgridenergy(float* eta, float* energy) {
    int index;
    index = blockIdx.x * blockDim.x + threadIdx.x;
    // do calculation if the index is in the range
    if (index < d_Nx * d_Ny * d_Nz) {
        // calculate the volume energy
        energy[index] = 0;
        // calculate alpha and beta terms
        for (int n = 0; n < d_Norient; n++) {
            energy[index] += - 0.5 * d_alpha * pow(eta[index + n], 2)
                + 0.25 * d_beta * pow(eta[index + n], 4);
        }
        // calculate the gamma term
        for (int n = 0; n < d_Norient; n++) {
            for (int m = n + 1; m < d_Norient; m++) {
                energy[index] += d_gamma * pow(eta[index + n], 2) * pow(eta[index + m], 2);
            }
        }
    }
}


__host__ void addnuclei(float* eta, float temp, float energy) {
    // write nucleation cod
    int index1D_4D;
    float sum_eta_pow2 = 0; // sum of eta**2
    float distance = 0;
    float pi = 3.14159265358979323846264;
    float random;
    float h_gamma_star = h_gamma_star1;

    //Trying to find the new nucelation energy barrier
    //float Ttrans = 1773.15;
    float dgv = energy * exp(1e-3 * (Ttrans - temp)) * 1e7;


    float dG = 16 * pi * pow(h_gamma_star, 3) / (3 * pow(dgv, 2));
    float nucleation_rate = h_N * exp(-dG / (kb * temp));
    float nucleation_probability = 1.0 - exp(-nucleation_rate * dt);
    float radius_critic = -2.0 * h_gamma_star / dgv * 1e10;

    //float nucleation_probability = 0.4;
    //float radius_critic = 2;

    printf("dgv: %e\n", dgv);
    printf("Gibbs* dG: %e\n", dG);
    printf("nucleation rate: %e\n", nucleation_rate);
    printf("nucleation probability: %e\n", nucleation_probability);
    printf("radius critic not round: %f\n", radius_critic);

    // Rounding radius_critic
    radius_critic = round(radius_critic);

    printf("radius critic int: %f\n", radius_critic);

    printf("########################################\n");


    //int tId = threadIdx.x + blockIdx.x * blockDim.x;
    //curandState state;
    //curand_init((unsigned long long)clock() + tId, 0, 0, &state);

    for (int i = 0; i < Nx * Ny * Nz * Norient; i++) {
        sum_eta_pow2 += pow(eta[i], 2);
    }

    srand(50);

    if (radius_critic < 1) {
        printf("radius critic: %f is less than 1. Exiting addnuclei function.\n", radius_critic);
        return;  // Terminate the function here
    }
    else {
		printf("radius critic: %f is greater than 1. Continue addnuclei function.\n", radius_critic);
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    random = ((float)rand() / RAND_MAX); //range 0~1
                    //random = 0.5;
                    //random = curand_uniform_double(&state);
                    //random = curand_uniform(&state);
                    //printf("random number: %f\n", random);
                    sum_eta_pow2 = 0;
                    for (int n = 0; n < Norient; n++) {
                        index1D_4D = hconvert4Dindex(i, j, k, n);
                        sum_eta_pow2 += pow(eta[index1D_4D], 2);
                    }

                    //change to sum eta**2 as judgement
                    if (random < nucleation_probability && sum_eta_pow2 < 0.9) {
                        if ((i - radius_critic - distance) >= 1 && (i + radius_critic + distance) <= Nx &&
                            (j - radius_critic - distance) >= 1 && (j + radius_critic + distance) <= Ny &&
                            (k - radius_critic - distance) >= 1 && (k + radius_critic + distance) <= Nz) {
                            int overlap = 0;

                            for (int ii = i - radius_critic - distance; ii < i + radius_critic + distance; ii++) {
                                for (int jj = j - radius_critic - distance; jj < j + radius_critic + distance; jj++) {
                                    for (int kk = k - radius_critic - distance; kk < k + radius_critic + distance; kk++) {
                                        if (pow((ii - i), 2) + pow((jj - j), 2) + pow((kk - k), 2)
                                            <= pow(radius_critic + distance, 2)) {

                                            sum_eta_pow2 = 0;
                                            //add sum eta**2 > 0.9 for loop grain overlap
                                            for (int nn = 0; nn < Norient; nn++) {
                                                index1D_4D = hconvert4Dindex(ii, jj, kk, nn);
                                                sum_eta_pow2 += pow(eta[index1D_4D], 2);
                                            }
                                            if (sum_eta_pow2 > 0.9) {
                                                overlap = overlap + 1;
                                            }
                                        }
                                    }
                                }
                            }

                            if (overlap == 0) {
                                //int random_2 = 35 * ((float)rand() / RAND_MAX); //range 0~35 * numer pf orientation = 36
                                //int random_2 = 35 * curand_uniform(&state);
                                int random_2 = (Norient - 1) * ((double)rand() / RAND_MAX); //range 0~35 * numer pf orientation = 36
                                for (int ii = i - radius_critic - distance; ii < i + radius_critic + distance; ii++) {
                                    for (int jj = j - radius_critic - distance; jj < j + radius_critic + distance; jj++) {
                                        for (int kk = k - radius_critic - distance; kk < k + radius_critic + distance; kk++) {
                                            if (pow((ii - i), 2) + pow((jj - j), 2) + pow((kk - k), 2)
                                                <= pow(radius_critic, 2)) {

                                                //for loop  eta[random_2] = 1, others = 0
                                                for (int nn = 0; nn < Norient; nn++) {
                                                    index1D_4D = hconvert4Dindex(ii, jj, kk, nn);
                                                    if (nn == random_2) {
                                                        eta[index1D_4D] = 1;
                                                    }
                                                    else {
                                                        eta[index1D_4D] = 0;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                std::cout << i << "" << j << "" << k << std::endl;
                            }
                        }
                    }
                }
            }
        }
	}
    
}

//periodic boundary condition
__global__
void calRHS_PeriodicBC(float* eta, float* RHS, float* temp) {
    // get current index
    int index;
    index = blockIdx.x * blockDim.x + threadIdx.x;
    // do calculation if the index is in the range
    if (index < d_Nx * d_Ny * d_Nz * d_Norient) {
        // calcualte the i, j, k, n index first
        int index_temp;
        index_temp = index;
        int i0, j0, k0, n0;
        n0 = index_temp % d_Norient;
        index_temp = index_temp / d_Norient;
        k0 = index_temp % d_Nz;
        index_temp = index_temp / d_Nz;
        j0 = index_temp % d_Ny;
        index_temp = index_temp / d_Ny;
        i0 = index_temp;
        // calculate the volume energy
        int index_othereta;
        // alpha and beta terms
        RHS[index] = d_alpha * eta[index] - d_beta * pow(eta[index], 3);
        // gamma terms
        for (int ntemp = 0; ntemp < d_Norient; ntemp++) {
            if (ntemp != n0) {
                index_othereta = dconvert4Dindex(i0, j0, k0, ntemp);
                RHS[index] = RHS[index] - 2 * d_gamma * eta[index] * pow(eta[index_othereta], 2);
            }
        }
        // calculate the gradient energy
        int index_x1, index_x2;
        int index_y1, index_y2;
        int index_z1, index_z2;
        bool x1D, y1D, z1D;
        // check the dimension
        if (d_Nx != 1) { x1D = false; }
        else { x1D = true; }
        if (d_Ny != 1) { y1D = false; }
        else { y1D = true; }
        if (d_Nz != 1) { z1D = false; }
        else { z1D = true; }
        // get index of neighbors first
        // Implemenet periodic boundary condition
        if (x1D == false) {
            index_x1 = i0 - 1;
            if (index_x1 < 0) { index_x1 = index_x1 + d_Nx; }
            index_x2 = i0 + 1;
            if (index_x2 >= d_Nx) { index_x2 = index_x2 - d_Nx; }
        }
        if (y1D == false) {
            index_y1 = j0 - 1;
            if (index_y1 < 0) { index_y1 = index_y1 + d_Ny; }
            index_y2 = j0 + 1;
            if (index_y2 >= d_Ny) { index_y2 = index_y2 - d_Ny; }
        }
        if (z1D == false) {
            index_z1 = k0 - 1;
            if (index_z1 < 0) { index_z1 = index_z1 + d_Nz; }
            index_z2 = k0 + 1;
            if (index_z2 >= d_Nz) { index_z2 = index_z2 - d_Nz; }
        }
        // calculate laplace of eta
        int indextemp1, indextemp2;
        if (x1D == false) {
            indextemp1 = dconvert4Dindex(index_x1, j0, k0, n0);
            indextemp2 = dconvert4Dindex(index_x2, j0, k0, n0);
            //RHS[index] = RHS[index] + d_kxx * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
            RHS[index] = RHS[index] + 1 * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
        }
        if (y1D == false) {
            indextemp1 = dconvert4Dindex(i0, index_y1, k0, n0);
            indextemp2 = dconvert4Dindex(i0, index_y2, k0, n0);
            //RHS[index] = RHS[index] + d_kyy * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
            RHS[index] = RHS[index] + 1 * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
        }
        if (z1D == false) {
            indextemp1 = dconvert4Dindex(i0, j0, index_z1, n0);
            indextemp2 = dconvert4Dindex(i0, j0, index_z2, n0);
            //RHS[index] = RHS[index] + d_kzz * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
            RHS[index] = RHS[index] + 1 * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
        }
        //RHS[index] = - A * eta[index] + B * pow(eta[index], 2) - C * eta[index] * sumeta2;
            //if (index == 0){
            //        printf("%f %f\n", eta[index], RHS[index]);
            //}

    }
}

// __global__
// void updateeta(float* eta, float* RHS, float* energy, float* temp) {
//     // get current index
//     // calculate L (open for test later)
//     //float L = 1.0;
//     //float migration_energy = 4.06045870414309e-19;
//     //float L = d_L0 * expf(- d_Gm / (d_kb * temp[0])); //d_Gm migration barrier
//     float L =  expf(- d_Gm / (d_kb * temp[0])); //d_Gm migration barrier
//     //printf("L value:%e, %e\n", d_Gm, L);
//     //printf("L value:%e,%e,%e\n", migration_energy, L, temp[0]);
//     //printf("L value:%f,%e\n", L, temp[0]);
//     int index;
//     index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < d_Nx * d_Ny * d_Nz * d_Norient) {
//         //eta[index] = eta[index] + L * d_dt * RHS[index];

//         //after nondimensionlization
//         eta[index] = eta[index] + d_dt * RHS[index] * L; //here should multiply expf(-migration_energy / (d_kb * temp[0]));
//                                                          //but we here just use L to test the code, we didn't adjust migration_energy
//     }
//     //if (index == 0){
//     //            printf("%f %f\n", eta[index], RHS[index]);
//     //    }
// }


__global__ void updateeta_withBC(float* eta, float* RHS, float* energy, float* temp, int* materialsID) {
    // get current index
    // calculate L (open for test later)
    //float L = 1.0;
    //float migration_energy = 4.06045870414309e-19;
    //float L = d_L0 * expf(- d_Gm / (d_kb * temp[0])); //d_Gm migration barrier
    //float L = expf(- d_Gm / (d_kb * temp[0])); //d_Gm migration barrier
    //printf("L value:%e, %e\n", d_Gm, L);
    //printf("L value:%e,%e,%e\n", migration_energy, L, temp[0]);
    //printf("L value:%f,%e\n", L, temp[0]);
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < d_Nx * d_Ny * d_Nz * d_Norient) {
        int index_temp = index;
        int i0, j0, k0, n0;
        n0 = index_temp % d_Norient;
        index_temp = index_temp / d_Norient;
        k0 = index_temp % d_Nz;
        index_temp = index_temp / d_Nz;
        j0 = index_temp % d_Ny;
        index_temp = index_temp / d_Ny;
        i0 = index_temp;

        float d_Gm_array[] = {0, d_Gm1, d_Gm2};  //related to materials ID  
        int index1D_center = dconvert3Dindex(i0, j0, k0);
        int materialID_center = materialsID[index1D_center];

        //int materialIndex = index % (d_Nx * d_Ny * d_Nz);

        //if (materialsID[materialIndex]!= 0) {
        if(materialID_center!=0){
            float L = expf(- d_Gm_array[materialID_center] / (d_kb * temp[0])); //d_Gm migration barrier
            eta[index] = eta[index] + d_dt * RHS[index] * L; //here should multiply expf(-migration_energy / (d_kb * temp[0]));
                                                            //but we here just use L to test the code, we didn't adjust migration_energe
        }else{
            eta[index] = 0;
        }
    }
}

// __global__ void setParametersGlobal(float* L, int* materialsID, float* eta){
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < d_Nx * d_Ny * d_Nz * d_Norient) {
//         int index_temp = index;
//         int i0, j0, k0, n0;
//         n0 = index_temp % d_Norient;
//         index_temp = index_temp / d_Norient;
//         k0 = index_temp % d_Nz;
//         index_temp = index_temp / d_Nz;
//         j0 = index_temp % d_Ny;
//         index_temp = index_temp / d_Ny;
//         i0 = index_temp;

//         //float d_Gm_array[] = {0, d_Gm1, d_Gm2};  //related to materials ID  
//         float Gm;           

//         int index1D_center = dconvert3Dindex(i0, j0, k0);
//         int materialID_center = materialsID[index1D_center];
//         if(materialID_center!=0){
//             //float L[index1D_center] = expf(- d_Gm_array[materialID_center] / (d_kb * temp[0])); //d_Gm migration barrier
//             float sum_eta_pow2 = 0.0f;
//             for (int n = 0; n < d_Norient; n++) {
//                 int index1D_4D = dconvert4Dindex(i0, j0, k0, n);
//                 sum_eta_pow2 += powf(eta[index1D_4D], 2);
//             }

//             if (sum_eta_pow2 > 0.8f) {
//                 if (materialID_center == 1) {
//                     Gm = d_Gm_nuclei1;
//                     L[index1D_center] = expf(- Gm / (d_kb * temp[0]));
//                 } else if (materialID_center == 2) {
//                     Gm = d_Gm_nuclei2;
//                     L[index1D_center] = expf(- Gm / (d_kb * temp[0]));
//                 }
//             } else if (sum_eta_pow2 < 0.8f) {
//                 if (materialID_center == 1) {
//                     Gm = d_Gm_boundary1;
//                     L[index1D_center] = expf(- Gm / (d_kb * temp[0]));
//                 } else if (materialID_center == 2) {
//                     Gm = d_Gm_boundary2;
//                     L[index1D_center] = expf(- Gm / (d_kb * temp[0]));
//                 }
//             }

//         }
//     }
// }


// Kernel to perform computation transform the value to non-dimension
__global__ void updateConstants() {
    float multiplier0 = (d_kxx / pow(1e-6f,2)) * d_L0; //kxx * L/X0**2
    float multiplier1 = pow(1e-6f, 2) / d_kxx; // X0**2/kxx
    d_dt *= multiplier0;
    d_alpha *= multiplier1;
    d_beta *= multiplier1;
    d_gamma *= multiplier1;
}



//!!! NEW MODULE ADDED
//Initial structure of a sphere with thickness, positioned inside a cubic grid.
//Function to check if a point (x, y, z) is inside the sphere, on the boundary, or in the air
__host__ int getMaterialID(int x, int y, int z){
    int centerX = Nx / 2;
    int centerY = Ny / 2;
    int centerZ = Nz / 2;   
    int dx = x - centerX;
    int dy = y - centerY;
    int dz = z - centerZ;
    int distanceSquared = dx * dx + dy * dy + dz * dz;
    int radiusSquared = h_radius * h_radius;
    int outerRadiusSquared = (h_radius + h_thickness) * (h_radius + h_thickness);

    if (distanceSquared < radiusSquared) {
        return 2; // Inside the sphere
    } else if (distanceSquared <= outerRadiusSquared) {
        return 1; // Boundary
    } else {
        return 0; // Air
    }

}


__host__ void fillMaterialIDArray(int* materialIDs) {
    int centerX = Nx / 2;
    int centerY = Ny / 2;
    int centerZ = Nz / 2;
    int index1D, distanceSquared, radiusSquared = h_radius * h_radius, outerRadiusSquared = (h_radius + h_thickness) * (h_radius + h_thickness);

    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            for (int z = 0; z < Nz; z++) {
                int dx = x - centerX;
                int dy = y - centerY;
                int dz = z - centerZ;
                distanceSquared = dx * dx + dy * dy + dz * dz;

                index1D = convert3Dindex(x, y, z);
                if (distanceSquared < radiusSquared) {
                    materialIDs[index1D] = 2; // Inside the sphere
                } else if (distanceSquared <= outerRadiusSquared) {
                    materialIDs[index1D] = 1; // Boundary
                } else {
                    materialIDs[index1D] = 0; // Air
                }
            }
        }
    }
}


//Save the Initial structure
__host__ void saveToFile(const std::string& filename, int* grid) {
    std::ofstream file(filename);
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                int index1D = convert3Dindex(i, j, k);
                file << i << " " << j << " " << k << " " << grid[index1D] << std::endl;
            }
        }
    }
    file.close();
}

//Initialization For Sphere eta
__host__ void initialize_Sphere_eta(float* eta, int* materialIDs) {
    srand(30);  // Seed for reproducibility
    int index = 0;
    int index1D;

    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            for (int z = 0; z < Nz; z++) {
                index1D = convert3Dindex(x, y, z);
                if (materialIDs[index1D] == 1 || materialIDs[index1D] == 2) {
                    for (int ori = 0; ori < Norient; ori++, index++) {
                        eta[index] = ((float)rand() / RAND_MAX) * 0.002 - 0.001;
                    }
                } else {
                    for (int ori = 0; ori < Norient; ori++, index++) {
                        eta[index] = 0;
                    }
                }
            }
        }
    }
}

//Adjustable parameters for the nucleation (surface&non-surface)
__host__ void addnuclei_sphere(float* eta, int* materialIDs, float temp, float energy) {
    // write nucleation cod
    int index1D_4D;
    float sum_eta_pow2 = 0; // sum of eta**2
    float distance = 0;
    float pi = 3.14159265358979323846264;
    float random;

    float h_gamma_star_values[] = {0, h_gamma_star1, h_gamma_star2};

    for (int i = 0; i < Nx * Ny * Nz * Norient; i++) {
        sum_eta_pow2 += pow(eta[i], 2);
    }

    //srand(50);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);


    //value check
    float* radius_critic_array = new float[Nx * Ny * Nz]; 
    float* dgv_array = new float[Nx * Ny * Nz]; 
    float* dG_array = new float[Nx * Ny * Nz]; 
    float* nucleation_rate_array = new float[Nx * Ny * Nz]; 
    float* nucleation_probability_array = new float[Nx * Ny * Nz];
    float* h_gamma_star_array = new float[Nx * Ny * Nz];
    float* energy_array = new float[Nx * Ny * Nz];
    float* random_array = new float[Nx * Ny * Nz];

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                int index1D = convert3Dindex(i,j,k);
                int materialID = materialIDs[index1D];
              
                float h_gamma_star = h_gamma_star_values[materialID];
                //Trying to find the new nucelation energy barrier
                float dgv = energy * exp(1e-3 * (Ttrans - temp)) * 3e8;
                float dG = 16 * pi * pow(h_gamma_star, 3) / (3 * pow(dgv, 2));
                float nucleation_rate = h_N * exp(-dG / (kb * temp));
                float nucleation_probability = 1.0 - exp(-nucleation_rate * dt);
                float radius_critic = -2.0 * h_gamma_star / dgv * 1e10;
                
                //printf("h_gamma_star: %f\n", h_gamma_star);
                //printf("dgv: %e\n", dgv);
                //printf("Gibbs* dG: %e\n", dG);
                //printf("nucleation rate: %e\n", nucleation_rate);
                //printf("nucleation probability: %e\n", nucleation_probability);
                //printf("radius critic not round: %f\n", radius_critic);

                // Rounding radius_critic
                radius_critic = round(radius_critic);

                //printf("radius critic int: %f\n", radius_critic);

                //value check
                h_gamma_star_array[index1D] = h_gamma_star;
                energy_array[index1D] = energy;
                radius_critic_array[index1D] = radius_critic;
                dgv_array[index1D] = dgv; 
                dG_array[index1D] = dG; 
                nucleation_rate_array[index1D] = nucleation_rate; 
                nucleation_probability_array[index1D] = nucleation_probability;

                //printf("########################################\n");

                if (radius_critic < 1) {
                    //printf("radius critic: %f is less than 1. Exiting addnuclei function.\n", radius_critic);
                    continue;  // Terminate the function here
                    }
                    else {
                        //printf("radius critic: %f is greater than 1. Continue addnuclei function.\n", radius_critic);
                    }

                //random = ((float)rand() / RAND_MAX); //range 0~1
                random = dist(rng);

                random_array[index1D] = random;

                sum_eta_pow2 = 0;
                for (int n = 0; n < Norient; n++) {
                    index1D_4D = hconvert4Dindex(i, j, k, n);
                    sum_eta_pow2 += pow(eta[index1D_4D], 2);
                }

                //change to sum eta**2 as judgement
                if (random < nucleation_probability && sum_eta_pow2 < 0.9) {
                    if ((i - radius_critic - distance) >= 1 && (i + radius_critic + distance) <= Nx &&
                        (j - radius_critic - distance) >= 1 && (j + radius_critic + distance) <= Ny &&
                        (k - radius_critic - distance) >= 1 && (k + radius_critic + distance) <= Nz) {
                        int overlap = 0;

                        for (int ii = i - radius_critic - distance; ii < i + radius_critic + distance; ii++) {
                            for (int jj = j - radius_critic - distance; jj < j + radius_critic + distance; jj++) {
                                for (int kk = k - radius_critic - distance; kk < k + radius_critic + distance; kk++) {
                                    if (pow((ii - i), 2) + pow((jj - j), 2) + pow((kk - k), 2)
                                        <= pow(radius_critic + distance, 2)) {

                                        sum_eta_pow2 = 0;
                                        //add sum eta**2 > 0.9 for loop grain overlap
                                        for (int nn = 0; nn < Norient; nn++) {
                                            index1D_4D = hconvert4Dindex(ii, jj, kk, nn);
                                            sum_eta_pow2 += pow(eta[index1D_4D], 2);
                                        }
                                        if (sum_eta_pow2 > 0.9) {
                                            overlap = overlap + 1;
                                        }
                                    }
                                }
                            }
                        }

                        if (overlap == 0) {
                            //int random_2 = 35 * ((float)rand() / RAND_MAX); //range 0~35 * numer pf orientation = 36
                            //int random_2 = 35 * curand_uniform(&state);
                            int random_2 = (Norient - 1) * ((double)rand() / RAND_MAX); //range 0~35 * numer pf orientation = 36
                            for (int ii = i - radius_critic - distance; ii < i + radius_critic + distance; ii++) {
                                for (int jj = j - radius_critic - distance; jj < j + radius_critic + distance; jj++) {
                                    for (int kk = k - radius_critic - distance; kk < k + radius_critic + distance; kk++) {
                                        if (pow((ii - i), 2) + pow((jj - j), 2) + pow((kk - k), 2)
                                            <= pow(radius_critic, 2)) {

                                            //for loop  eta[random_2] = 1, others = 0
                                            for (int nn = 0; nn < Norient; nn++) {
                                                index1D_4D = hconvert4Dindex(ii, jj, kk, nn);
                                                if (nn == random_2) {
                                                    eta[index1D_4D] = 1;
                                                }
                                                else {
                                                    eta[index1D_4D] = 0;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            std::cout << i << "" << j << "" << k << std::endl;
                        }
                    }
                }
            }
        }
    }
    // Generate filename based on the current step
    //std::string filename = "radius_critic_output_step" + std::to_string(step) + ".txt";
    std::ofstream output_file("radius_critic_output.txt");
    output_file << "x   y   z  energy h_gamma_star  radius   dgv     dG     nucleation_rate   nucleation_probability    random_number\n";
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                int index1D = convert3Dindex(i, j, k);
                output_file << i << "   " << j << "   " << k << "   " 
                << energy_array[index1D] << "      "
                << h_gamma_star_array[index1D] << "      "
                << radius_critic_array[index1D] << "      "
                << dgv_array[index1D] << "      "
                << dG_array[index1D] << "      "
                << nucleation_rate_array[index1D] << "      "
                << nucleation_probability_array[index1D] << "      "
                << random_array[index1D]
                << "\n";
            }
        }
    }
    output_file.close();
   
}


//New Nucleation Growth With Boundary No Leak on Air(materialsID=0)
__host__ void addnuclei_sphere_withBoundary(float* eta, int* materialIDs, float temp, float energy) {
    int index1D_4D;
    float sum_eta_pow2 = 0; 
    float distance = 0;
    float pi = 3.14159265358979323846264;
    float random;
    float h_gamma_star_values[] = {0, h_gamma_star1, h_gamma_star2};

    for (int i = 0; i < Nx * Ny * Nz * Norient; i++) {
        sum_eta_pow2 += pow(eta[i], 2);
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float* radius_critic_array = new float[Nx * Ny * Nz]; 
    float* dgv_array = new float[Nx * Ny * Nz]; 
    float* dG_array = new float[Nx * Ny * Nz]; 
    float* nucleation_rate_array = new float[Nx * Ny * Nz]; 
    float* nucleation_probability_array = new float[Nx * Ny * Nz];
    float* h_gamma_star_array = new float[Nx * Ny * Nz];
    float* energy_array = new float[Nx * Ny * Nz];
    float* random_array = new float[Nx * Ny * Nz];

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                int index1D = convert3Dindex(i, j, k);
                int materialID = materialIDs[index1D];

                if (materialID == 0) {
                   continue;
                }

                float h_gamma_star = h_gamma_star_values[materialID];
                float dgv = energy * exp(1e-3 * (Ttrans - temp)) * 3e8;
                float dG = 16 * pi * pow(h_gamma_star, 3) / (3 * pow(dgv, 2));
                float nucleation_rate = h_N * exp(-dG / (kb * temp));
                float nucleation_probability = 1.0 - exp(-nucleation_rate * dt);
                float radius_critic = -2.0 * h_gamma_star / dgv * 1e10;

                radius_critic = round(radius_critic);

                h_gamma_star_array[index1D] = h_gamma_star;
                energy_array[index1D] = energy;
                radius_critic_array[index1D] = radius_critic;
                dgv_array[index1D] = dgv; 
                dG_array[index1D] = dG; 
                nucleation_rate_array[index1D] = nucleation_rate; 
                nucleation_probability_array[index1D] = nucleation_probability;

                if (radius_critic < 1) {
                    continue;  
                }

                random = dist(rng);
                random_array[index1D] = random;

                sum_eta_pow2 = 0;
                for (int n = 0; n < Norient; n++) {
                    index1D_4D = hconvert4Dindex(i, j, k, n);
                    sum_eta_pow2 += pow(eta[index1D_4D], 2);
                }

                if (random < nucleation_probability && sum_eta_pow2 < 0.9) {
                    if ((i - radius_critic - distance) >= 1 && (i + radius_critic + distance) <= Nx &&
                        (j - radius_critic - distance) >= 1 && (j + radius_critic + distance) <= Ny &&
                        (k - radius_critic - distance) >= 1 && (k + radius_critic + distance) <= Nz) {

                        // Check if the nuclei will extend to Air
                        bool valid_nucleation = true;
                        for (int ii = i - radius_critic - distance; ii <= i + radius_critic + distance && valid_nucleation; ii++) {
                            for (int jj = j - radius_critic - distance; jj <= j + radius_critic + distance && valid_nucleation; jj++) {
                                for (int kk = k - radius_critic - distance; kk <= k + radius_critic + distance && valid_nucleation; kk++) {
                                    int idx = convert3Dindex(ii, jj, kk);
                                    if (materialIDs[idx] == 0) {
                                        valid_nucleation = false;
                                    }
                                }
                            }
                        }

                        if (!valid_nucleation) {
                            continue; 
                        }

                        int overlap = 0;

                        // Check nuclei overlap
                        for (int ii = i - radius_critic - distance; ii < i + radius_critic + distance; ii++) {
                            for (int jj = j - radius_critic - distance; jj < j + radius_critic + distance; jj++) {
                                for (int kk = k - radius_critic - distance; kk < k + radius_critic + distance; kk++) {
                                    if (pow((ii - i), 2) + pow((jj - j), 2) + pow((kk - k), 2)
                                        <= pow(radius_critic + distance, 2)) {

                                        sum_eta_pow2 = 0;
                                        for (int nn = 0; nn < Norient; nn++) {
                                            index1D_4D = hconvert4Dindex(ii, jj, kk, nn);
                                            sum_eta_pow2 += pow(eta[index1D_4D], 2);
                                        }
                                        if (sum_eta_pow2 > 0.9) {
                                            overlap = overlap + 1;
                                        }
                                    }
                                }
                            }
                        }

                        if (overlap == 0) {
                            int random_2 = (Norient - 1) * dist(rng); 
                            for (int ii = i - radius_critic - distance; ii < i + radius_critic + distance; ii++) {
                                for (int jj = j - radius_critic - distance; jj < j + radius_critic + distance; jj++) {
                                    for (int kk = k - radius_critic - distance; kk < k + radius_critic + distance; kk++) {
                                        if (pow((ii - i), 2) + pow((jj - j), 2) + pow((kk - k), 2)
                                            <= pow(radius_critic, 2)) {
                                            for (int nn = 0; nn < Norient; nn++) {
                                                index1D_4D = hconvert4Dindex(ii, jj, kk, nn);
                                                if (nn == random_2) {
                                                    eta[index1D_4D] = 1;
                                                }
                                                else {
                                                    eta[index1D_4D] = 0;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            std::cout << i << " " << j << " " << k << std::endl;
                        }
                    }
                }
            }
        }
    }

    // Generate filename based on the current step
    //std::string filename = "radius_critic_output_step" + std::to_string(step) + ".txt";
    // std::ofstream output_file("radius_critic_output.txt");
    // output_file << "x   y   z  energy h_gamma_star  radius   dgv     dG     nucleation_rate   nucleation_probability    random_number\n";
    // for (int i = 0; i < Nx; i++) {
    //     for (int j = 0; j < Ny; j++) {
    //         for (int k = 0; k < Nz; k++) {
    //             int index1D = convert3Dindex(i, j, k);
    //             output_file << i << "   " << j << "   " << k << "   " 
    //             << energy_array[index1D] << "      "
    //             << h_gamma_star_array[index1D] << "      "
    //             << radius_critic_array[index1D] << "      "
    //             << dgv_array[index1D] << "      "
    //             << dG_array[index1D] << "      "
    //             << nucleation_rate_array[index1D] << "      "
    //             << nucleation_probability_array[index1D] << "      "
    //             << random_array[index1D]
    //             << "\n";
    //         }
    //     }
    // }
    // output_file.close();

    //delete[] radius_critic_array;
    //delete[] dgv_array;
    //delete[] dG_array;
    //delete[] nucleation_rate_array;
    //delete[] nucleation_probability_array;
    //delete[] h_gamma_star_array;
    //delete[] energy_array;
    //delete[] random_array;
}

//New Neumann Boundary Condition
__global__
void calRHS_NeumannAndPeriodicBC(float* eta, float* RHS, float* temp, int* materialIDs) {
    
    // get current index
    int index;
    index = blockIdx.x * blockDim.x + threadIdx.x;
    // do calculation if the index is in the range
    if (index < d_Nx * d_Ny * d_Nz * d_Norient) {
        // calcualte the i, j, k, n index first
        int index_temp;
        index_temp = index;
        int i0, j0, k0, n0;
        n0 = index_temp % d_Norient;
        index_temp = index_temp / d_Norient;
        k0 = index_temp % d_Nz;
        index_temp = index_temp / d_Nz;
        j0 = index_temp % d_Ny;
        index_temp = index_temp / d_Ny;
        i0 = index_temp;
        // calculate the volume energy
        int index_othereta;
        // alpha and beta terms
        RHS[index] = d_alpha * eta[index] - d_beta * pow(eta[index], 3);
        // gamma terms
        for (int ntemp = 0; ntemp < d_Norient; ntemp++) {
            if (ntemp != n0) {
                index_othereta = dconvert4Dindex(i0, j0, k0, ntemp);
                RHS[index] = RHS[index] - 2 * d_gamma * eta[index] * pow(eta[index_othereta], 2);
            }
        }
        // calculate the gradient energy
        int index_x1, index_x2;
        int index_y1, index_y2;
        int index_z1, index_z2;
        bool x1D, y1D, z1D;
        // check the dimension
        if (d_Nx != 1) { x1D = false; }
        else { x1D = true; }
        if (d_Ny != 1) { y1D = false; }
        else { y1D = true; }
        if (d_Nz != 1) { z1D = false; }
        else { z1D = true; }
        // get index of neighbors first
        // Implemenet periodic boundary condition
        int index1D_center = dconvert3Dindex(i0, j0, k0);
        int materialID_center = materialIDs[index1D_center];
        if (x1D == false) {
            index_x1 = i0 - 1;
            int index1D_x1 = dconvert3Dindex(index_x1, j0, k0);           
            int materialID_x1 = materialIDs[index_x1];
            if(materialID_x1 == 0 && materialID_center == 1){
                index_x1 = i0;  // Neumann BC
            }
            if (index_x1 < 0) { index_x1 = index_x1 + d_Nx; } // Periodic BC

            index_x2 = i0 + 1;
            int index1D_x2 = dconvert3Dindex(index_x2, j0, k0);
            int materialID_x2 = materialIDs[index_x2];
            if(materialID_x2 == 0 && materialID_center == 1){
                index_x2 = i0;
            }
            if (index_x2 >= d_Nx) { index_x2 = index_x2 - d_Nx; }
        }
        if (y1D == false) {
            index_y1 = j0 - 1;
            int index1D_y1 = dconvert3Dindex(i0, index_y1, k0);
            int materialID_y1 = materialIDs[index1D_y1];
            if (materialID_y1 == 0 && materialID_center == 1) {
                index_y1 = j0;
            }
            if (index_y1 < 0) { index_y1 = index_y1 + d_Ny; }

            index_y2 = j0 + 1;
            int index1D_y2 = dconvert3Dindex(i0, index_y2, k0);
            int materialID_y2 = materialIDs[index1D_y2];
            if (materialID_y2 == 0 && materialID_center == 1) {
                index_y2 = j0;
            }
            if (index_y2 >= d_Ny) { index_y2 = index_y2 - d_Ny; }
        }
        if (z1D == false) {
            index_z1 = k0 - 1;
            int index1D_z1 = dconvert3Dindex(i0, j0, index_z1);
            int materialID_z1 = materialIDs[index1D_z1];
            if (materialID_z1 == 0 && materialID_center == 1) {
                index_z1 = k0;
            }
            if (index_z1 < 0) { index_z1 = index_z1 + d_Nz; }

            index_z2 = k0 + 1;
            int index1D_z2 = dconvert3Dindex(i0, j0, index_z2);
            int materialID_z2 = materialIDs[index1D_z2];
            if (materialID_z2 == 0 && materialID_center == 1) {
                index_z2 = k0;
            }
            if (index_z2 >= d_Nz) { index_z2 = index_z2 - d_Nz;}
        }
        // calculate laplace of eta
        int indextemp1, indextemp2;
        if (x1D == false) {
            indextemp1 = dconvert4Dindex(index_x1, j0, k0, n0);
            indextemp2 = dconvert4Dindex(index_x2, j0, k0, n0);
            //RHS[index] = RHS[index] + d_kxx * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
            RHS[index] = RHS[index] + 1 * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
        }
        if (y1D == false) {
            indextemp1 = dconvert4Dindex(i0, index_y1, k0, n0);
            indextemp2 = dconvert4Dindex(i0, index_y2, k0, n0);
            //RHS[index] = RHS[index] + d_kyy * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
            RHS[index] = RHS[index] + 1 * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
        }
        if (z1D == false) {
            indextemp1 = dconvert4Dindex(i0, j0, index_z1, n0);
            indextemp2 = dconvert4Dindex(i0, j0, index_z2, n0);
            //RHS[index] = RHS[index] + d_kzz * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
            RHS[index] = RHS[index] + 1 * (eta[indextemp1] + eta[indextemp2] - 2 * eta[index]) / pow(d_dx, 2);
        }
    }
}

// check the boundary condition and set corresponding parameters


int main() {
    // read parameters
    readparam();
    // specify host arrays
    float* eta, * RHS, * grainvol;
    eta = (float*)malloc(Nx * Ny * Nz * Norient * sizeof(float));
    RHS = (float*)malloc(Nx * Ny * Nz * Norient * sizeof(float));
    grainvol = (float*)malloc(Nx * Ny * Nz * sizeof(float));
    // specify host temperature
    float* temp;
    temp = (float*)malloc(sizeof(float));
    // specify host energy
    float* energy;
    energy = (float*)malloc(sizeof(float));
    // specify device arrays
    float* d_eta, * d_RHS;
    checkCuda(cudaMalloc(&d_eta, Nx * Ny * Nz * Norient * sizeof(float)));
    checkCuda(cudaMalloc(&d_RHS, Nx * Ny * Nz * Norient * sizeof(float)));
    // specify device temperature
    float* d_temp;
    checkCuda(cudaMalloc(&d_temp, sizeof(float)));
    // specify device energy
    float* d_energy;
    checkCuda(cudaMalloc(&d_energy, sizeof(float)));
    // specify device grid energy
    float* d_gridenergy;
    checkCuda(cudaMalloc(&d_gridenergy, Nx * Ny * Nz * sizeof(float)));
    //copy parameters
    checkCuda(cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_Norient, &Norient, sizeof(int), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_kxx, &kxx, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_kyy, &kyy, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_kzz, &kzz, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_alpha, &h_alpha, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_beta, &h_beta, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_gamma, &h_gamma, sizeof(float), 0, cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpyToSymbol(d_L0, &L0, sizeof(float), 0, cudaMemcpyHostToDevice));
    //checkCuda(cudaMemcpyToSymbol(d_L, &L, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_Gm1, &h_Gm1, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_Gm2, &h_Gm2, sizeof(float), 0, cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpyToSymbol(d_Ttrans, &Ttrans, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_dx, &dx, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_dt, &dt, sizeof(float), 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(d_kb, &kb, sizeof(float), 0, cudaMemcpyHostToDevice));
    //nucleation
    //checkCuda(cudaMemcpyToSymbol(d_gamma_star, &h_gamma_star, sizeof(float), 0, cudaMemcpyHostToDevice));
    //checkCuda(cudaMemcpyToSymbol(d_N, &h_N, sizeof(float), 0, cudaMemcpyHostToDevice));
    

    //transform to non-dimension
    updateConstants << <1, 1 >> > ();
    // Variables to hold the copied values
    float alpha_check, beta_check, gamma_check, time_check;

    // Copy values from device constants to host variables
    cudaMemcpyFromSymbol(&alpha_check, d_alpha, sizeof(float));
    cudaMemcpyFromSymbol(&beta_check, d_beta, sizeof(float));
    cudaMemcpyFromSymbol(&gamma_check, d_gamma, sizeof(float));
    cudaMemcpyFromSymbol(&time_check, d_dt, sizeof(float));
    cudaMemcpyFromSymbol(&dt, d_dt, sizeof(float));


    // Print the values
    std::cout << "Alpha: " << alpha_check << std::endl;
    std::cout << "Beta: " << beta_check << std::endl;
    std::cout << "Gamma: " << gamma_check << std::endl;
    std::cout << "Time: " << time_check << std::endl;
    std::cout << "Time: " << dt << std::endl;

    //Check the initial structure
    //int* materialIDs = new int[Nx * Ny * Nz];
    int* materialIDs;
    materialIDs = (int*)malloc(Nx * Ny * Nz * sizeof(int));
    int* d_materialIDs;
    checkCuda(cudaMalloc(&d_materialIDs, Nx * Ny * Nz * sizeof(int)));
    // Fill the materialID array
    fillMaterialIDArray(materialIDs);
    // Save materialIDs to a file
    saveToFile("materialIDs.txt", materialIDs);
    //std::cout << "Material IDs have been written to materialIDs.txt." << std::endl;
    checkCuda(cudaMemcpy(d_materialIDs, materialIDs, Nx * Ny * Nz * sizeof(int), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    // Don't forget to free the allocated memory
    // delete[] materialIDs;

    // average grainvol
    float avegrainvol;
    // initialization
    //initialize(eta);
    initialize_Sphere_eta(eta, materialIDs);
    // initialize cuda time
    float milliseconds;
    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));
    // loop
    checkCuda(cudaEventRecord(startEvent, 0));
    // copy variables
    checkCuda(cudaMemcpy(d_eta, eta, Nx * Ny * Nz * Norient * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_RHS, RHS, Nx * Ny * Nz * Norient * sizeof(float), cudaMemcpyHostToDevice));
    // output
    outputgrainvol(eta, grainvol, 0);
    std::ofstream grainfile;
    grainfile.open("average.txt");
    grainfile << "#step" << " " << "temperature" << " " << "energy" << " " << "average grain vol" << std::endl;
    // specify dimension
    int blocksize = 256;
    int numblocks = (Nx * Ny * Nz * Norient + blocksize - 1) / blocksize;
    int numgridblocks = (Nx * Ny * Nz + blocksize - 1) / blocksize;

    // Loop Loop Loop ##########################################################################################
    checkCuda(cudaEventRecord(startEvent, 0));

    int roomTemperature = 300;
    for (int s = 1; s <= Nstep; s++) {
        if (s == 1) {
            // Keep the initial temperature for the first step
            temp[0] = Tstart;
        }
        else if (temp[0] > roomTemperature) {
            //temp[0] -= NcoolingRate * dt; // Decrease the temperature by the cooling rate
            temp[0] -= NcoolingRate; // Decrease the temperature by the cooling rate
            if (temp[0] < roomTemperature) {
                temp[0] = roomTemperature; // Set the temperature to the room temperature if it becomes lower
            }
        }
        //temp[0] = Tstart + (Tend - Tstart) / Nstep * s;
        checkCuda(cudaMemcpy(d_temp, temp, sizeof(float), cudaMemcpyHostToDevice));
        // calculate energy of each cell
        calgridenergy << <numgridblocks, blocksize >> > (d_eta, d_gridenergy);
        cudaDeviceSynchronize();
        // calculate the total energy
        thrust::device_ptr<float> tdp = thrust::device_pointer_cast(d_gridenergy);
        energy[0] = thrust::reduce(tdp, tdp + Nx * Ny * Nz);
        std::cout << "free energy f0:" << energy[0] << std::endl;
        //std::cout << energy[0] << std::endl;
        energy[0] = energy[0] * tempcoeff * (Ttrans - temp[0]);
        std::cout << "free energy:" << energy[0] << std::endl;

        std::cout << "temperature:" << temp[0] << std::endl;
        // transfer the energy value to device
        checkCuda(cudaMemcpy(d_energy, energy, sizeof(float), cudaMemcpyHostToDevice));

        //frist need to transfor all the value on host         
        checkCuda(cudaMemcpy(eta, d_eta, Nx * Ny * Nz * Norient * sizeof(float), cudaMemcpyDeviceToHost));

        //nucleation on host CPU
        //addnuclei(eta, temp[0], energy[0]);
        //addnuclei_sphere(eta, materialIDs, temp[0], energy[0]);
        addnuclei_sphere_withBoundary(eta, materialIDs, temp[0], energy[0]);

        //Then transfor the host value to device
        checkCuda(cudaMemcpy(d_eta, eta, Nx * Ny * Nz * Norient * sizeof(float), cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();

        //boundary condition
        //calRHS_PeriodicBC << <numblocks, blocksize >> > (d_eta, d_RHS, d_temp);
        calRHS_NeumannAndPeriodicBC << <numblocks, blocksize >> > (d_eta, d_RHS, d_temp, d_materialIDs);
        cudaDeviceSynchronize();

        //updateeta << <numblocks, blocksize >> > (d_eta, d_RHS, d_energy, d_temp);
        updateeta_withBC << <numblocks, blocksize >> > (d_eta, d_RHS, d_energy, d_temp, d_materialIDs);
        cudaDeviceSynchronize();

        if (s % Noutput == 0) {
            checkCuda(cudaMemcpy(eta, d_eta, Nx * Ny * Nz * Norient * sizeof(float), cudaMemcpyDeviceToHost));

            //output_eta(eta, s);
            //output_eta_index(eta, s);

            checkCuda(cudaMemcpy(energy, d_energy, sizeof(float), cudaMemcpyDeviceToHost));
            avegrainvol = outputgrainvol(eta, grainvol, s);
            grainfile << s << " " << temp[0] << " " << energy[0] << " " << avegrainvol << std::endl;
            std::cout << s << " " << temp[0] << " " << energy[0] << " " << avegrainvol << std::endl;
        }
    }
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

    grainfile << "Total Loop time " << milliseconds << "ms" << std::endl;
    std::cout << "Total Loop time " << milliseconds << "ms" << std::endl;

    free(eta);
    free(RHS);
    free(grainvol);
    free(materialIDs);
    cudaFree(d_eta);
    cudaFree(d_RHS);
    cudaFree(d_materialIDs);

    return 0;
}