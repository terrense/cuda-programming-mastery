#include<stdio.h>
#include<cuda.h>

__global__ void dkernel(){
    printf("Hello! CUDA time!\n");
}

int main(){
    dkernel<<<32,32>>>();
    return 0;
}
