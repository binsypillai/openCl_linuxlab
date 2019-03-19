#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <CL/opencl.h>
//#include <OpenCL/OpenCL.h>

const char *kernelSource =                                      "\n"\
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \

"__kernel void Euerl(  __global long *a,                        \n"\
"                       __global long *b,                        \n"\
"                       __global long *c,                        \n"\
"                       const unsigned long n)                    \n"\
"{                                                               \n"\

"    long gid = get_global_id(0);        \n"\
"    long i,j,k,length,temp;             \n"\
"    length = 0;                         \n"\
"    temp = 0;                           \n"\
"    j = 0;                              \n"\
"   if(gid < n)                          \n"\
"    {                                   \n"\
"        for (i = 1; i < a[gid]; i++)    \n"\
"        {                               \n"\
"             k = a[gid];                \n"\
"             j = i;                     \n"\
"            while (j != 0 ) {           \n"\
"                temp = k % j;           \n"\
"                k = j ;                 \n"\
"                j = temp;               \n"\
"               }                        \n"\
"               if(k == 1)               \n"\
"               {                        \n"\
"                   length++;            \n"\
"               }                        \n"\
"        }                               \n"\
"      c[gid] = length ;                 \n"\
"    }                                   \n"\
"}";


int main( int argc, char* argv[] )
{
    // Length of vectors
    //unsigned int n = 100000;
    
    long n = 100000;
    
    // Host input vectors
    long *A;
    long *B;
    // Host output vector
    long *C;
    
    // Device input buffers
    cl_mem a;
    cl_mem b;
    // Device output buffer
    cl_mem c;
    
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
    
    long lowerbound , upperbound, diff, tempVar,sum;
    
    tempVar = 0;
    
    if (argc != 3) {
   	printf("not 2 arguments\n");
    	return 1;
    }
    sscanf(argv[1], "%ld", &lowerbound);
    sscanf(argv[2], "%ld", &upperbound);
    
    
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
    
    // Allocate memory for each vector on host
    A = (long*)malloc(bytes);
    B = (long*)malloc(bytes);
    C = (long*)malloc(bytes);
 
    tempVar = lowerbound;
    diff = upperbound-lowerbound+1;
    n = diff;
    // Initializing , Once user enters upper bound and lower bound . The numbers between it
    //will be passed in this *B input can be used if summation also needs to be done in multicore .
    int i;
    for( i = 0; i < diff; i++ )
    {
        A[i] = tempVar;
        tempVar++;
        B[i] = 1;
    }
    
    size_t glSize, lSize;
    cl_int err;
    
    
    lSize = 512;   // Number of work items changed from 64 to 512 to 
                   // check if there is any change in performance
    
    clock_t t0= clock();
    glSize = ceil(n/(float)lSize)*lSize;
    
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    
    // Create context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    
    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) & kernelSource, NULL, &err);
    
    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "Euerl", &err);
    
    // reserve space
    a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    
    // Write array
    err = clEnqueueWriteBuffer(queue, a, CL_TRUE, 0,
                               bytes, A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, b, CL_TRUE, 0,
                                bytes, B, 0, NULL, NULL);
    
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c);
    err |= clSetKernelArg(kernel, 3, sizeof(long), &n);
    
    // Execute the kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &glSize, &lSize,
                                 0, NULL, NULL);
    
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
    
    // Read the results from the device
    clEnqueueReadBuffer(queue, c, CL_TRUE, 0,bytes, C, 0, NULL, NULL );
    
    //Sum up vector c and print result divided by n, this should equal 1 within error
    sum = 0;
    for(i=0; i<n; i++)
        sum += C[i];
    printf("final result: %ld\n", sum);


    printf("Time take is %f seconds\n", (double)(clock()-t0)/CLOCKS_PER_SEC);    

    
    // release OpenCL resources
    clReleaseMemObject(a);
    clReleaseMemObject(b);
    clReleaseMemObject(c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    //release host memory
    free(A);
    free(B);
    free(C);
    
    return 0;
}
