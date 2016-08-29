#include "ITMEKF_CUDA.h"
#include "ITMCUDAUtils.h"
//#include "../../DeviceAgnostic/ITMDepthTracker.h"
#include "../../../../ORUtils/CUDADefines.h"
//#include "../../DeviceAgnostic/ITMPixelUtils.h"
#include "../../DeviceAgnostic/ITMEKF.h"

using namespace ITMLib::Engine;

struct ITMEKF_CUDA::accuCell {
    int numPoints;
    float g[36];
};

ITMEKF_CUDA::ITMEKF_CUDA():ITMEKF()
{
    ITMSafeCall(cudaMallocHost((void**)&accu_host1, sizeof(accuCell)));
    ITMSafeCall(cudaMalloc((void**)&accu_device1, sizeof(accuCell)));
}

ITMEKF_CUDA::~ITMEKF_CUDA(void)
{
    ITMSafeCall(cudaFreeHost(accu_host1));
    ITMSafeCall(cudaFree(accu_device1));
}

__global__ void g_rt_device(ITMEKF_CUDA::accuCell *accu, float *depth, Matrix4f approxInvPose, Vector4f *pointsMap,
                            Vector4f *normalsMap, Vector4f sceneIntrinsics, Vector2i sceneImageSize, Matrix4f scenePose,
                           Vector4f viewIntrinsics, Vector2i viewImageSize, float distThresh)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

    int locId_local = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ float dim_shared[256];
    __shared__ float dim_shared0[256];
    __shared__ float dim_shared1[256];
    __shared__ float dim_shared2[256];
    __shared__ float dim_shared3[256];
    __shared__ float dim_shared4[256];
    __shared__ float dim_shared5[256];
    __shared__ float dim_shared6[256];
    __shared__ float dim_shared7[256];
    __shared__ float dim_shared8[256];
    __shared__ float dim_shared9[256];
    __shared__ float dim_shared10[256];
    __shared__ float dim_shared11[256];
    __shared__ float dim_shared12[256];
    __shared__ float dim_shared13[256];
    __shared__ float dim_shared14[256];
    __shared__ float dim_shared15[256];
    __shared__ float dim_shared16[256];
    __shared__ float dim_shared17[256];
    __shared__ float dim_shared18[256];
    __shared__ float dim_shared19[256];
    __shared__ float dim_shared20[256];
    __shared__ float dim_shared21[256];
    __shared__ float dim_shared22[256];
    __shared__ float dim_shared23[256];
    __shared__ float dim_shared24[256];
    __shared__ float dim_shared25[256];
    __shared__ float dim_shared26[256];
    __shared__ float dim_shared27[256];
    __shared__ float dim_shared28[256];
    __shared__ float dim_shared29[256];
    __shared__ float dim_shared30[256];
    __shared__ float dim_shared31[256];
    __shared__ float dim_shared32[256];
    __shared__ float dim_shared33[256];
    __shared__ float dim_shared34[256];
    __shared__ float dim_shared35[256];
    __shared__ bool should_prefix;

    should_prefix = false;
    __syncthreads();

    float A[36];
    bool isValidPoint = false;

    if (x < viewImageSize.x && y < viewImageSize.y)
    {
        isValidPoint = computePerPoint(A, x, y, depth[x + y * viewImageSize.x],
        viewImageSize, viewIntrinsics, sceneImageSize, sceneIntrinsics, approxInvPose, scenePose, pointsMap, normalsMap, distThresh);
        if (isValidPoint) should_prefix = true;
    }

    __syncthreads();

    if (!should_prefix) return;

    { //reduction for noValidPoints
        dim_shared[locId_local] = isValidPoint;
        __syncthreads();

        if (locId_local < 128) dim_shared[locId_local] += dim_shared[locId_local + 128];
        __syncthreads();
        if (locId_local < 64) dim_shared[locId_local] += dim_shared[locId_local + 64];
        __syncthreads();
        if (locId_local < 32) warpReduce(dim_shared, locId_local);
        if (locId_local == 0) atomicAdd(&(accu->numPoints), (int)dim_shared[locId_local]);
    }

    { //reduction for energy function value
        dim_shared0[locId_local] = A[0];
        dim_shared1[locId_local] = A[1];
        dim_shared2[locId_local] = A[2];
        dim_shared3[locId_local] = A[3];
        dim_shared4[locId_local] = A[4];
        dim_shared5[locId_local] = A[5];
        dim_shared6[locId_local] = A[6];
        dim_shared7[locId_local] = A[7];
        dim_shared8[locId_local] = A[8];
        dim_shared9[locId_local] = A[9];
        dim_shared10[locId_local] = A[10];
        dim_shared11[locId_local] = A[11];
        dim_shared12[locId_local] = A[12];
        dim_shared13[locId_local] = A[13];
        dim_shared14[locId_local] = A[14];
        dim_shared15[locId_local] = A[15];
        dim_shared16[locId_local] = A[16];
        dim_shared17[locId_local] = A[17];
        dim_shared18[locId_local] = A[18];
        dim_shared19[locId_local] = A[19];
        dim_shared20[locId_local] = A[20];
        dim_shared21[locId_local] = A[21];
        dim_shared22[locId_local] = A[22];
        dim_shared23[locId_local] = A[23];
        dim_shared24[locId_local] = A[24];
        dim_shared25[locId_local] = A[25];
        dim_shared26[locId_local] = A[26];
        dim_shared27[locId_local] = A[27];
        dim_shared28[locId_local] = A[28];
        dim_shared29[locId_local] = A[29];
        dim_shared30[locId_local] = A[30];
        dim_shared31[locId_local] = A[31];
        dim_shared32[locId_local] = A[32];
        dim_shared33[locId_local] = A[33];
        dim_shared34[locId_local] = A[34];
        dim_shared35[locId_local] = A[35];

        __syncthreads();

        if (locId_local < 128)
        {
            dim_shared0[locId_local] += dim_shared0[locId_local + 128];
            dim_shared1[locId_local] += dim_shared1[locId_local + 128];
            dim_shared2[locId_local] += dim_shared2[locId_local + 128];
            dim_shared3[locId_local] += dim_shared3[locId_local + 128];
            dim_shared4[locId_local] += dim_shared4[locId_local + 128];
            dim_shared5[locId_local] += dim_shared5[locId_local + 128];
            dim_shared6[locId_local] += dim_shared6[locId_local + 128];
            dim_shared7[locId_local] += dim_shared7[locId_local + 128];
            dim_shared8[locId_local] += dim_shared8[locId_local + 128];
            dim_shared9[locId_local] += dim_shared9[locId_local + 128];
            dim_shared10[locId_local] += dim_shared10[locId_local + 128];
            dim_shared11[locId_local] += dim_shared11[locId_local + 128];
            dim_shared12[locId_local] += dim_shared12[locId_local + 128];
            dim_shared13[locId_local] += dim_shared13[locId_local + 128];
            dim_shared14[locId_local] += dim_shared14[locId_local + 128];
            dim_shared15[locId_local] += dim_shared15[locId_local + 128];
            dim_shared16[locId_local] += dim_shared16[locId_local + 128];
            dim_shared17[locId_local] += dim_shared17[locId_local + 128];
            dim_shared18[locId_local] += dim_shared18[locId_local + 128];
            dim_shared19[locId_local] += dim_shared19[locId_local + 128];
            dim_shared20[locId_local] += dim_shared20[locId_local + 128];
            dim_shared21[locId_local] += dim_shared21[locId_local + 128];
            dim_shared22[locId_local] += dim_shared22[locId_local + 128];
            dim_shared23[locId_local] += dim_shared23[locId_local + 128];
            dim_shared24[locId_local] += dim_shared24[locId_local + 128];
            dim_shared25[locId_local] += dim_shared25[locId_local + 128];
            dim_shared26[locId_local] += dim_shared26[locId_local + 128];
            dim_shared27[locId_local] += dim_shared27[locId_local + 128];
            dim_shared28[locId_local] += dim_shared28[locId_local + 128];
            dim_shared29[locId_local] += dim_shared29[locId_local + 128];
            dim_shared30[locId_local] += dim_shared30[locId_local + 128];
            dim_shared31[locId_local] += dim_shared31[locId_local + 128];
            dim_shared32[locId_local] += dim_shared32[locId_local + 128];
            dim_shared33[locId_local] += dim_shared33[locId_local + 128];
            dim_shared34[locId_local] += dim_shared34[locId_local + 128];
            dim_shared35[locId_local] += dim_shared35[locId_local + 128];

        }
        __syncthreads();

        if (locId_local < 64)
        {
            dim_shared0[locId_local] += dim_shared0[locId_local + 64];
            dim_shared1[locId_local] += dim_shared1[locId_local + 64];
            dim_shared2[locId_local] += dim_shared2[locId_local + 64];
            dim_shared3[locId_local] += dim_shared3[locId_local + 64];
            dim_shared4[locId_local] += dim_shared4[locId_local + 64];
            dim_shared5[locId_local] += dim_shared5[locId_local + 64];
            dim_shared6[locId_local] += dim_shared6[locId_local + 64];
            dim_shared7[locId_local] += dim_shared7[locId_local + 64];
            dim_shared8[locId_local] += dim_shared8[locId_local + 64];
            dim_shared9[locId_local] += dim_shared9[locId_local + 64];
            dim_shared10[locId_local] += dim_shared10[locId_local + 64];
            dim_shared11[locId_local] += dim_shared11[locId_local + 64];
            dim_shared12[locId_local] += dim_shared12[locId_local + 64];
            dim_shared13[locId_local] += dim_shared13[locId_local + 64];
            dim_shared14[locId_local] += dim_shared14[locId_local + 64];
            dim_shared15[locId_local] += dim_shared15[locId_local + 64];
            dim_shared16[locId_local] += dim_shared16[locId_local + 64];
            dim_shared17[locId_local] += dim_shared17[locId_local + 64];
            dim_shared18[locId_local] += dim_shared18[locId_local + 64];
            dim_shared19[locId_local] += dim_shared19[locId_local + 64];
            dim_shared20[locId_local] += dim_shared20[locId_local + 64];
            dim_shared21[locId_local] += dim_shared21[locId_local + 64];
            dim_shared22[locId_local] += dim_shared22[locId_local + 64];
            dim_shared23[locId_local] += dim_shared23[locId_local + 64];
            dim_shared24[locId_local] += dim_shared24[locId_local + 64];
            dim_shared25[locId_local] += dim_shared25[locId_local + 64];
            dim_shared26[locId_local] += dim_shared26[locId_local + 64];
            dim_shared27[locId_local] += dim_shared27[locId_local + 64];
            dim_shared28[locId_local] += dim_shared28[locId_local + 64];
            dim_shared29[locId_local] += dim_shared29[locId_local + 64];
            dim_shared30[locId_local] += dim_shared30[locId_local + 64];
            dim_shared31[locId_local] += dim_shared31[locId_local + 64];
            dim_shared32[locId_local] += dim_shared32[locId_local + 64];
            dim_shared33[locId_local] += dim_shared33[locId_local + 64];
            dim_shared34[locId_local] += dim_shared34[locId_local + 64];
            dim_shared35[locId_local] += dim_shared35[locId_local + 64];

        }

        __syncthreads();

        if (locId_local < 32)
        {
            warpReduce(dim_shared0, locId_local);
            warpReduce(dim_shared1, locId_local);
            warpReduce(dim_shared2, locId_local);
            warpReduce(dim_shared3, locId_local);
            warpReduce(dim_shared4, locId_local);
            warpReduce(dim_shared5, locId_local);
            warpReduce(dim_shared6, locId_local);
            warpReduce(dim_shared7, locId_local);
            warpReduce(dim_shared8, locId_local);
            warpReduce(dim_shared9, locId_local);
            warpReduce(dim_shared10, locId_local);
            warpReduce(dim_shared11, locId_local);
            warpReduce(dim_shared12, locId_local);
            warpReduce(dim_shared13, locId_local);
            warpReduce(dim_shared14, locId_local);
            warpReduce(dim_shared15, locId_local);
            warpReduce(dim_shared16, locId_local);
            warpReduce(dim_shared17, locId_local);
            warpReduce(dim_shared18, locId_local);
            warpReduce(dim_shared19, locId_local);
            warpReduce(dim_shared20, locId_local);
            warpReduce(dim_shared21, locId_local);
            warpReduce(dim_shared22, locId_local);
            warpReduce(dim_shared23, locId_local);
            warpReduce(dim_shared24, locId_local);
            warpReduce(dim_shared25, locId_local);
            warpReduce(dim_shared26, locId_local);
            warpReduce(dim_shared27, locId_local);
            warpReduce(dim_shared28, locId_local);
            warpReduce(dim_shared29, locId_local);
            warpReduce(dim_shared30, locId_local);
            warpReduce(dim_shared31, locId_local);
            warpReduce(dim_shared32, locId_local);
            warpReduce(dim_shared33, locId_local);
            warpReduce(dim_shared34, locId_local);
            warpReduce(dim_shared35, locId_local);
        }

        if (locId_local == 0)
        {
            atomicAdd(&(accu->g[0]), dim_shared0[locId_local]);
            atomicAdd(&(accu->g[1]), dim_shared1[locId_local]);
            atomicAdd(&(accu->g[2]), dim_shared2[locId_local]);
            atomicAdd(&(accu->g[3]), dim_shared3[locId_local]);
            atomicAdd(&(accu->g[4]), dim_shared4[locId_local]);
            atomicAdd(&(accu->g[5]), dim_shared5[locId_local]);
            atomicAdd(&(accu->g[6]), dim_shared6[locId_local]);
            atomicAdd(&(accu->g[7]), dim_shared7[locId_local]);
            atomicAdd(&(accu->g[8]), dim_shared8[locId_local]);
            atomicAdd(&(accu->g[9]), dim_shared9[locId_local]);
            atomicAdd(&(accu->g[10]), dim_shared10[locId_local]);
            atomicAdd(&(accu->g[11]), dim_shared11[locId_local]);
            atomicAdd(&(accu->g[12]), dim_shared12[locId_local]);
            atomicAdd(&(accu->g[13]), dim_shared13[locId_local]);
            atomicAdd(&(accu->g[14]), dim_shared14[locId_local]);
            atomicAdd(&(accu->g[15]), dim_shared15[locId_local]);
            atomicAdd(&(accu->g[16]), dim_shared16[locId_local]);
            atomicAdd(&(accu->g[17]), dim_shared17[locId_local]);
            atomicAdd(&(accu->g[18]), dim_shared18[locId_local]);
            atomicAdd(&(accu->g[19]), dim_shared19[locId_local]);
            atomicAdd(&(accu->g[20]), dim_shared20[locId_local]);
            atomicAdd(&(accu->g[21]), dim_shared21[locId_local]);
            atomicAdd(&(accu->g[22]), dim_shared22[locId_local]);
            atomicAdd(&(accu->g[23]), dim_shared23[locId_local]);
            atomicAdd(&(accu->g[24]), dim_shared24[locId_local]);
            atomicAdd(&(accu->g[25]), dim_shared25[locId_local]);
            atomicAdd(&(accu->g[26]), dim_shared26[locId_local]);
            atomicAdd(&(accu->g[27]), dim_shared27[locId_local]);
            atomicAdd(&(accu->g[28]), dim_shared28[locId_local]);
            atomicAdd(&(accu->g[29]), dim_shared29[locId_local]);
            atomicAdd(&(accu->g[30]), dim_shared30[locId_local]);
            atomicAdd(&(accu->g[31]), dim_shared31[locId_local]);
            atomicAdd(&(accu->g[32]), dim_shared32[locId_local]);
            atomicAdd(&(accu->g[33]), dim_shared33[locId_local]);
            atomicAdd(&(accu->g[34]), dim_shared34[locId_local]);
            atomicAdd(&(accu->g[35]), dim_shared35[locId_local]);
        }
    }
}

void ITMEKF_CUDA::covmat(ITMTrackingState *trackingState, const ITMView *view, bool &validpoint, float *covN)
{
    Matrix4f approxInvPose = trackingState->pose_d->GetInvM();
    float *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
    Vector4f *pointsMap = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA);
    Vector4f *normalsMap = trackingState->pointCloud->colours->GetData(MEMORYDEVICE_CUDA);
    Vector4f sceneIntrinsics = view->calib->intrinsics_d.projectionParamsSimple.all;
    Vector2i sceneImageSize = trackingState->pointCloud->locations->noDims;
    Vector4f viewIntrinsics = view->calib->intrinsics_d.projectionParamsSimple.all;
    Vector2i viewImageSize = view->depth->noDims;
    Matrix4f scenePose  = trackingState->pose_pointCloud->GetM();
    float distThresh = 0.1f * 0.1f;
    dim3 blockSize(16, 16);
    dim3 gridSize((int)ceil((float)viewImageSize.x / (float)blockSize.x), (int)ceil((float)viewImageSize.y / (float)blockSize.y));

    ITMSafeCall(cudaMemset(accu_device1, 0, sizeof(accuCell)));
    g_rt_device << <gridSize, blockSize >> >(accu_device1, depth, approxInvPose, pointsMap, normalsMap,
                                            sceneIntrinsics, sceneImageSize, scenePose, viewIntrinsics,
                                            viewImageSize, distThresh);
    ITMSafeCall(cudaMemcpy(accu_host1, accu_device1, sizeof(accuCell), cudaMemcpyDeviceToHost));
    /*printf("%f,%f,%f,%f,%f,%f,%f,%f,%f\n",accu_host1->g[0],accu_host1->g[1],accu_host1->g[2],accu_host1->g[3],
            accu_host1->g[4],accu_host1->g[5],accu_host1->g[6],accu_host1->g[7],accu_host1->g[8]);*/

    memcpy(covN, accu_host1->g, 36 * sizeof(float));
    validpoint = accu_host1->numPoints/(viewImageSize.x * viewImageSize.y)>0.8;
}
/*com cov end*/











