#pragma once

//#include "../Utils/ITMLibDefines.h"
#include "../../ORUtils/CUDADefines.h"
//#include <cstdio>
#include <cstdlib>
//#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../Objects/ITMTrackingState.h"
#include "../Objects/ITMView.h"
//#include "../Engine/ITMLowLevelEngine.h"
//#include "../Engine/DeviceAgnostic/ITMPixelUtils.h"
//#include "../Engine/DeviceSpecific/CUDA/ITMCUDAUtils.h"
#include "../../ORUtils/PlatformIndependence.h"

using namespace ITMLib::Objects;


namespace ITMLib
{
    namespace Engine
    {
        class ITMEKF
        {
        private:
            void tranS(const Matrix3f & R, Vector3f &p1);//invS
            void splitRT(const Matrix4f & M, Matrix3f & R, Vector3f & T);//4to3+1
            void proM(Matrix3f & R,const Vector3f & T, Matrix4f &p2);
            void tranSinv(const Vector3f & q,Matrix3f & M);//com S
            void proH(const Vector3f & R,const Vector3f & T, Matrix4f & H);// com H
            void proHinv(const Matrix4f &H, Vector6f & Xh);// com invH
            Matrix6f constM;
            void expmat(float* R, float* T, float* dst);
        public:
            ITMEKF();
            virtual ~ITMEKF(void);
            void invert(float* src, float* dst, int n);
            void multicublas(float* src1, float*src2,float* dst, int n1,int n2,int n3);
            void compRT(float dt, Matrix6f &Pk, const Matrix4f &Xk,const Matrix4f &Yk,Matrix3f &curR, Vector3f &curT, bool &mattruth);
            virtual void covmat(ITMTrackingState *trackingState, const ITMView *view, bool &validpoint, float *covN)=0;
            void updateP(ITMTrackingState *trackingState,const ITMView *view,float dt);
        };
    }
}
