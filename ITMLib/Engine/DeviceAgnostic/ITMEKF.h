// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../Utils/ITMLibDefines.h"
#include "ITMPixelUtils.h"

_CPU_AND_GPU_CODE_  inline bool computePerPoint(THREADPTR(float) *A,const THREADPTR(int) & x, const THREADPTR(int) & y,
                 const CONSTPTR(float) &depth, const CONSTPTR(Vector2i) & viewImageSize, const CONSTPTR(Vector4f) & viewIntrinsics, const CONSTPTR(Vector2i) & sceneImageSize,
                 const CONSTPTR(Vector4f) & sceneIntrinsics, const CONSTPTR(Matrix4f) & approxInvPose, const CONSTPTR(Matrix4f) & scenePose, const CONSTPTR(Vector4f) *pointsMap,
                 const CONSTPTR(Vector4f) *normalsMap, float distThresh)
{
    if (depth <= 1e-8f) return false; //check if valid -- != 0.0f
    Vector4f tmp3Dpoint, t3Dpoint, tmp3Dpoint_reproj; Vector3f ptDiff;
    Vector4f curr3Dpoint, corr3Dnormal; Vector2f tmp2Dpoint;

    tmp3Dpoint.x = depth * ((float(x) - viewIntrinsics.z) / viewIntrinsics.x);
    tmp3Dpoint.y = depth * ((float(y) - viewIntrinsics.w) / viewIntrinsics.y);
    tmp3Dpoint.z = depth;
    tmp3Dpoint.w = 1.0f;

    /*t3Dpoint.x = tmp3Dpoint.x;
    t3Dpoint.y = tmp3Dpoint.y;
    t3Dpoint.z = tmp3Dpoint.z;
    t3Dpoint.w = tmp3Dpoint.w;*/

// transform to previous frame coordinates
    tmp3Dpoint = approxInvPose * tmp3Dpoint;
    tmp3Dpoint.w = 1.0f;

// project into previous rendered image
    tmp3Dpoint_reproj = scenePose * tmp3Dpoint;
    if (tmp3Dpoint_reproj.z <= 0.0f) return false;
    tmp2Dpoint.x = sceneIntrinsics.x * tmp3Dpoint_reproj.x / tmp3Dpoint_reproj.z + sceneIntrinsics.z;
    tmp2Dpoint.y = sceneIntrinsics.y * tmp3Dpoint_reproj.y / tmp3Dpoint_reproj.z + sceneIntrinsics.w;

    if (!((tmp2Dpoint.x >= 0.0f) && (tmp2Dpoint.x <= sceneImageSize.x - 2) && (tmp2Dpoint.y >= 0.0f) && (tmp2Dpoint.y <= sceneImageSize.y - 2)))
        return false;

    curr3Dpoint = interpolateBilinear_withHoles(pointsMap, tmp2Dpoint, sceneImageSize);
    if (curr3Dpoint.w < 0.0f) return false;

    ptDiff.x = curr3Dpoint.x - tmp3Dpoint.x;
    ptDiff.y = curr3Dpoint.y - tmp3Dpoint.y;
    ptDiff.z = curr3Dpoint.z - tmp3Dpoint.z;
    float dist = ptDiff.x * ptDiff.x + ptDiff.y * ptDiff.y + ptDiff.z * ptDiff.z;

    if (dist > distThresh) return false;

    corr3Dnormal = interpolateBilinear_withHoles(normalsMap, tmp2Dpoint, sceneImageSize);
    Vector3f r,r1;
    r.x = curr3Dpoint.y * corr3Dnormal.z - curr3Dpoint.z * corr3Dnormal.y;
    r.y = curr3Dpoint.z * corr3Dnormal.x - curr3Dpoint.x * corr3Dnormal.z;
    r.z = curr3Dpoint.x * corr3Dnormal.y - curr3Dpoint.y * corr3Dnormal.x;

    A[0] = r.x * r.x;
    A[1] = r.y * r.x;
    A[2] = r.z * r.x;
    A[6] = r.x * r.y;
    A[7] = r.y * r.y;
    A[8] = r.z * r.y;
    A[12] = r.x * r.z;
    A[13] = r.y * r.z;
    A[14] = r.z * r.z;
    A[3] = corr3Dnormal.x * r.x;
    A[4] = corr3Dnormal.y * r.y;
    A[5] = corr3Dnormal.z * r.x;
    A[9] = corr3Dnormal.x * r.y;
    A[10] = corr3Dnormal.y * r.y;
    A[11] = corr3Dnormal.z * r.y;
    A[15] = corr3Dnormal.x * r.z;
    A[16] = corr3Dnormal.y * r.z;
    A[17] = corr3Dnormal.z * r.z;
    A[18] = r.x * corr3Dnormal.x;
    A[19] = r.y * corr3Dnormal.x;
    A[20] = r.z * corr3Dnormal.x;
    A[24] = r.x * corr3Dnormal.y;
    A[25] = r.y * corr3Dnormal.y;
    A[26] = r.z * corr3Dnormal.y;
    A[30] = r.x * corr3Dnormal.z;
    A[31] = r.y * corr3Dnormal.z;
    A[32] = r.z * corr3Dnormal.z;
    A[21] = corr3Dnormal.x * corr3Dnormal.x;
    A[22] = corr3Dnormal.y * corr3Dnormal.x;
    A[23] = corr3Dnormal.z * corr3Dnormal.x;
    A[27] = corr3Dnormal.x * corr3Dnormal.y;
    A[28] = corr3Dnormal.y * corr3Dnormal.y;
    A[29] = corr3Dnormal.z * corr3Dnormal.y;
    A[33] = corr3Dnormal.x * corr3Dnormal.z;
    A[34] = corr3Dnormal.y * corr3Dnormal.z;
    A[35] = corr3Dnormal.z * corr3Dnormal.z;
    return true;
}
