#include "ITMEKF.h"
#include <TooN/TooN.h>
#include <TooN/se3.h>

using namespace ITMLib::Engine;


#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

ITMEKF::ITMEKF()
{
    for (int i = 0; i < 36; i++)	this->constM.m[i] = 0.0f;
    this->constM.m[0] = 0.007f*0.007f;
    this->constM.m[7] = 0.008f*0.008f;
    this->constM.m[14] = 0.008f*0.008f;
    this->constM.m[21] = 0.017f*0.017f;
    this->constM.m[28] = 0.05f*0.05f;
    this->constM.m[35] = 0.141f*0.141f;
}

ITMEKF::~ITMEKF()
{

}
void ITMEKF::multicublas(float* src1, float*src2,float* dst, int n1,int n2,int n3)
{
    float* src_d1,*src_d2, *dst_d;

    cudacall(cudaMalloc((void**)&src_d1,n1 * n2 * sizeof(float)));
    cudacall(cudaMemcpy(src_d1,src1,n1 * n2 * sizeof(float),cudaMemcpyHostToDevice));
    cudacall(cudaMalloc((void**)&src_d2,n2 * n3 * sizeof(float)));
    cudacall(cudaMemcpy(src_d2,src2,n2 * n3 * sizeof(float),cudaMemcpyHostToDevice));
    cudacall(cudaMalloc((void**)&dst_d,n3 * n1 * sizeof(float)));
    cudacall(cudaMemset(dst_d,0,n3 * n1 * sizeof(float)));
    /*float* C = dst;
    cudacall(cudaMalloc<float>(&C,n * n * sizeof(float)));*/
    cublasHandle_t handle;
    cublascall(cublasCreate_v2(&handle));

    float alpha=1.0;
    float beta=0.0;

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n3,n1,n2,&alpha,src_d2,n3,src_d1,n2,&beta,dst_d,n3);
    cudacall(cudaMemcpy(dst,dst_d,n3 * n1 * sizeof(float),cudaMemcpyDeviceToHost));
//printf("%f",*(dst+2));
    cublasDestroy_v2(handle);

    cudaFree(src_d1);
    cudaFree(src_d2);
    cudaFree(dst_d);

}

void ITMEKF::invert(float* src, float* dst, int n)
{
    float* src_d, *dst_d;

    cudacall(cudaMalloc<float>(&src_d,n * n * sizeof(float)));
    cudacall(cudaMemcpy(src_d,src,n * n * sizeof(float),cudaMemcpyHostToDevice));
    cudacall(cudaMalloc<float>(&dst_d,n * n * sizeof(float)));

    cublasHandle_t handle;
    cublascall(cublasCreate_v2(&handle));

    int batchSize = 1;

    int *P, *INFO;

    cudacall(cudaMalloc<int>(&P,n * batchSize * sizeof(int)));
    cudacall(cudaMalloc<int>(&INFO,batchSize * sizeof(int)));

    int lda = n;

    float *A[] = { src_d };
    float** A_d;
    cudacall(cudaMalloc<float*>(&A_d,sizeof(A)));
    cudacall(cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice));
    cublascall(cublasSgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize));

    int INFOh = 0;
    cudacall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh == n)
    {
        fprintf(stderr, "Factorization Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    float* C[] = { dst_d };
    float** C_d;
    cudacall(cudaMalloc<float*>(&C_d,sizeof(C)));
    cudacall(cudaMemcpy(C_d,C,sizeof(C),cudaMemcpyHostToDevice));

    cublascall(cublasSgetriBatched(handle,n,(const float**)A_d,lda,P,C_d,lda,INFO,batchSize));

    cudacall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh != 0)
    {
        fprintf(stderr, "Inversion Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    cudaFree(P);
    cudaFree(INFO);
    cublasDestroy_v2(handle);

    cudacall(cudaMemcpy(dst,dst_d,n * n * sizeof(float),cudaMemcpyDeviceToHost));

    cudaFree(src_d);
    cudaFree(dst_d);
    cudaFree(C_d);
    cudaFree(A_d);
}

void ITMEKF::tranS(const Matrix3f & R, Vector3f &p1)//invS
{
    p1.v[0] = R.m[2 + 3*1];
    p1.v[1] = R.m[0 + 3*2];
    p1.v[2] = R.m[1 + 3*0];
}

void ITMEKF::splitRT(const Matrix4f & M, Matrix3f & R, Vector3f & T)//4to3+1
{
    R.m[0 + 3*0] = M.m[0 + 4*0]; R.m[1 + 3*0] = M.m[1 + 4*0]; R.m[2 + 3*0] = M.m[2 + 4*0];
    R.m[0 + 3*1] = M.m[0 + 4*1]; R.m[1 + 3*1] = M.m[1 + 4*1]; R.m[2 + 3*1] = M.m[2 + 4*1];
    R.m[0 + 3*2] = M.m[0 + 4*2]; R.m[1 + 3*2] = M.m[1 + 4*2]; R.m[2 + 3*2] = M.m[2 + 4*2];
    T.v[0] = M.m[0 + 4*3]; T.v[1] = M.m[1 + 4*3]; T.v[2] = M.m[2 + 4*3];
}

void ITMEKF::proM(Matrix3f & R,const Vector3f & T, Matrix4f &p2)//com pro
{
    Matrix3f RT = R.t();
    for(int i=0;i<9;i++)
        RT.m[i] = R.m[i] + RT.m[i] * (-1.0f);
    Matrix3f tmp;
    for(int i=0;i<9;i++)
        tmp.m[i] = RT.m[i] * 0.5f;
    p2.m[0 + 4*0] = tmp.m[0 + 3*0];
    p2.m[1 + 4*0] = tmp.m[1 + 3*0];
    p2.m[2 + 4*0] = tmp.m[2 + 3*0];
    p2.m[3 + 4*0] = 0;
    p2.m[0 + 4*1] = tmp.m[0 + 3*1];
    p2.m[1 + 4*1] = tmp.m[1 + 3*1];
    p2.m[2 + 4*1] = tmp.m[2 + 3*1];
    p2.m[3 + 4*1] = 0;
    p2.m[0 + 4*2] = tmp.m[0 + 3*2];
    p2.m[1 + 4*2] = tmp.m[1 + 3*2];
    p2.m[2 + 4*2] = tmp.m[2 + 3*2];
    p2.m[3 + 4*2] = 0;
    p2.m[0 + 4*3] = T.v[0];
    p2.m[1 + 4*3] = T.v[1];
    p2.m[2 + 4*3] = T.v[2];
    p2.m[3 + 4*3] = 0;
}

void ITMEKF::tranSinv(const Vector3f & q,Matrix3f & M)//com S
{
    M.setZeros();
    //memset(M.m, 0.0f, sizeof(float) * 9);
    M.m[1] = q.v[2];
    M.m[2] = (-1.0f)*q.v[1];
    M.m[3] = (-1.0f)*q.v[2];
    M.m[5] = q.v[0];
    M.m[6] = q.v[1];
    M.m[7] = (-1.0f)*q.v[0];
}
void ITMEKF::proH(const Vector3f & R,const Vector3f & T, Matrix4f & H)// com H
{
    Matrix3f M;
    tranSinv(R, M);
    H.m[0 + 4*0] = M.m[0 + 3*0];
    H.m[1 + 4*0] = M.m[1 + 3*0];
    H.m[2 + 4*0] = M.m[2 + 3*0];
    H.m[3 + 4*0] = 0;
    H.m[0 + 4*1] = M.m[0 + 3*1];
    H.m[1 + 4*1] = M.m[1 + 3*1];
    H.m[2 + 4*1] = M.m[2 + 3*1];
    H.m[3 + 4*1] = 0;
    H.m[0 + 4*2] = M.m[0 + 3*2];
    H.m[1 + 4*2] = M.m[1 + 3*2];
    H.m[2 + 4*2] = M.m[2 + 3*2];
    H.m[3 + 4*2] = 0;
    H.m[0 + 4*3] = T.v[0];
    H.m[1 + 4*3] = T.v[1];
    H.m[2 + 4*3] = T.v[2];
    H.m[3 + 4*3] = 0;
}
void ITMEKF::proHinv(const Matrix4f &H, Vector6f & Xh)// com invH
{
    Matrix3f R;
    Vector3f xr;
    Vector3f xt;
    R.setZeros();
    splitRT(H, R, xt);
    tranS(R,xr);
    Xh.v[0] = xr.v[0];
    Xh.v[1] = xr.v[1];
    Xh.v[2] = xr.v[2];
    Xh.v[3] = xt.v[0];
    Xh.v[4] = xt.v[1];
    Xh.v[5] = xt.v[2];
}

void ITMEKF::expmat(float* R, float* T, float* dst)
{
    TooN::Vector<3,float> rotation;
    for(int i=0;i<3;i++)
    {
        rotation[i] = *(R + i);
        //std::cout << rotation[i] <<'\n';
    }
    TooN::Vector<3,float> trans;
    for(int i=0;i<3;i++)
    {
        trans[i] = *(T + i);
        //std::cout << trans[i] <<'\n';
    }
    TooN::Vector<6,float> mu;
    mu.slice(0,3) = trans;
    mu.slice(3,3) = rotation;

    TooN::SE3<> se = TooN::SE3<>::exp(mu);

    TooN::SO3<> t = se.get_rotation();
    TooN::Matrix<3,3,float> Rf= t.get_matrix();
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
    {
        //std::cout << Rf(i,j) << '\n';
            *dst++ = Rf(j,i);
    }
}

void ITMEKF::compRT(float dt,Matrix6f &Pk, const Matrix4f & Xk,const Matrix4f & Yk, Matrix3f &curR, Vector3f &curT, bool &mattruth)//main func
{
    //Matrix6f Pk = trackingState->Pk;
    //dt = dt/1000.0f;
    mattruth = true;
    Matrix6f k;
    //k.setZeros();
    for (int i = 0; i < 36; i++)	{k.m[i] = 0.0f;
        //printf("%f,",Pk.m[i]);
        //printf("\n");
    }
    //float covN[36];
    //float *a;// = (Pk + constM).m;
    float a[36];
    for (int i = 0; i < 36; i++)
            {
            //covN[i] = 0.01f * 0.01f * covN[i] * dt;
       // printf("%f,",Pk.m[i]);
       // printf("\n");
       a[i] = (Pk + constM).m[i];
        //printf("%f,",*(a+i));
        //printf("\n");
            }
    TooN::Matrix<6,6,float> mat;
    int c=0;
    for(int i=0;i<6;i++)
        for(int j=0;j<6;j++)
    {
        mat(j,i) = a[c];
        //std::cout << mat(i,j) <<'\n';
        ++c;
    }
    float dst1;
    dst1 = TooN::determinant_gaussian_elimination(mat);
    if(dst1!=0.0f)
        {
    invert(a,a,6);}
    else{
        mattruth = false;
        return;
    }
    for(int i=0;i<36;i++)
    {
        k.m[i]=a[i];
        //printf("%f,",*(a+i));
        //printf("\n");
    }
    k = Pk * k;
    dt = 1.0f/dt;
    //printf("dt%f\n",dt);
    //k = k * dt;
    for (int i = 0; i < 36; i++)
        {
        k.m[i] = k.m[i] * dt;
        //printf("%f,",k.m[i]);
        //printf("\n");
        }
    Matrix4f tmp1;
    Vector6f tmp2;
    Matrix4f Xkinv;
    Xk.inv(Xkinv);
    /*for (int i = 0; i < 16; i++)
        {
        //k.m[i] = k.m[i] * dt;
        printf("%f,",Xkinv.m[i]);
        printf("\n");
        }*/
    Xkinv = Yk * Xkinv;
    /*for (int i = 0; i < 16; i++)
            {
            //k.m[i] = k.m[i] * dt;
            printf("%f,",Xkinv.m[i]);
            printf("\n");
            }*/
    Matrix3f tmp3;
    Vector3f tmp4;
    //float conN1[6];
    splitRT(Xkinv, tmp3, tmp4);
    /*for (int i = 0; i < 9; i++)
                {
                //k.m[i] = k.m[i] * dt;
                printf("%f,",tmp3.m[i]);
                printf("\n");
                }
    for (int i = 0; i < 3; i++)
                {
                //k.m[i] = k.m[i] * dt;
                printf("%f,",tmp4.v[i]);
                printf("\n");
                }*/
    proM(tmp3,tmp4,tmp1);
    /*for (int i = 0; i < 16; i++)
                    {
                    //k.m[i] = k.m[i] * dt;
                    printf("%f,",tmp1.m[i]);
                    printf("\n");
                    }*/
    proHinv(tmp1,tmp2);
    /*for (int i = 0; i < 6; i++)
                        {
                        //k.m[i] = k.m[i] * dt;
                        printf("%f,",tmp2.v[i]);
                        printf("\n");
                        }*/
    float aa[36];
    float a1[6];
    for (int i = 0; i < 36; i++)
            {
            //covN[i] = 0.01f * 0.01f * covN[i] * dt;
       aa[i] = k.m[i];
            }
    for (int i = 0; i < 6; i++)
            {
            //covN[i] = 0.01f * 0.01f * covN[i] * dt;
       a1[i] = tmp2.v[i];
            }
    //aa = k.m;
    //a1 = tmp2.v;
    multicublas(aa,a1,a1,6,6,1);
    Vector3f R1, T1;
    R1.v[0] = a1[0];
    R1.v[1] = a1[1];
    R1.v[2] = a1[2];
    T1.v[0] = a1[3];
    T1.v[1] = a1[4];
    T1.v[2] = a1[5];
    proH(R1,T1,tmp1);
    Matrix3f er;
    Vector3f et;
    splitRT(tmp1,er,et);
    //printf("%f,%f,%f,%f,%f,%f,%f,%f,%f\n",er.m[0],er.m[1],er.m[2],er.m[3],er.m[4],er.m[5],er.m[6],er.m[7],er.m[8]);
    //Matrix3f curR;
    //Vector3f curT;
    Matrix3f preR;
    Vector3f preT;
    splitRT(Yk,preR,preT);
    //curT = preT + (er * preT + et) * dt;
    /*for (int i = 0; i < 9; i++)
    {
        printf("%f,",er.m[i]);
    }
    printf("\n");*/
    /*for (int i=0;i<3;i++)
    {
        //et.v[i] = et.v[i] * dt;
        printf("%f,",et.v[i]);
    }
    printf("\n");*/

    et = er * preT + et;
    for (int i=0;i<3;i++)
    {
        et.v[i] = et.v[i] * dt;
    }
    curT = preT + et;
    //curR = er * dt;
    Vector3f tmper1;
    for (int i = 0; i < 9; i++)
        {curR.m[i] = curR.m[i]*dt;}
    tranS(curR,tmper1);
    //float tmp3vec[3];
    float expa[3];
    for (int i=0;i<3;i++)
    {
        expa[i] = tmper1.v[i];
    }
    float expb[3];
    for (int i=0;i<3;i++)
    {
        expb[i] = et.v[i];
    }
    expmat(expa,expb,curR.m);
        //printf("%f,",curR.m[i]);}
    curR = preR * curR;
    /*for (int i = 0; i < 9; i++)
    {
        printf("%f,",curR.m[i]);
        printf("prev%f,",preR.m[i]);
    }
    printf("\n");
    for (int i=0;i<3;i++)
    {
        //et.v[i] = et.v[i] * dt;
        printf("%f,",curT.v[i]);
        printf("prevT%f,",preT.v[i]);
    }
    printf("\n");*/
    //trackingState->pose_d->SetRT(curR,curT);
    //trackingState->pose_d->Coerce();
}
void ITMEKF::updateP(ITMTrackingState *trackingState,const ITMView *view,float dt)
{
    Matrix6f Pk1 = trackingState->Pk;
    /*for (int i = 0; i < 36; i++)
            {
            //covN[i] = 0.01f * 0.01f * covN[i] * dt;
            printf("%f,",Pk1.m[i]);
            printf("\n");
            }*/
    float covN[36];
    bool validpoint;
    covmat(trackingState,view,validpoint,covN);
   /* for (int i = 0; i < 36; i++)
        {
        //covN[i] = 0.01f * 0.01f * covN[i] * dt;
        printf("%f,",covN[i]);
        printf("\n");
        }*/
    float a1[36];
    TooN::Matrix<6,6,float> mat;
    int c=0;
    for(int i=0;i<6;i++)
        for(int j=0;j<6;j++)
    {
        mat(j,i) = a1[c];
        ++c;
        //std::cout << mat(i,j) <<'\n';
    }
    float dst1;
    dst1 = TooN::determinant_gaussian_elimination(mat);
    if(dst1!=0.0f)
    {
    //invert(a,a,6);

    invert(covN,covN,6);
    /*for (int i = 0; i < 36; i++)
        {
        //covN[i] = 0.01f * 0.01f * covN[i] * dt;
        printf("%f,",covN[i]);
        printf("\n");
        }*/
    for (int i = 0; i < 36; i++)
        {
        covN[i] = 0.01f * 0.01f * covN[i] * dt;
        //printf("%f,",covN[i]);
        //printf("\n");
        }

    Matrix6f N;
    for(int i=0;i<36;i++)
    {
        N.m[i]=covN[i];
    }
    //N.m = covN;
    float a[36];// = &((Pk1 + constM).m[0]);
    //a=(float*)malloc(10*sizeof(float));
    //memcpy(a, (Pk1 + constM).m, 36 * sizeof(float));
    for (int i = 0; i < 36; i++)
            {
            //covN[i] = 0.01f * 0.01f * covN[i] * dt;
       a[i] = (Pk1 + constM).m[i];
            }
    /*for (int i = 0; i < 36; i++)
            {
            //covN[i] = 0.01f * 0.01f * covN[i] * dt;
        printf("pk%f,",Pk1.m[i]);
        printf("M%f,",constM.m[i]);
            printf("%f,",*(a+i));
            printf("\n");
            }*/
    TooN::Matrix<6,6,float> mat;
    c=0;
    for(int i=0;i<6;i++)
        for(int j=0;j<6;j++)
    {
        mat(j,i) = a[c];
        //std::cout << mat(i,j) <<'\n';
        ++c;
    }
    float dst1;
    dst1 = TooN::determinant_gaussian_elimination(mat);
    if(dst1!=0.0f)
        {

    invert(a,a,6);
    /*for (int i = 0; i < 36; i++)
                {
                //covN[i] = 0.01f * 0.01f * covN[i] * dt;
            //printf("pk%f,",Pk1.m[i]);
            //printf("M%f,",constM.m[i]);
                //printf("%f,",*(a+i));
                }*/
    Matrix6f k;
    for(int i=0;i<36;i++)
    {
        k.m[i]=a[i];
        //printf("%f",k.m[i]);
        //printf("\n");
    }
    Matrix6f k1;
    k1 = Pk1 * k * Pk1;
    for(int i=0;i<36;i++)
    {
        k1.m[i]= (-1.0f) * k1.m[i];
        //printf("%f,",N.m[i]);
        //printf("%f",k1.m[i]);
        //printf("\n");
    }
    trackingState->Pk = N + Pk1 + k1;
    }
    }
    /*for(int i=0;i<36;i++)
    {
        //k1.m[i]= (-1.0f) * k1.m[i];
        printf("%f",trackingState->Pk.m[i]);
        printf("\n");
    }*/
}
