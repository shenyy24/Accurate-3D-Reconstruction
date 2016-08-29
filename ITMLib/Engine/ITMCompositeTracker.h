// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Utils/ITMLibDefines.h"
#include "../Engine/ITMTracker.h"
//#include "../Engine/ITMEKF.h"
#include "DeviceSpecific/CUDA/ITMEKF_CUDA.h"

using namespace ITMLib::Objects;

namespace ITMLib
{
    namespace Engine
    {
        class ITMCompositeTracker : public ITMTracker
        {
        private:
            ITMTracker **trackers; int noTrackers;
            ITMEKF_CUDA *iekf;
            //ITMEKF *iekf;
        public:

            void SetTracker(ITMTracker *tracker, int trackerId)
            {
                if (trackers[trackerId] != NULL) delete trackers[trackerId];
                trackers[trackerId] = tracker;
            }

            ITMCompositeTracker(int noTrackers)
            {
                trackers = new ITMTracker*[noTrackers];
                for (int i = 0; i < noTrackers; i++) trackers[i] = NULL;

                this->noTrackers = noTrackers;
                //iekf = new ITMEKF_CUDA();
                iekf = new ITMEKF_CUDA();
            }

            ~ITMCompositeTracker(void)
            {
                delete iekf;
                for (int i = 0; i < noTrackers; i++)
                    if (trackers[i] != NULL) delete trackers[i];

                delete [] trackers;
                //delete iekf;
            }

            void TrackCamera(ITMTrackingState *trackingState, const ITMView *view)
            {
                Matrix4f tmp1,tmp2;
                float dt = trackingState->dt*1000.0f;
                for (int i = 0; i < noTrackers; i++)
                {
                    trackers[i]->TrackCamera(trackingState, view);
                    if(i==0)
                    {
                        tmp1 = trackingState->pose_d->GetM();
                        /*if(dt>0.0f){
                        iekf->updateP(trackingState,view,dt);}*/
                        //printf("*f,%f,%f\n",tmp1.m[12],tmp1.m[13],tmp1.m[14]);
                    }
                    if(i==1)
                    {
                        tmp2 = trackingState->pose_d->GetM();
                    }
                }
                //ITMTrackingState *trackingState1 = trackingState;

                /*Matrix6f Pk = trackingState->Pk;
                //printf("init%f\n",trackingState->Pk.m[2]);
                //printf("tracking%f\n",trackingState->dt);
                if(dt > 0.0f)
                {
                    Matrix3f curR;
                    Vector3f curT;
                    bool mattruth;

                    iekf->compRT(dt,Pk,tmp1,tmp2,curR,curT,mattruth);
                    //iekf->updateP(trackingState,view,dt);
                    if(mattruth)
                    {
                        iekf->updateP(trackingState,view,dt);
                        //trackingState->pose_d->SetRT(curR,curT);
                        //trackingState->pose_d->Coerce();
                        //trackingState->pose_d->SetM(tmp1);
                    }
                    //iekf->updateP(trackingState1,view,dt);
                    //printf("%f\n",trackingState->Pk.m[0]);
                }*/
            }

            void UpdateInitialPose(ITMTrackingState *trackingState)
            {
                for (int i = 0; i < noTrackers; i++) trackers[i]->UpdateInitialPose(trackingState);
            }

            // Suppress the default copy constructor and assignment operator
            ITMCompositeTracker(const ITMCompositeTracker&);
            ITMCompositeTracker& operator=(const ITMCompositeTracker&);
        };
    }
}
