#pragma once

#include "../../ITMEKF.h"

namespace ITMLib
{
    namespace Engine
    {
        class ITMEKF_CUDA : public ITMEKF
        {
        public:
            struct accuCell;
            ITMEKF_CUDA();
            ~ITMEKF_CUDA(void);
            void covmat(ITMTrackingState *trackingState, const ITMView *view, bool &validpoint, float *covN);
        private:
            accuCell *accu_host1;
            accuCell *accu_device1;
        };
    }
}
