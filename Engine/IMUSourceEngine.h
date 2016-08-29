// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../ITMLib/ITMLib.h"
#include <cstdio>
#include <thread>
#include "LpmsSensorI.h"
#include "LpmsSensorManagerI.h"

/*namespace InfiniTAM
{
    namespace Engine
    {
        class IMUSourceEngine
        {
        private:
            static const int BUF_SIZE = 2048;
            char imuMask[BUF_SIZE];

            ITMIMUMeasurement *cached_imu;

            void loadIMUIntoCache();
            int cachedFrameNo;
            int currentFrameNo;

        public:
            IMUSourceEngine(const char *imuMask);
            ~IMUSourceEngine() { }

            bool hasMoreMeasurements(void);
            void getMeasurement(ITMIMUMeasurement *imu);
        };
    }
}*/

namespace InfiniTAM
{
    namespace Engine
    {
        class IMUSourceEngine
        {
        private:
            ImuData d;
            // Gets a LpmsSensorManager instance
            LpmsSensorManagerI* manager;
            // Connects to LPMS-B sensor with address 00:11:22:33:44:55
            LpmsSensorI* lpms;
            ITMIMUMeasurement *cached_imu;
            void loadIMUIntoCache();
            int cachedFrameNo;
            int currentFrameNo;
        public:
            IMUSourceEngine();
            ~IMUSourceEngine();
            bool hasMoreMeasurements(void);
            void getMeasurement(ITMIMUMeasurement *imu);
        };
    }
}

