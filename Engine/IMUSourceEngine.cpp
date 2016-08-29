// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

/*#include "IMUSourceEngine.h"

#include "../Utils/FileUtils.h"

#include <stdio.h>

using namespace InfiniTAM::Engine;

IMUSourceEngine::IMUSourceEngine(const char *imuMask)
{
    strncpy(this->imuMask, imuMask, BUF_SIZE);

    currentFrameNo = 0;
    cachedFrameNo = -1;

    cached_imu = NULL;
}

void IMUSourceEngine::loadIMUIntoCache(void)
{
    char str[2048]; FILE *f; bool success = false;

    cached_imu = new ITMIMUMeasurement();

    sprintf(str, imuMask, currentFrameNo);
    f = fopen(str, "r");
    if (f)
    {
        size_t ret = fscanf(f, "%f %f %f %f %f %f %f %f %f",
            &cached_imu->R.m00, &cached_imu->R.m01, &cached_imu->R.m02,
            &cached_imu->R.m10, &cached_imu->R.m11, &cached_imu->R.m12,
            &cached_imu->R.m20, &cached_imu->R.m21, &cached_imu->R.m22);

        fclose(f);

        if (ret == 9) success = true;
    }

    if (!success) {
        delete cached_imu; cached_imu = NULL;
        printf("error reading file '%s'\n", str);
    }
}

bool IMUSourceEngine::hasMoreMeasurements(void)
{
    loadIMUIntoCache();

    return (cached_imu != NULL);
}

void IMUSourceEngine::getMeasurement(ITMIMUMeasurement *imu)
{
    bool bUsedCache = false;

    if (cached_imu != NULL)
    {
        imu->R = cached_imu->R;
        delete cached_imu;
        cached_imu = NULL;
        bUsedCache = true;
    }

    if (!bUsedCache) this->loadIMUIntoCache();

    ++currentFrameNo;
}*/
#include "IMUSourceEngine.h"
#include "../Utils/FileUtils.h"
//#include <stdio.h>
using namespace InfiniTAM::Engine;
IMUSourceEngine::IMUSourceEngine()
{
    currentFrameNo = 0;
    cachedFrameNo = -1;
    cached_imu = NULL;
    manager = LpmsSensorManagerFactory();
    lpms = manager->addSensor(DEVICE_LPMS_U, "A5022WD0");
}
void IMUSourceEngine::loadIMUIntoCache(void)
{
    bool success = false;
    cached_imu = new ITMIMUMeasurement();
    //sprintf(str, imuMask, currentFrameNo);
    //f = fopen(str, "r");
    if (lpms->getConnectionStatus() == SENSOR_CONNECTION_CONNECTED &&
            lpms->hasImuData())
    {
        d = lpms->getCurrentData();
        cached_imu->R.m00 = d.rotationM[0];
        cached_imu->R.m01 = d.rotationM[3];
        cached_imu->R.m02 = d.rotationM[6];
        cached_imu->R.m10 = d.rotationM[1];
        cached_imu->R.m11 = d.rotationM[4];
        cached_imu->R.m12 = d.rotationM[7];
        cached_imu->R.m20 = d.rotationM[2];
        cached_imu->R.m21 = d.rotationM[5];
        cached_imu->R.m22 = d.rotationM[8];
        cached_imu->dt = d.timeStamp;
        success = true;
    }
    if (!success) {
        delete cached_imu; cached_imu = NULL;
        printf("error reading\n");
    }
}
bool IMUSourceEngine::hasMoreMeasurements(void)
{
    loadIMUIntoCache();
    return (cached_imu != NULL);
}
void IMUSourceEngine::getMeasurement(ITMIMUMeasurement *imu)
{
    bool bUsedCache = false;
    if (cached_imu != NULL)
    {
        imu->R = cached_imu->R;
        imu->dt = cached_imu->dt;
        delete cached_imu;
        cached_imu = NULL;
        bUsedCache = true;
    }
    if (!bUsedCache) this->loadIMUIntoCache();
    ++currentFrameNo;
}
IMUSourceEngine::~IMUSourceEngine()
{
    // Removes the initialized sensor
    manager->removeSensor(lpms);
    // Deletes LpmsSensorManager object
    delete manager;
}
