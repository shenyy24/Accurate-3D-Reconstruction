include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})

IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
ELSE()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
ENDIF()

IF(OPENNI2_FOUND)
  include_directories(${OpenNI2_INCLUDE_DIR})
ELSE(OPENNI2_FOUND)
  add_definitions(-DCOMPILE_WITHOUT_OpenNI)
ENDIF(OPENNI2_FOUND)

IF(MSKINECTAPI_FOUND)
  include_directories(${MSKINECTAPI_INCLUDE_DIR})
ELSE(MSKINECTAPI_FOUND)
  add_definitions(-DCOMPILE_WITHOUT_Kinect2)
ENDIF(MSKINECTAPI_FOUND)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=gnu++11")
#include_directories("/usr/include")
endif()

add_library(Engine
ImageSourceEngine.cpp
ImageSourceEngine.h
IMUSourceEngine.cpp
IMUSourceEngine.h
LibUVCEngine.cpp
LibUVCEngine.h
Kinect2Engine.cpp
Kinect2Engine.h
OpenNIEngine.cpp
OpenNIEngine.h
UIEngine.cpp
UIEngine.h
)

target_link_libraries(Engine ${GLUT_LIBRARIES})
target_link_libraries(Engine ${OPENGL_LIBRARIES})
target_link_libraries(Engine ITMLib)
IF(WITH_CUDA)
  target_link_libraries(Engine ${CUDA_LIBRARY})
ENDIF()
IF(OPENNI2_FOUND)
  target_link_libraries(Engine ${OpenNI2_LIBRARY})
ENDIF(OPENNI2_FOUND)
    target_link_libraries(Engine LpSensor.so)
    target_link_libraries(Engine pthread)
    target_link_libraries(Engine dl)
    target_link_libraries(Engine rt)
