# - Find OpenNI
# This module defines
#  OpenNI_INCLUDE_DIR, where to find OpenNI include files
#  OpenNI_LIBRARIES, the libraries needed to use OpenNI
#  OpenNI_FOUND, If false, do not try to use OpenNI.
# also defined, but not for general use are
#  OpenNI_LIBRARY, where to find the OpenNI library.

set(OPEN_NI_ROOT "/home/hwj/OpenNI2" CACHE FILEPATH "Root directory of OpenNI2")

# Finally the library itself
find_library(OpenNI2_LIBRARY
NAMES OpenNI2
PATHS "/home/hwj/OpenNI2/Bin/x64-Release" "C:/Program Files (x86)/OpenNI/Lib" "C:/Program Files/OpenNI/Lib"
)

find_path(OpenNI2_INCLUDE_DIR OpenNI.h PATH "${OPEN_NI_ROOT}/Include")

find_library(OpenNI2_LIBRARY OpenNI2 PATH "${OPEN_NI_ROOT}/Bin/x64-Release/")

# handle the QUIETLY and REQUIRED arguments and set JPEG_FOUND to TRUE if
# all listed variables are TRUE
#include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
#include(${CMAKE_MODULE_PATH}/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(OpenNI2 DEFAULT_MSG OpenNI2_LIBRARY OpenNI2_INCLUDE_DIR)

if(OPENNI2_FOUND)
  set(OpenNI2_LIBRARIES ${OpenNI2_LIBRARY})
endif()

mark_as_advanced(OpenNI2_LIBRARY OpenNI2_INCLUDE_DIR)

