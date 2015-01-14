# CMakeLists for FastMultipole/Utility
# Author: J. P. Barrett

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (KFM_UTIL_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNamedScalarData.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMNamedScalarDataCollection.hh
)

set (KFM_UTIL_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMNamedScalarData.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMNamedScalarDataCollection.cc
)

add_library (KFMUtility SHARED ${KFM_UTIL_SOURCEFILES})
target_link_libraries (KFMUtility)

kasper_install_headers (${KFM_UTIL_HEADERFILES})
kasper_install_libraries (KFMUtility)
