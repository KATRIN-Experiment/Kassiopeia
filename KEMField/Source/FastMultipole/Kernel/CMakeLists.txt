# CMakeLists for KEMField/FastMultipole/Kernel
# Author: J. P. Barrett

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (KERNEL_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMKernelExpansion.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMScalarMultipoleExpansion.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMResponseKernel_3DLaplaceL2L.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMResponseKernel_3DLaplaceM2M.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMResponseKernel_3DLaplaceM2L.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMMomentTransformer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMMomentTransformerTypes.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMKernelResponseArray.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMReducedKernelResponseArray.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMKernelResponseArrayTypes.hh
)

set (KERNEL_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMResponseKernel_3DLaplaceL2L.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMResponseKernel_3DLaplaceM2M.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMResponseKernel_3DLaplaceM2L.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMScalarMultipoleExpansion.cc
)

##################################################

add_library (KFMKernel SHARED ${KERNEL_SOURCEFILES} ${KERNEL_HEADERFILES})
target_link_libraries (KFMKernel KFMCore KFMMath)

kasper_install_headers (${KERNEL_HEADERFILES})
kasper_install_libraries (KFMKernel)
