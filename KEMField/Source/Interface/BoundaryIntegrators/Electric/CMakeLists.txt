# CMakeLists for KEMField/Interface/BoundaryIntegrators/Electric
# Author: W. Gosda

# header files
set( ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_HEADER_BASENAMES
    KElectrostaticBoundaryIntegratorPolicy.hh

)

set( ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

foreach( BASENAME ${ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_HEADER_BASENAMES} )
    list( APPEND ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_HEADER_FILENAMES ${ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_SOURCE_BASENAMES
    KElectrostaticBoundaryIntegratorPolicy.cc
)

set( ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
foreach( BASENAME ${ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_SOURCE_BASENAMES} )
    list( APPEND ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_SOURCE_FILENAMES ${ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
add_library( KEMFieldElectricBoundaryIntegrators SHARED
    ${ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_SOURCE_FILENAMES} ${ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_HEADER_FILENAMES} )
#set_target_properties( KEMFieldElectricBoundaryIntegrators PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )

target_include_directories(KEMFieldElectricBoundaryIntegrators
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries( KEMFieldElectricBoundaryIntegrators
    PUBLIC
        KEMElectrostaticBoundaryIntegrals
        KEMFieldExceptions
)

if(KEMField_USE_OPENCL)
    target_link_libraries(KEMFieldElectricBoundaryIntegrators PUBLIC KEMOpenCLPlugin)
endif(KEMField_USE_OPENCL)

# install
kasper_install_headers( ${ELECTRICBOUNDARYINTEGRATOR_ELECTRIC_HEADER_FILENAMES} )
kasper_install_libraries( KEMFieldElectricBoundaryIntegrators )
