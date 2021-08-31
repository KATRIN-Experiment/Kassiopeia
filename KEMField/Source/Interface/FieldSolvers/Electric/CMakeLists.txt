# CMakeLists for KEMField/Interface/FieldSolvers/Electric
# Author: W. Gosda

# header files
set( FIELDSOLVER_ELECTRIC_HEADER_BASENAMES
    KElectricFastMultipoleFieldSolver.hh
    KElectricFieldSolver.hh
    KElectricZHFieldSolver.hh
    KIntegratingElectrostaticFieldSolver.hh
)

set( FIELDSOLVER_ELECTRIC_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

foreach( BASENAME ${FIELDSOLVER_ELECTRIC_HEADER_BASENAMES} )
    list( APPEND FIELDSOLVER_ELECTRIC_HEADER_FILENAMES ${FIELDSOLVER_ELECTRIC_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( FIELDSOLVER_ELECTRIC_SOURCE_BASENAMES
    KElectricFastMultipoleFieldSolver.cc
    KElectricZHFieldSolver.cc
    KIntegratingElectrostaticFieldSolver.cc
)

set( FIELDSOLVER_ELECTRIC_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
foreach( BASENAME ${FIELDSOLVER_ELECTRIC_SOURCE_BASENAMES} )
    list( APPEND FIELDSOLVER_ELECTRIC_SOURCE_FILENAMES ${FIELDSOLVER_ELECTRIC_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
add_library( KEMElectricFieldSolvers SHARED
    ${FIELDSOLVER_ELECTRIC_SOURCE_FILENAMES} ${FIELDSOLVER_ELECTRIC_HEADER_FILENAMES} )
target_include_directories(KEMElectricFieldSolvers
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries( KEMElectricFieldSolvers
    PUBLIC
        KEMCore
        KEMZHSolver
        KFMInterfaceFieldSolvers
        KEMFieldElectricBoundaryIntegrators
        KEMElectrostaticBoundaryIntegrals
)

if(KEMField_USE_OPENCL)
    target_link_libraries(KEMElectricFieldSolvers PUBLIC KEMOpenCLPlugin)
endif(KEMField_USE_OPENCL)

if(KEMField_USE_VTK)
    target_link_libraries(KEMElectricFieldSolvers PUBLIC KEMVTKPlugin)
endif(KEMField_USE_VTK)

#set_target_properties( KEMElectricFieldSolvers PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )

# install
kasper_install_headers( ${FIELDSOLVER_ELECTRIC_HEADER_FILENAMES} )
kasper_install_libraries( KEMElectricFieldSolvers )
