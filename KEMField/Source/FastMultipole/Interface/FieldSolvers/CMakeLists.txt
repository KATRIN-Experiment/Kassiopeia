# CMakeLists for KEMField/FastMultipole/Subdivision
# Author: J. P. Barrett

set (INTERFACE_FIELDSOLVER_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticFastMultipoleFieldSolver.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticTreeConstructor.hh
)

set (INTERFACE_FIELDSOLVER_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticFastMultipoleFieldSolver.cc
)

add_library (KFMInterfaceFieldSolvers SHARED
    ${INTERFACE_FIELDSOLVER_SOURCEFILES} ${INTERFACE_FIELDSOLVER_HEADERFILES})
target_include_directories(KFMInterfaceFieldSolvers
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KFMInterfaceFieldSolvers
    PUBLIC
        KEMIntegratingSolver
        KEMElectrostaticBoundaryIntegrals
        KFMInterfaceExtraction
)

kasper_install_headers (${INTERFACE_FIELDSOLVER_HEADERFILES})
kasper_install_libraries (KFMInterfaceFieldSolvers)
