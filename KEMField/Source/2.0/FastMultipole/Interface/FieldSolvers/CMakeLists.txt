# CMakeLists for KEMField/FastMultipole/Subdivision
# Author: J. P. Barrett

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (INTERFACE_FIELDSOLVER_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticFastMultipoleFieldSolver.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticFastMultipoleMultipleTreeFieldSolver.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticTreeConstructor.hh
)

set (INTERFACE_FIELDSOLVER_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticFastMultipoleFieldSolver.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticFastMultipoleMultipleTreeFieldSolver.cc
)

##################################################

add_library (KFMInterfaceFieldSolvers SHARED ${INTERFACE_FIELDSOLVER_SOURCEFILES})
target_link_libraries (KFMInterfaceFieldSolvers KFMInterfaceExtraction KEMMath KEMSurfaces KFMCore KFMElectrostatics KEMIntegratingFieldSolver)

kasper_install_headers (${INTERFACE_FIELDSOLVER_HEADERFILES})
kasper_install_libraries (KFMInterfaceFieldSolvers)
