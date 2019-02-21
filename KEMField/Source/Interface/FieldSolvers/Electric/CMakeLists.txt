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

# link libraries
set (FIELDSOLVER_ELECTRIC_LINK_LIBRARIES
    KEMZHSolver
    KEMZHGenerator
    KFMElectrostatics
    KFMInterfaceFieldSolvers
    KEMHashGenerator
    KEMCore
    KEMFieldElectricBoundaryIntegrators
    KEMElectrostaticBoundaryIntegrals
    KEMFieldExceptions
)

if (${PROJECT_NAME}_USE_OPENCL)
    kasper_external_include_directories( ${OPENCL_INCLUDE_DIRS} )
    list (APPEND FIELDSOLVER_ELECTRIC_LINK_LIBRARIES
        KEMOpenCLPlugin
    )
endif (${PROJECT_NAME}_USE_OPENCL)

if (${PROJECT_NAME}_USE_MPI)
    list (APPEND FIELDSOLVER_ELECTRIC_LINK_LIBRARIES
        KEMCore #for MPIInterface.hh
    )
endif(${PROJECT_NAME}_USE_MPI)

if (${PROJECT_NAME}_USE_VTK)
    list (APPEND FIELDSOLVER_ELECTRIC_LINK_LIBRARIES
	KEMVTKPlugin
    )
endif (${PROJECT_NAME}_USE_VTK)

# library
kasper_internal_include_directories( ${FIELDSOLVER_ELECTRIC_HEADER_PATH} )
add_library( KEMElectricFieldSolvers SHARED ${FIELDSOLVER_ELECTRIC_SOURCE_FILENAMES} )
set_target_properties( KEMElectricFieldSolvers PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KEMElectricFieldSolvers
    ${FIELDSOLVER_ELECTRIC_LINK_LIBRARIES}
)

# install
kasper_install_headers( ${FIELDSOLVER_ELECTRIC_HEADER_FILENAMES} )
kasper_install_libraries( KEMElectricFieldSolvers )
