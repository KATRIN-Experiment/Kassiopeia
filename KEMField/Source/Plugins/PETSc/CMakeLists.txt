# CMakeLists for KEMField/Plugins/PETSc
# T.J. Corona

cmake_dependent_option (${PROJECT_NAME}_USE_PETSc "Use PETSc for charge density computation" OFF ${PROJECT_NAME}_USE_MPI OFF)

if (NOT ${PROJECT_NAME}_USE_MPI)
  set (${PROJECT_NAME}_USE_PETSc OFF)
  set (${PROJECT_NAME}_USE_PETSc ${${PROJECT_NAME}_USE_PETSc} PARENT_SCOPE)
endif (NOT ${PROJECT_NAME}_USE_MPI)

if (${PROJECT_NAME}_USE_PETSc)

  find_package (PETSc REQUIRED)
  kasper_external_include_directories (${PETSC_INCLUDES})
  add_cflag (KEMFIELD_USE_PETSC)

  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

  set (PETSCPLUGIN_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KPETScInterface.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KPETScSolver.hh
    )

  set (PETSCPLUGIN_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KPETScInterface.cc
    )

  add_library (KEMPETScPlugin SHARED ${PETSCPLUGIN_SOURCEFILES})
  target_link_libraries (KEMPETScPlugin
    KEMCore
    KEMSurfaces
    ${PETSC_LIBRARY})

  kasper_install_headers (${PETSCPLUGIN_HEADERFILES})
  kasper_install_libraries (KEMPETScPlugin)

endif (${PROJECT_NAME}_USE_PETSc)
