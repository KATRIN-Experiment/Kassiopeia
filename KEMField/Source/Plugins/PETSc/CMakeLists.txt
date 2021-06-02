# CMakeLists for KEMField/Plugins/PETSc
# T.J. Corona

cmake_dependent_option (KEMField_USE_PETSc "Use PETSc for charge density computation" OFF KEMField_USE_MPI OFF)
if (KEMField_USE_PETSc)

    find_package (PETSc REQUIRED)

    set (PETSCPLUGIN_HEADERFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KPETScInterface.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KPETScSolver.hh
    )

    set (PETSCPLUGIN_SOURCEFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KPETScInterface.cc
    )

    add_library (KEMPETScPlugin SHARED
        ${PETSCPLUGIN_SOURCEFILES} ${PETSCPLUGIN_HEADERFILES})
    target_include_directories(KEMPETScPlugin
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
    target_include_directories(KEMPETScPlugin PUBLIC ${PETSC_INCLUDES})

    target_compile_definitions(KEMPETScPlugin PUBLIC KEMFIELD_USE_PETSC)
    target_link_libraries (KEMPETScPlugin
        PUBLIC
            KEMSurfaces
        PRIVATE
            ${PETSC_LIBRARY}
    )

    kasper_install_headers (${PETSCPLUGIN_HEADERFILES})
    kasper_install_libraries (KEMPETScPlugin)

endif (KEMField_USE_PETSc)
