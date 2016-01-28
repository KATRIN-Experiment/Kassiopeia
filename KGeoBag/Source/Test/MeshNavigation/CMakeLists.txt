if(KGeoBag_USE_VTK)

set (TESTMESHNAVIGATION_LIBS
    ${Kommon_LIBRARIES}
    KGeoBagMath
    KGeoBagCore
    KGeoBagShapes
    KGeoBagAppearance
    KGeoBagMesh
    KGeoBagAxialMesh
    KGeoBagBindings
    KGeoBagRandom
    KGeoBagMetrics
    KGeoBagBasicVisualization
    KGeoBagVtkVisualization
)

# internal
kasper_internal_include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/Complex/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/PlanarShapes/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/FlattenedAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/RotatedAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/RotatedVolumes/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/ExtrudedAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/ExtrudedVolumes/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Shapes/ShellAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Extensions/Appearance/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Extensions/Deformation/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Extensions/Mesh/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Extensions/Mesh/Navigation/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Extensions/AxialMesh/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Extensions/DiscreteRotationalMesh/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Visualization/Include
)




add_executable (TestMeshNavigation
${CMAKE_CURRENT_SOURCE_DIR}/TestMeshNavigation.cc)
target_link_libraries (TestMeshNavigation ${TESTMESHNAVIGATION_LIBS})

kasper_install_executables (
    TestMeshNavigation
)

endif(KGeoBag_USE_VTK)
