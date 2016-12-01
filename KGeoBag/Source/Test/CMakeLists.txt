set (TEST_LIBS
    KGeoBagCore
    KGeoBagShapes
    KGeoBagMesh
    ${Kommon_LIBRARIES}
    ${Kommon_Vtk_LIBRARIES}
)

# internal
kasper_internal_include_directories(
 	${CMAKE_CURRENT_SOURCE_DIR}/Basic/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Complex/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/Utility/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/PlanarShapes/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/FlattenedAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/RotatedAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/RotatedVolumes/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/ExtrudedAreas/Include
    ${CMAKE_CURRENT_SOURCE_DIR}/ExtrudedVolumes/Include
)

#if(KGeoBag_USE_VTK)
#list( APPEND TEST_LIBS
#      KGeoBagVisualization
#      )
#endif(KGeoBag_USE_VTK)

#add_executable (TestKGBeam
#${CMAKE_CURRENT_SOURCE_DIR}/TestKGBeam.cc)
#target_link_libraries (TestKGBeam ${TEST_LIBS})

#kasper_install_executables (
#TestKGBeam
#)

#add_executable (TestStructure
#${CMAKE_CURRENT_SOURCE_DIR}/TestStructure.cc)
#target_link_libraries (TestStructure ${TEST_LIBS})

#add_executable (TestCylinders
#${CMAKE_CURRENT_SOURCE_DIR}/TestCylinders.cc)
#target_link_libraries (TestCylinders ${TEST_LIBS})

#add_executable (TestEMCylinders
#${CMAKE_CURRENT_SOURCE_DIR}/TestEMCylinders.cc)
#target_link_libraries (TestEMCylinders ${TEST_LIBS})

#add_executable (TestInterpolation
#${CMAKE_CURRENT_SOURCE_DIR}/TestInterpolation.cc)
#target_link_libraries (TestInterpolation ${TEST_LIBS} KGeoBagKatrin)

#kasper_install_executables (
#    TestStructure
#    TestCylinders
#    TestEMCylinders
#    TestInterpolation
#)
