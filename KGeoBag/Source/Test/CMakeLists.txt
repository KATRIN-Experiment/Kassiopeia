set (TEST_LIBS
    KGeoBagCore
    KGeoBagShapes
    KGeoBagMesh
    KGeoBagInterpolators
    ${Kommon_LIBRARIES}
)

if(KGeoBag_USE_VTK)
list( APPEND TEST_LIBS
      KGeoBagVisualization
      )
endif(KGeoBag_USE_VTK)

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
