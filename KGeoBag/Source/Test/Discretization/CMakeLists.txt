set (TESTDISCRETIZATION_LIBS
    KGeoBagCore
    KGeoBagShapes
    KGeoBagMesh
    KGeoBagDeformation
    KGeoBagVisualization
#    KGeoBagKatrin
)

 add_executable (TestBeamDiscretization
 ${CMAKE_CURRENT_SOURCE_DIR}/TestBeamDiscretization.cc)
 target_link_libraries (TestBeamDiscretization ${TESTDISCRETIZATION_LIBS})

# add_executable (TestConicSectPortHousingDiscretization
# ${CMAKE_CURRENT_SOURCE_DIR}/TestConicSectPortHousingDiscretization.cc)
# target_link_libraries (TestConicSectPortHousingDiscretization ${TESTDISCRETIZATION_LIBS})

 add_executable (TestCylinderDiscretization
 ${CMAKE_CURRENT_SOURCE_DIR}/TestCylinderDiscretization.cc)
 target_link_libraries (TestCylinderDiscretization ${TESTDISCRETIZATION_LIBS})

# add_executable (TestExtrudedSurfaceDiscretization
# ${CMAKE_CURRENT_SOURCE_DIR}/TestExtrudedSurfaceDiscretization.cc)
# target_link_libraries (TestExtrudedSurfaceDiscretization ${TESTDISCRETIZATION_LIBS})

# add_executable (TestGateValveDiscretization
# ${CMAKE_CURRENT_SOURCE_DIR}/TestGateValveDiscretization.cc)
# target_link_libraries (TestGateValveDiscretization ${TESTDISCRETIZATION_LIBS})

# add_executable (TestPortHousingDiscretization
# ${CMAKE_CURRENT_SOURCE_DIR}/TestPortHousingDiscretization.cc)
# target_link_libraries (TestPortHousingDiscretization ${TESTDISCRETIZATION_LIBS})

# add_executable (TestRodDiscretization
# ${CMAKE_CURRENT_SOURCE_DIR}/TestRodDiscretization.cc)
# target_link_libraries (TestRodDiscretization ${TESTDISCRETIZATION_LIBS})

# add_executable (TestRotatedSurfaceDiscretization
# ${CMAKE_CURRENT_SOURCE_DIR}/TestRotatedSurfaceDiscretization.cc)
# target_link_libraries (TestRotatedSurfaceDiscretization ${TESTDISCRETIZATION_LIBS})

#add_executable (TestDiscretization
#${CMAKE_CURRENT_SOURCE_DIR}/TestDiscretization.cc)
#target_link_libraries (TestDiscretization ${TESTDISCRETIZATION_LIBS})

kasper_install_executables (
                    TestBeamDiscretization
#                    TestConicSectPortHousingDiscretization
                    TestCylinderDiscretization
#                    TestExtrudedSurfaceDiscretization
#                    TestGateValveDiscretization
#                    TestPortHousingDiscretization
#                    TestRotatedSurfaceDiscretization
#                    TestRodDiscretization
#                    TestDiscretization
)
