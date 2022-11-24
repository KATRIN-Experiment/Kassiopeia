# examples
kasper_install_config_subdir( Examples ${CMAKE_CURRENT_SOURCE_DIR}/Examples/ComplexShapes.xml )
kasper_install_config_subdir( Examples ${CMAKE_CURRENT_SOURCE_DIR}/Examples/SimpleShapes.xml )
kasper_install_config_subdir( Examples ${CMAKE_CURRENT_SOURCE_DIR}/Examples/ExternalShapes.xml )

kasper_install_config_subdir( Scripts ${CMAKE_CURRENT_SOURCE_DIR}/Scripts/PaintGeometry.xml )
kasper_install_config_subdir( Scripts ${CMAKE_CURRENT_SOURCE_DIR}/Scripts/PaintMesh.xml )
kasper_install_config_subdir( Scripts ${CMAKE_CURRENT_SOURCE_DIR}/Scripts/PaintAxialMesh.xml )
kasper_install_config_subdir( Scripts ${CMAKE_CURRENT_SOURCE_DIR}/Scripts/TestDistance.xml )
kasper_install_config_subdir( Scripts ${CMAKE_CURRENT_SOURCE_DIR}/Scripts/TestOutside.xml )
kasper_install_config_subdir( Scripts ${CMAKE_CURRENT_SOURCE_DIR}/Scripts/TestPoint.xml )
kasper_install_config_subdir( Scripts ${CMAKE_CURRENT_SOURCE_DIR}/Scripts/TestNormal.xml )

kasper_install_data( "${CMAKE_CURRENT_SOURCE_DIR}/DataFiles/Menger_sponge.stl" )
kasper_install_data( "${CMAKE_CURRENT_SOURCE_DIR}/DataFiles/Utah_teapot.stl" )
kasper_install_data( "${CMAKE_CURRENT_SOURCE_DIR}/DataFiles/Stanford_bunny.stl" )

kasper_install_data( "${CMAKE_CURRENT_SOURCE_DIR}/DataFiles/Menger_sponge.ply" )
kasper_install_data( "${CMAKE_CURRENT_SOURCE_DIR}/DataFiles/Utah_teapot.ply" )
kasper_install_data( "${CMAKE_CURRENT_SOURCE_DIR}/DataFiles/Stanford_bunny.ply" )
