# CMakeLists for KEMField/Source/2.0/XML

if (@PROJECT_NAME@_USE_KOMMON_BINDINGS)
  kasper_install_config_subdir(Complete ${CMAKE_CURRENT_SOURCE_DIR}/Complete/ElectricFields.xml)
  kasper_install_config_subdir(Complete ${CMAKE_CURRENT_SOURCE_DIR}/Complete/MagneticFields.xml)
  kasper_install_config_subdir(Complete ${CMAKE_CURRENT_SOURCE_DIR}/Complete/ElectricFieldExplicitSuperposition.xml)
endif( @PROJECT_NAME@_USE_KOMMON_BINDINGS)