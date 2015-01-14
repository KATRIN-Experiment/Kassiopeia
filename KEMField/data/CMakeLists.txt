# CMakeLists for KEMField/Data
# T.J. Corona

install (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/input_files/sphericalCapacitorFiles
  DESTINATION ${@PROJECT_NAME@_CONFIG_INSTALL_DIR}
  PATTERN "*.gitignore" EXCLUDE
  )
