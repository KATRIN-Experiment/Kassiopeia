cmake_minimum_required(VERSION 3.8)
project(Binning)

# dependencies
find_package(Boost 1.58 REQUIRED)

# headers
set(HEADER_PATH
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

# sources
set(SOURCE_BASENAMES
	ConstantBinning.cpp
	VariableBinning.cpp
)

foreach(BASENAME ${SOURCE_BASENAMES})
	list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/${BASENAME})
endforeach(BASENAME)

# library
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_include_directories(${PROJECT_NAME}
	PUBLIC $<BUILD_INTERFACE:${HEADER_PATH}> $<INSTALL_INTERFACE:include>
	PRIVATE ${Boost_INCLUDE_DIRS} 
)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
target_link_libraries(${PROJECT_NAME})

# install headers and lib
include(GNUInstallDirs)
install(DIRECTORY ${HEADER_PATH} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
	DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# install CMake Targets and Config
set(INSTALL_CMAKE_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
install(EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${INSTALL_CMAKE_CONFIGDIR}
)
install(FILES ${PROJECT_NAME}Config.cmake DESTINATION ${INSTALL_CMAKE_CONFIGDIR})
