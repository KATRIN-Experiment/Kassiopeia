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
#include(GNUInstallDirs)
install(DIRECTORY ${HEADER_PATH} DESTINATION ${INCLUDE_INSTALL_DIR})
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
	DESTINATION ${LIB_INSTALL_DIR}
)

# install CMake Targets and Config
install(EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_DIR}/${PROJECT_NAME}
)
install(FILES ${PROJECT_NAME}Config.cmake DESTINATION ${CMAKE_INSTALL_DIR}/${PROJECT_NAME})
