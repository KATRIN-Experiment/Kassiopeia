cmake_minimum_required(VERSION ${CMAKE_MINIMUM_VERSION})
project(Binning)

kasper_module_paths(Kommon)

# dependencies
find_package(Boost ${BOOST_MINIMUM_VERSION} REQUIRED)

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
add_library(Binning SHARED ${SOURCES})
target_include_directories(Binning
    PUBLIC $<BUILD_INTERFACE:${HEADER_PATH}> $<INSTALL_INTERFACE:include>
)
target_compile_features(Binning PRIVATE cxx_std_14)
target_link_libraries(Binning
    PRIVATE
        Kommon
        Boost::boost
)

kasper_install_libraries(Binning)

kasper_install_module()
