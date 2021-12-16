#cmake_minimum_required(VERSION ${CMAKE_MINIMUM_VERSION})
#project(KommonMath)

#kasper_module_paths(Kommon)

# headers
set(KOMMON_MATH_HEADER_FILES
    include/Convolver.hpp
    include/Gaussian.hpp
    include/RangedConvolver.hpp
    include/SmearingKernel.hpp

    Geometry/include/KAxis.hh
    Geometry/include/KFrame.hh
    Geometry/include/KRotation.hh
    Geometry/include/KThreeMatrix.hh
    Geometry/include/KThreeVector.hh
    Geometry/include/KTransformation.hh
    Geometry/include/KTwoMatrix.hh
    Geometry/include/KTwoVector.hh
)

# sources
set(KOMMON_MATH_SOURCE_FILES
    src/Gaussian.cpp
    src/SmearingKernel.cpp

    Geometry/src/KAxis.cc
    Geometry/src/KFrame.cc
    Geometry/src/KRotation.cc
    Geometry/src/KThreeMatrix.cc
    Geometry/src/KThreeVector.cc
    Geometry/src/KTransformation.cc
    Geometry/src/KTwoMatrix.cc
    Geometry/src/KTwoVector.cc
)

# library
add_library(KommonMath SHARED
${KOMMON_MATH_SOURCE_FILES} ${KOMMON_MATH_HEADER_FILES})

# get header paths from collected header files
foreach(HEADER ${KOMMON_MATH_HEADER_FILES})
    get_filename_component(DIRNAME ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER} DIRECTORY)
    target_include_directories(KommonMath PUBLIC $<BUILD_INTERFACE:${DIRNAME}>)
endforeach(HEADER)
target_include_directories(KommonMath PUBLIC $<INSTALL_INTERFACE:include>)

#target_compile_features(KommonMath PRIVATE cxx_std_14)
target_link_libraries(KommonMath
    PRIVATE
        Boost::boost
        KommonBase
)

if( KASPER_USE_GSL )
    target_link_libraries(KommonMath
        PUBLIC
            KommonGsl
    )
endif( KASPER_USE_GSL )

kasper_install_libraries(KommonMath)
kasper_install_headers( ${KOMMON_MATH_HEADER_FILES} )

#kasper_install_module()
