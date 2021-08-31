# header files
set( MATH_HEADER_BASENAMES
    KSMathMessage.h
    KSMathArray.h
    KSMathSystem.h
    KSMathControl.h
    KSMathDifferentiator.h
    KSMathIntegrator.h
    KSMathRK8.h
    KSMathRK87.h
    KSMathRK86.h
    KSMathRK65.h
    KSMathRKF54.h
    KSMathRKDP54.h
    KSMathRKDP853.h
    KSMathSym4.h
    KSMathInterpolator.h
)
set( MATH_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${MATH_HEADER_BASENAMES} )
    list( APPEND MATH_HEADER_FILENAMES ${MATH_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( MATH_SOURCE_BASENAMES
    KSMathMessage.cxx
)
set( MATH_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${MATH_SOURCE_BASENAMES} )
    list( APPEND MATH_SOURCE_FILENAMES ${MATH_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
add_library( KassiopeiaMath SHARED
    ${MATH_SOURCE_FILENAMES} ${MATH_HEADER_FILENAMES} )
target_include_directories(KassiopeiaMath
    PUBLIC $<BUILD_INTERFACE:${MATH_HEADER_PATH}> $<INSTALL_INTERFACE:include> )
target_link_libraries( KassiopeiaMath
    PUBLIC
        Kommon
)

# install
kasper_install_headers( ${MATH_HEADER_FILENAMES} )
kasper_install_libraries( KassiopeiaMath )
