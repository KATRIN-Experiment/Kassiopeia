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
    KSMathRK54.h
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

# target
kasper_internal_include_directories( ${MATH_HEADER_PATH} )
add_library( KassiopeiaMath SHARED ${MATH_SOURCE_FILENAMES} )
target_link_libraries( KassiopeiaMath
    ${Kommon_LIBRARIES}
)

# install
kasper_install_headers( ${MATH_HEADER_FILENAMES} )
kasper_install_libraries( KassiopeiaMath )
         
