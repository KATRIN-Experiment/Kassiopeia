# headers
set( KOMMON_ROOT_HEADER_FILENAMES
    File/KRootFile.h
    Initialization/KFormulaProcessor.hh
    Utility/KTRandom.h
    Utility/KROOTPainter.h
    Utility/KROOTWindow.h
    Utility/KROOTPad.h
)

# source
set( KOMMON_ROOT_SOURCE_FILENAMES
    File/KRootFile.cxx
    Initialization/KFormulaProcessor.cc
    Utility/KTRandom.cxx
    Utility/KROOTPainter.cxx
    Utility/KROOTWindow.cxx
    Utility/KROOTPad.cxx
)

# external
find_package( ROOT 5.24.0 REQUIRED )
kasper_external_include_directories(
	${ROOT_INCLUDE_DIR}
)

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/File
	${CMAKE_CURRENT_SOURCE_DIR}/Initialization
	${CMAKE_CURRENT_SOURCE_DIR}/Utility
)

# link
add_library( KommonRoot SHARED ${KOMMON_ROOT_SOURCE_FILENAMES} )
target_link_libraries( KommonRoot KommonCore ${ROOT_LIBRARIES} )

# install
kasper_install_libraries(KommonRoot)
kasper_install_headers( ${KOMMON_ROOT_HEADER_FILENAMES} )