# headers
set(KOMMON_ROOT_HEADER_FILES
    File/KRootFile.h
    Initialization/KFormulaProcessor.hh
    Utility/KTRandom.h
    Utility/KROOTPainter.h
    Utility/KROOTWindow.h
    Utility/KROOTPad.h
    Utility/KROOTWindowBuilder.h
    Utility/KROOTPadBuilder.h
)

# source
set(KOMMON_ROOT_SOURCE_FILES
    File/KRootFile.cxx
    Initialization/KFormulaProcessor.cc
    Utility/KTRandom.cxx
    Utility/KROOTPainter.cxx
    Utility/KROOTWindow.cxx
    Utility/KROOTPad.cxx
    Utility/KROOTWindowBuilder.cxx
    Utility/KROOTPadBuilder.cxx
)

# propagate path variable to parent scope
kasper_append_paths( KOMMON_HEADER_FILES ${KOMMON_ROOT_HEADER_FILES} )
kasper_append_paths( KOMMON_SOURCE_FILES ${KOMMON_ROOT_SOURCE_FILES} )
