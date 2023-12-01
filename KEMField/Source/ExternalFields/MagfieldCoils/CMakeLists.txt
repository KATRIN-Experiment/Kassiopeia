# CMakeLists for KEMField/ExternalFields/MagfieldCoils
# Author: J. Behrens

set (MAGFIELDCOILS_HEADER_BASENAMES
    MagfieldCoils.h
)

set( MAGFIELDCOILS_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

foreach( BASENAME ${MAGFIELDCOILS_HEADER_BASENAMES} )
    list( APPEND MAGFIELDCOILS_HEADERFILES ${MAGFIELDCOILS_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

set (MAGFIELDCOILS_SOURCE_BASENAMES
    MagfieldCoils.cxx

#    CarlsonEllipticIntegrals.cc
#    GaussLegendreIntegration.cc
#    Magfield.cc
#    MagfieldCentral.cc
#    MagfieldCoil.cc
#    MagfieldElliptic.cc
#    MagfieldGroup.cc
#    MagfieldMagcharge.cc
#    MagfieldRemote.cc
#    MagfieldSPS.cc
#    MagsourceCentral.cc
#    MagsourceMagcharge.cc
#    MagsourceRemote.cc
)

set( MAGFIELDCOILS_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

foreach( BASENAME ${MAGFIELDCOILS_SOURCE_BASENAMES} )
    list( APPEND MAGFIELDCOILS_SOURCEFILES ${MAGFIELDCOILS_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

add_library (KEMMagfieldCoils SHARED
    ${MAGFIELDCOILS_SOURCEFILES} ${MAGFIELDCOILS_HEADERFILES})
target_include_directories(KEMMagfieldCoils
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KEMMagfieldCoils
    PUBLIC
#        KEMCore
#        KEMMath
#        KEMFieldSolverCore
)

kasper_install_headers (${MAGFIELDCOILS_HEADERFILES})
kasper_install_libraries (KEMMagfieldCoils)
