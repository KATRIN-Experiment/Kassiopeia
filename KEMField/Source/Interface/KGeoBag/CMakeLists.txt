# CMakeLists for KEMField/Interface/KGeoBag
# T.J. Corona, D. Hilk

set (KGEOBAGINTERFACE_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGElectrostaticBoundaryField.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGMagnetostaticBoundaryField.hh    
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGBEM.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGBEMConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGElectromagnet.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGElectromagnetConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGStaticElectromagnetField.hh
)

set (KGEOBAGINTERFACE_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGElectrostaticBoundaryField.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGMagnetostaticBoundaryField.cc    
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGBEM.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGBEMConverter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGElectromagnet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGElectromagnetConverter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGStaticElectromagnetField.cc
)

add_library (KGeoBagInterface SHARED
    ${KGEOBAGINTERFACE_SOURCEFILES} ${KGEOBAGINTERFACE_HEADERFILES})
target_include_directories(KGeoBagInterface
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KGeoBagInterface
    PUBLIC
        KEMSurfaces
        KEMFieldsElectric
        KEMFieldsMagnetic
        KEMElectromagnets
        KGeoBagAxialMesh
        KGeoBagDiscreteRotationalMesh
        KGeoBagMesh
)

target_compile_definitions( KGeoBagInterface PUBLIC KEMFIELD_USE_KGEOBAG )

kasper_install_headers (${KGEOBAGINTERFACE_HEADERFILES})
kasper_install_libraries (KGeoBagInterface)
