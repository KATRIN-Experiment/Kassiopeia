# CMakeLists for KEMField/Source/Bindings
# W. Gosda

set (KEMFIELD_BINDINGS_HEADERFILES
#    ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/include/KEMToolbox.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/include/KEMToolboxHelper.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/include/KEMToolboxBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/include/KEMBindingsMessage.hh
    #${CMAKE_CURRENT_SOURCE_DIR}/Utilities/include/KSmartPointerRelease.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/include/KBoundaryElementInfoDisplayBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/include/KElectricQuadrupoleFieldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/include/KElectrostaticConstantFieldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/include/KElectrostaticLinearFieldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/include/KElectrostaticBoundaryFieldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/include/KElectrostaticBoundaryFieldTimerBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/include/KVTKViewerVisitorBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/include/KInducedAzimuthalElectricFieldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/include/KRampedElectricFieldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/include/KRampedElectric2FieldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/include/KMagnetostaticConstantFieldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/include/KMagneticDipoleFieldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/include/KStaticElectromagnetFieldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/include/KMagneticSuperpositionFieldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/include/KRampedMagneticFieldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/KGeoBag/include/KGBEMBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/KGeoBag/include/KGElectromagnetBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KElectrostaticBoundaryIntegratorAttributeProcessor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KCachedChargeDensitySolverBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KExplicitSuperpositionCachedChargeDensitySolverBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KExplicitSuperpositionSolutionComponentBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KFastMultipoleMatrixGeneratorBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KGaussianEliminationChargeDensitySolverBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KGaussSeidelChargeDensitySolverBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KRobinHoodChargeDensitySolverBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KKrylovChargeDensitySolverBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KKrylovChargeDensitySolverOldBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KKrylovPreconditionerGeneratorBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KKrylovSolverConfigurationReader.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/include/KFMElectrostaticParametersBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Electric/include/KIntegratingElectrostaticFieldSolverBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Electric/include/KElectricZHFieldSolverBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Electric/include/KElectricFastMultipoleFieldSolverBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Magnetic/include/KIntegratingMagnetostaticFieldSolverBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Magnetic/include/KZonalHarmonicMagnetostaticFieldSolverBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/include/KElectrostaticPotentialmapBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/include/KMagnetostaticFieldmapBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/include/KMagnetostaticConstantFieldBuilder.hh
)

set (KEMFIELD_BINDINGS_SOURCEFILES
#    ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/src/KEMToolbox.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/src/KEMToolboxBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/src/KEMBindingsMessage.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/src/KBoundaryElementInfoDisplayBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/src/KElectricQuadrupoleFieldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/src/KElectrostaticConstantFieldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/src/KElectrostaticLinearFieldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/src/KElectrostaticBoundaryFieldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/src/KElectrostaticBoundaryFieldTimerBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/src/KVTKViewerVisitorBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/src/KInducedAzimuthalElectricFieldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/src/KRampedElectricFieldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/src/KRampedElectric2FieldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/src/KMagnetostaticConstantFieldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/src/KMagneticDipoleFieldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/src/KStaticElectromagnetFieldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/src/KMagneticSuperpositionFieldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/src/KRampedMagneticFieldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/KGeoBag/src/KGBEMBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/KGeoBag/src/KGElectromagnetBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/src/KCachedChargeDensitySolverBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/src/KExplicitSuperpositionCachedChargeDensitySolverBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/src/KExplicitSuperpositionSolutionComponentBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/src/KFastMultipoleMatrixGeneratorBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/src/KGaussianEliminationChargeDensitySolverBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/src/KGaussSeidelChargeDensitySolverBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/src/KRobinHoodChargeDensitySolverBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/src/KKrylovChargeDensitySolverBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/src/KKrylovChargeDensitySolverOldBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/src/KKrylovPreconditionerGeneratorBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ChargeDensitySolvers/Electric/src/KFMElectrostaticParametersBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Electric/src/KIntegratingElectrostaticFieldSolverBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Electric/src/KElectricZHFieldSolverBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Electric/src/KElectricFastMultipoleFieldSolverBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Magnetic/src/KIntegratingMagnetostaticFieldSolverBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Magnetic/src/KZonalHarmonicMagnetostaticFieldSolverBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Electric/src/KElectrostaticPotentialmapBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/src/KMagnetostaticFieldmapBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Fields/Magnetic/src/KMagnetostaticConstantFieldBuilder.cc
)

if (KEMField_USE_VTK)
    list (APPEND KEMFIELD_BINDINGS_HEADERFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Electric/include/KFMVTKElectrostaticTreeViewerBuilder.hh
    )
    list (APPEND KEMFIELD_BINDINGS_SOURCEFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Electric/src/KFMVTKElectrostaticTreeViewerBuilder.cc
    )
endif (KEMField_USE_VTK)

add_library (KEMFieldBindings SHARED
    ${KEMFIELD_BINDINGS_SOURCEFILES} ${KEMFIELD_BINDINGS_HEADERFILES}
)
add_library (KEMFieldBindingsHeaders INTERFACE)

# get header paths from collected header files
foreach(HEADER ${KEMFIELD_BINDINGS_HEADERFILES})
    get_filename_component(DIRNAME ${HEADER} DIRECTORY)
    target_include_directories(KEMFieldBindings PUBLIC $<BUILD_INTERFACE:${DIRNAME}>)
    target_include_directories(KEMFieldBindingsHeaders INTERFACE $<BUILD_INTERFACE:${DIRNAME}>)
endforeach(HEADER)
target_include_directories(KEMFieldBindings PUBLIC $<INSTALL_INTERFACE:include>)

target_link_libraries (KEMFieldBindings
    PUBLIC
        KommonCore
        KGeoBagInterface
        KEMCore
        KEMFieldElectricBoundaryIntegrators
        KEMFieldsElectric
        KEMFieldsMagnetic
        KEMElectricFieldSolvers
        KEMMagneticFieldSolvers
        KEMVTKPart2
    PRIVATE
        KGeoBagBindings
)

kasper_install_headers (${KEMFIELD_BINDINGS_HEADERFILES})
kasper_install_libraries (KEMFieldBindings KEMFieldBindingsHeaders)
