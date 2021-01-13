# CMakeLists for KEMField/BoundaryIntegrals/Electrostatic
# Author: T.J. Corona

kasper_internal_include_directories(Core/include)
kasper_internal_include_directories(Analytic/include)
kasper_internal_include_directories(Numeric/include)
kasper_internal_include_directories(Reference/include)
kasper_internal_include_directories(RWG/include)

set (ELECTROSTATICBOUNDARYINTEGRALS_HEADERFILES
    Core/include/KElectrostaticElementIntegrator.hh
    Core/include/KElectrostaticBoundaryIntegrator.hh
    Core/include/KElectrostaticBoundaryIntegratorFactory.hh
    
    Analytic/include/KElectrostaticAnalyticTriangleIntegrator.hh
    Analytic/include/KElectrostaticAnalyticRectangleIntegrator.hh
    Analytic/include/KElectrostaticAnalyticRingIntegrator.hh
    Analytic/include/KElectrostaticAnalyticLineSegmentIntegrator.hh
    Analytic/include/KElectrostaticAnalyticConicSectionIntegrator.hh
    
    Numeric/include/KElectrostaticCubatureTriangleIntegrator.hh
    Numeric/include/KElectrostaticCubatureRectangleIntegrator.hh
    Numeric/include/KElectrostaticQuadratureLineSegmentIntegrator.hh
    
    Reference/include/KElectrostaticBiQuadratureTriangleIntegrator.hh
    Reference/include/KElectrostaticBiQuadratureRectangleIntegrator.hh
    Reference/include/KElectrostatic256NodeQuadratureLineSegmentIntegrator.hh
    
    RWG/include/KElectrostaticRWGTriangleIntegrator.hh
    RWG/include/KElectrostaticRWGRectangleIntegrator.hh
)

set (ELECTROSTATICBOUNDARYINTEGRALS_SOURCEFILES
    Core/src/KElectrostaticBoundaryIntegrator.cc
    Core/src/KElectrostaticBoundaryIntegratorFactory.cc
    
    Analytic/src/KElectrostaticAnalyticTriangleIntegrator.cc
    Analytic/src/KElectrostaticAnalyticRectangleIntegrator.cc
    Analytic/src/KElectrostaticAnalyticRingIntegrator.cc
    Analytic/src/KElectrostaticAnalyticLineSegmentIntegrator.cc
    Analytic/src/KElectrostaticAnalyticConicSectionIntegrator.cc
    
    Numeric/src/KElectrostaticCubatureTriangleIntegrator.cc
    Numeric/src/KElectrostaticCubatureRectangleIntegrator.cc
    Numeric/src/KElectrostaticQuadratureLineSegmentIntegrator.cc

    Reference/src/KElectrostaticBiQuadratureTriangleIntegrator.cc
    Reference/src/KElectrostaticBiQuadratureRectangleIntegrator.cc  
    Reference/src/KElectrostatic256NodeQuadratureLineSegmentIntegrator.cc
    
    RWG/src/KElectrostaticRWGTriangleIntegrator.cc
    RWG/src/KElectrostaticRWGRectangleIntegrator.cc
)

##################################################

add_library (KEMElectrostaticBoundaryIntegrals SHARED ${ELECTROSTATICBOUNDARYINTEGRALS_SOURCEFILES} ${ELECTROSTATICBOUNDARYINTEGRALS_HEADERFILES})
target_link_libraries (KEMElectrostaticBoundaryIntegrals
    KEMMath
    KEMSurfaces
    KEMCore
    KEMFieldExceptions
    ${GSL_LIBRARIES})

kasper_install_headers (${ELECTROSTATICBOUNDARYINTEGRALS_HEADERFILES})
kasper_install_libraries (KEMElectrostaticBoundaryIntegrals)

