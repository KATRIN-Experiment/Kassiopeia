/*
 * KVTKViewerVisitorBuilder.cc
 *
 *  Created on:
 *      Author:
 */

#include "KFMVTKElectrostaticTreeViewerBuilder.hh"

#include "KElectricFastMultipoleFieldSolverBuilder.hh"

namespace katrin
{

template<> KFMVTKElectrostaticTreeViewerBuilder::~KComplexElement() {}

STATICINT sKFMVTKElectrostaticTreeViewerBuilderStructure =
    KFMVTKElectrostaticTreeViewerBuilder::Attribute<std::string>("file") +
    //KFMVTKElectrostaticTreeViewerBuilder::Attribute< bool >( "view" ) +
    KFMVTKElectrostaticTreeViewerBuilder::Attribute<bool>("save");

STATICINT sKElectricFastMultipoleFieldSolver =
    KElectricFastMultipoleFieldSolverBuilder::ComplexElement<KEMField::KFMVTKElectrostaticTreeViewerData>("viewer");

} /* namespace katrin */
