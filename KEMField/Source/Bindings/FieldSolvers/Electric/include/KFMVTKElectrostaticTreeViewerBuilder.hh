/*
 * KFMVTKElectrostaticTreeViewerBuilder.hh
 *
 *  Created on:
 *      Author:
 */

#ifndef KFMVTKELECTROSTATICTREEVIEWERBUILDER_HH_
#define KFMVTKELECTROSTATICTREEVIEWERBUILDER_HH_

#include "KComplexElement.hh"
#include "KFMVTKElectrostaticTreeViewer.hh"

namespace KEMField
{

class KFMVTKElectrostaticTreeViewerData
{
  public:
    std::string fFileName;
    //bool fViewGeometry;
    bool fSaveGeometry;
};

}  // namespace KEMField

namespace katrin
{

typedef KComplexElement<KEMField::KFMVTKElectrostaticTreeViewerData> KFMVTKElectrostaticTreeViewerBuilder;

template<> inline bool KFMVTKElectrostaticTreeViewerBuilder::Begin()
{
    fObject = new KEMField::KFMVTKElectrostaticTreeViewerData;
    return true;
}

template<> inline bool KFMVTKElectrostaticTreeViewerBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject->fFileName);
        return true;
    }
    if (aContainer->GetName() == "save") {
        aContainer->CopyTo(fObject->fSaveGeometry);
        return true;
    }
    return false;
}

template<> inline bool KFMVTKElectrostaticTreeViewerBuilder::End()
{
    auto* tTree = dynamic_cast<KEMField::KFMElectrostaticTree*>(fParentElement);

    auto* tViewer = new KEMField::KFMVTKElectrostaticTreeViewer(*tTree);

    if (fObject->fSaveGeometry) {
        tViewer->GenerateGeometryFile(fObject->fFileName);
    }

    delete tViewer;
    return true;
}

} /* namespace katrin */

#endif /* KFMVTKELECTROSTATICTREEVIEWERBUILDER_HH_ */
