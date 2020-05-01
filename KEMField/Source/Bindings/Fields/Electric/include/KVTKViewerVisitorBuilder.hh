/*
 * KVTKViewerVisitorBuilder.hh
 *
 *  Created on: 30 Jul 2015
 *      Author: wolfgang
 */

#ifndef KVTKVIEWERVISITORBUILDER_HH_
#define KVTKVIEWERVISITORBUILDER_HH_

#include "KComplexElement.hh"
#include "KVTKViewerAsBoundaryFieldVisitor.hh"


namespace katrin
{

typedef KComplexElement<KEMField::KVTKViewerAsBoundaryFieldVisitor> KVTKViewerVisitorBuilder;

template<> inline bool KVTKViewerVisitorBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "file") {
        std::string name = "";
        aContainer->CopyTo(name);
        fObject->SetFile(name);


        return true;
    }
    if (aContainer->GetName() == "view") {
        bool choice = false;
        aContainer->CopyTo(choice);
        fObject->ViewGeometry(choice);
        return true;
    }
    if (aContainer->GetName() == "save") {
        bool choice = false;
        aContainer->CopyTo(choice);
        fObject->SaveGeometry(choice);
        return true;
    }
    if (aContainer->GetName() == "preprocessing") {
        bool choice = false;
        aContainer->CopyTo(choice);
        fObject->Preprocessing(choice);
        return true;
    }
    if (aContainer->GetName() == "postprocessing") {
        bool choice = false;
        aContainer->CopyTo(choice);
        fObject->Postprocessing(choice);
        return true;
    }
    return false;
}


} /* namespace katrin */

#endif /* KVTKVIEWERVISITORBUILDER_HH_ */
