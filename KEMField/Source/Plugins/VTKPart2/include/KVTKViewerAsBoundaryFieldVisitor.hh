/*
 * KVTKViewerAsBoundaryFieldVisitor.hh
 *
 *  Created on: 29 Jul 2015
 *      Author: wolfgang
 */

#ifndef KVTKVIEWERASBOUNDARYFIELDVISITOR_HH_
#define KVTKVIEWERASBOUNDARYFIELDVISITOR_HH_

#include "KElectrostaticBoundaryField.hh"

namespace KEMField
{

class KVTKViewerAsBoundaryFieldVisitor : public KElectrostaticBoundaryField::Visitor
{
  public:
    KVTKViewerAsBoundaryFieldVisitor();
    ~KVTKViewerAsBoundaryFieldVisitor() override;


    void ViewGeometry(bool choice)
    {
        fViewGeometry = choice;
    }

    void SaveGeometry(bool choice)
    {
        fSaveGeometry = choice;
    }

    void SetFile(string file)
    {
        fFile = file;
    }

    bool ViewGeometry() const
    {
        return fViewGeometry;
    }

    bool SaveGeometry() const
    {
        return fSaveGeometry;
    }

    void PreVisit(KElectrostaticBoundaryField&) override;
    void PostVisit(KElectrostaticBoundaryField&) override;

  private:
    bool fViewGeometry;
    bool fSaveGeometry;
    std::string fFile;
};

} /* namespace KEMField */

#endif /* KVTKVIEWERASBOUNDARYFIELDVISITOR_HH_ */
