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

    void SetFile(const std::string& file)
    {
        fFile = file;
    }

    void SetPath(const std::string& path)
    {
        fPath = path;
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
    std::string fPath;
};

} /* namespace KEMField */

#endif /* KVTKVIEWERASBOUNDARYFIELDVISITOR_HH_ */
