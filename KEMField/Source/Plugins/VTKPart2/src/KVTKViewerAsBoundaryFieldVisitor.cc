/*
 * KVTKViewerAsBoundaryFieldVisitor.cc
 *
 *  Created on: 29 Jul 2015
 *      Author: wolfgang
 */

#include "KVTKViewerAsBoundaryFieldVisitor.hh"

#ifdef KEMFIELD_USE_VTK
#include "KEMVTKViewer.hh"
#endif

#include "KFile.h"

#include "KEMCoreMessage.hh"
#include "KMPIEnvironment.hh"

namespace KEMField
{

KVTKViewerAsBoundaryFieldVisitor::KVTKViewerAsBoundaryFieldVisitor() :
    fViewGeometry(false),
    fSaveGeometry(false),
    fFile("ElectrostaticGeometry.vtp"),
    fPath()
{}

KVTKViewerAsBoundaryFieldVisitor::~KVTKViewerAsBoundaryFieldVisitor() = default;

void KVTKViewerAsBoundaryFieldVisitor::PreVisit(KElectrostaticBoundaryField& electrostaticField)
{
    PostVisit(electrostaticField);
}

#ifdef KEMFIELD_USE_VTK
void KVTKViewerAsBoundaryFieldVisitor::PostVisit(KElectrostaticBoundaryField& electrostaticField)
{
    MPI_SINGLE_PROCESS
    {
        if (fViewGeometry || fSaveGeometry) {
            KEMVTKViewer viewer(*(electrostaticField.GetContainer()));
            if (fViewGeometry)
                viewer.ViewGeometry();
            if (fSaveGeometry) {
                std::string tFileName;

                if (fFile.length() > 0) {
                    if (!fPath.empty()) {
                        tFileName = std::string(fPath) + std::string("/") + fFile;
                    }
                    else {
                        tFileName = std::string(SCRATCH_DEFAULT_DIR) + std::string("/") + fFile;
                    }
                }

                kem_cout << "Saving electrode geometry to <" << tFileName << ">" << eom;
                viewer.GenerateGeometryFile(tFileName);
            }
        }
    }
}
#else
void KVTKViewerAsBoundaryFieldVisitor::PostVisit(KElectrostaticBoundaryField& /*electrostaticField*/)
{
    return;
}
#endif

} /* namespace KEMField */
