/**
 * @file KGStlFile.cc
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2021-07-02
 */

#include "KGStlFile.hh"
#include "KGCoreMessage.hh"

#include "KFile.h"
#include "stl_reader.h"

#include <limits>

using namespace std;
using namespace KGeoBag;

using katrin::KThreeVector;

KGStlFile::KGStlFile() :
    fFile(),
    fPath(),
    fNDisc(0),
    fScaleFactor(1.),
    fSelectedIndices(),
    fTriangles()
{}

KGStlFile::~KGStlFile() = default;

KGStlFile* KGStlFile::Clone() const
{
    auto clone = new KGStlFile();

    clone->fFile = fFile;
    clone->fPath = fPath;
    clone->fNDisc = fNDisc;
    clone->fScaleFactor = fScaleFactor;
    clone->fSelectedIndices = fSelectedIndices;
    clone->fTriangles = fTriangles;

    return clone;
}

void KGStlFile::Initialize() const
{
    if (fInitialized)
        return;

    ReadStlFile();

    fInitialized = true;
}

void KGStlFile::SetFile(const string& aFile)
{
    fFile = aFile;
}

void KGStlFile::SetPath(const string& aPath)
{
    fPath = aPath;
}

void KGStlFile::SelectCell(size_t index)
{
    fSelectedIndices.insert({ index, index });
}

void KGStlFile::SelectCellRange(size_t firstIndex, size_t lastIndex)
{
    fSelectedIndices.insert({ firstIndex, lastIndex });
}

bool KGStlFile::ContainsPoint(const double* P) const
{
    KThreeVector point(P);
    for (auto & elem : fTriangles) {
        if (elem.ContainsPoint(point))
            return true;
    }
    return false;
}

double KGStlFile::DistanceTo(const double* P, double* P_in, double* P_norm) const
{
    KThreeVector point(P);
    KThreeVector nearestPoint, nearestNormal;
    double nearestDistance = std::numeric_limits<double>::max();

    for (auto & elem : fTriangles) {
        double d = elem.DistanceTo(point, nearestPoint);
        if (d < nearestDistance) {
            nearestDistance = d;
            nearestNormal = elem.GetN3();

            if (P_in != nullptr) {
                for (unsigned i = 0; i < 3; ++i)
                    P_in[i] = nearestPoint[i];
            }

            if (P_norm != nullptr) {
                for (unsigned i = 0; i < 3; ++i)
                    P_norm[i] = nearestNormal[i];
            }
        }
    }

    return nearestDistance;
}

bool KGStlFile::IsCellSelected(size_t index) const
{
    if (fSelectedIndices.empty())
        return true;

    for (auto & range : fSelectedIndices) {
        if ((index >= range.first) && (index <= range.second))
            return true;
    }
    return false;
}

template<typename ValueT, typename IndexT>
KGTriangle GetTriangle(stl_reader::StlMesh<ValueT, IndexT>& mesh, size_t index, double scale = 1.)
{
    KThreeVector p[3], n;
    for (size_t icorner = 0; icorner < 3; ++icorner) {
        const double* c = mesh.tri_corner_coords(index, icorner);
        p[icorner] = c;
    }
    n = mesh.tri_normal(index);

    auto tri = KGTriangle(p[0] * scale, p[1] * scale, p[2] * scale);
    if (n.Dot(tri.GetN3()) < 0) {
        // normals are flipped, so change order of points
        tri = KGTriangle(p[2] * scale, p[1] * scale, p[0] * scale);
    }

    return tri;
}

void KGStlFile::ReadStlFile() const
{
    string tFile;

    if (!fFile.empty()) {
        if (fPath.empty()) {
            tFile = string(DATA_DEFAULT_DIR) + string("/") + fFile;
        }
        else {
            tFile = fPath + string("/") + fFile;
        }
    }
    else {
        tFile = string(DATA_DEFAULT_DIR) + string("/") + GetName() + string(".vtp");
    }

    coremsg_debug("reading elements from STL file <" << tFile << ">" << eom);

    // Adapted from https://github.com/sreiter/stl_reader
    try {
        stl_reader::StlMesh <double, unsigned int> mesh(tFile.c_str());

        const auto num_tris = mesh.num_tris();
        fTriangles.clear();
        fTriangles.reserve(num_tris);

        for(size_t itri = 0; itri < num_tris; ++itri) {
            if (! IsCellSelected(itri))
                continue;

            KGTriangle tri = GetTriangle(mesh, itri, fScaleFactor);
            fTriangles.emplace_back(tri);
        }

        /// TODO: avoid storing triangles twice if they're in a solid

        const auto num_solids = mesh.num_solids();
        fSolids.clear();
        fSolids.reserve(num_solids);

        for (size_t isol = 0; isol < num_solids; ++isol) {
            vector<KGTriangle> group;
            for (size_t itri = mesh.solid_tris_begin(isol); itri < mesh.solid_tris_end(isol); ++itri) {
                if (! IsCellSelected(itri))
                    continue;

                KGTriangle tri = GetTriangle(mesh, itri, fScaleFactor);
                group.emplace_back(tri);
            }
            if (! group.empty())
                fSolids.emplace_back(group);
        }

        coremsg(eNormal) << "STL file <" << tFile << "> contains <" << fTriangles.size()
                         << "> triangles and <" << fSolids.size() << "> solids" << eom;
    }
    catch (std::exception &e) {
        coremsg(eError) << "could not read from file <" << tFile << ">: " << e.what() << eom;
        throw;
    }


}
