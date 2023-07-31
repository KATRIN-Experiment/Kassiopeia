/**
 * @file KGPlyFile.cc
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2022-11-24
 */

#include "KGPlyFile.hh"
#include "KGCoreMessage.hh"

#include "KFile.h"
#include "happly.h"

#include <limits>

using namespace std;
using namespace KGeoBag;

using katrin::KThreeVector;

KGPlyFile::KGPlyFile() :
    fFile(),
    fPath(),
    fNDisc(0),
    fScaleFactor(1.),
    fSelectedIndices(),
    fTriangles(),
    fRectangles()
{}

KGPlyFile::~KGPlyFile() = default;

KGPlyFile* KGPlyFile::Clone() const
{
    auto clone = new KGPlyFile();

    clone->fFile = fFile;
    clone->fPath = fPath;
    clone->fNDisc = fNDisc;
    clone->fScaleFactor = fScaleFactor;
    clone->fSelectedIndices = fSelectedIndices;
    clone->fTriangles = fTriangles;
    clone->fRectangles = fRectangles;

    return clone;
}

void KGPlyFile::Initialize() const
{
    if (fInitialized)
        return;

    ReadPlyFile();

    fInitialized = true;
}

void KGPlyFile::SetFile(const string& aFile)
{
    fFile = aFile;
}

void KGPlyFile::SetPath(const string& aPath)
{
    fPath = aPath;
}

void KGPlyFile::SelectCell(size_t index)
{
    fSelectedIndices.insert({ index, index });
}

void KGPlyFile::SelectCellRange(size_t firstIndex, size_t lastIndex)
{
    fSelectedIndices.insert({ firstIndex, lastIndex });
}

bool KGPlyFile::ContainsPoint(const double* P) const
{
    KThreeVector point(P);
    for (auto & elem : fTriangles) {
        if (elem.ContainsPoint(point))
            return true;
    }
    for (auto & elem : fRectangles) {
        if (elem.ContainsPoint(point))
            return true;
    }
    return false;
}

double KGPlyFile::DistanceTo(const double* P, double* P_in, double* P_norm) const
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

    for (auto & elem : fRectangles) {
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

bool KGPlyFile::IsCellSelected(size_t index) const
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
KGTriangle GetTriangle(const std::vector<std::array<ValueT, 3>>& vertices, const std::vector<IndexT>& indices, double scale = 1.)
{
    KThreeVector p[3], n;
    assert(indices.size() == 3);

    for (auto icorner = 0; icorner < 3; icorner++) {
        p[icorner] = vertices[indices[icorner]];
    }

    auto tri = KGTriangle(p[0] * scale, p[1] * scale, p[2] * scale);
    // TODO: flip normals?

    return tri;
}

template<typename ValueT, typename IndexT>
KGRectangle GetRectangle(const std::vector<std::array<ValueT, 3>>& vertices, const std::vector<IndexT>& indices, double scale = 1.)
{
    KThreeVector p[4], n;
    assert(indices.size() == 4);

    for (auto icorner = 0; icorner < 4; icorner++) {
        p[icorner] = vertices[indices[icorner]];
    }

    KGRectangle rect = KGRectangle(p[0] * scale, p[1] * scale, p[2] * scale, p[3] * scale);
    // TODO: flip normals?

    return rect;
}

void KGPlyFile::ReadPlyFile() const
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

    coremsg_debug("reading elements from Ply file <" << tFile << ">" << eom);

    // Adapted from https://github.com/nmwsharp/happly
    try {
        happly::PLYData mesh(tFile.c_str());

        const std::vector<std::array<double, 3>> vertex_positions = mesh.getVertexPositions();
        const std::vector<std::vector<size_t>> face_indices = mesh.getFaceIndices<size_t>();

        for (size_t iface = 0; iface < face_indices.size(); ++iface) {
            if (! IsCellSelected(iface))
                continue;

            const size_t n_dim = face_indices[iface].size();
            switch (n_dim) {
                case 3:
                {
                    auto tri = GetTriangle(vertex_positions, face_indices[iface], fScaleFactor);
                    fTriangles.emplace_back(tri);
                    break;
                }
                case 4:
                {
                    auto rect = GetRectangle(vertex_positions, face_indices[iface], fScaleFactor);
                    fRectangles.emplace_back(rect);
                    break;
                }
                default:
                    coremsg(eError) << n_dim << "-sided face in ply file <" << tFile << "> is not supported (index " << iface << ")";
            }
        }


        coremsg(eNormal) << "ply file <" << tFile << "> contains <" << fTriangles.size()
                         << "> triangles and <" << fRectangles.size() << "> rectangles" << eom;
    }
    catch (std::exception &e) {
        coremsg(eError) << "could not read from file <" << tFile << ">: " << e.what() << eom;
        throw;
    }


}
