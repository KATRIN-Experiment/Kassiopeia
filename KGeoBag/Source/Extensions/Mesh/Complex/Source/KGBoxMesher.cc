#include "KGBoxMesher.hh"

#include "KGMeshRectangle.hh"

namespace KGeoBag
{
void KGBoxMesher::VisitBox(KGBox* box)
{
    KThreeVector p0;
    KThreeVector p1;
    KThreeVector p2;
    KThreeVector p3;

    for (unsigned int i = 0; i < 3; i++) {
        p0 = box->GetP0();
        unsigned int index1 = (i + 1) % 3;
        unsigned int index2 = (i + 2) % 3;

        KThreeVector n1(0., 0., 0.);
        KThreeVector n2(0., 0., 0.);

        n1[index1] = 1.;
        n2[index2] = 1.;

        double dist1 = box->GetP1()[index1] - box->GetP0()[index1];
        double dist2 = box->GetP1()[index2] - box->GetP0()[index2];

        std::vector<double> d1(box->GetMeshCount(index1), 0.);
        std::vector<double> d2(box->GetMeshCount(index2), 0.);

        DiscretizeInterval(dist1, box->GetMeshCount(index1), box->GetMeshPower(index1), d1);
        DiscretizeInterval(dist2, box->GetMeshCount(index2), box->GetMeshPower(index2), d2);

        for (double j : d1) {
            for (double k : d2) {
                p1 = p0 + j * n1;
                p2 = p0 + j * n1 + k * n2;
                p3 = p0 + k * n2;

                AddElement(new KGMeshRectangle(p0, p3, p2, p1));

                p0[i] = p1[i] = p2[i] = p3[i] = box->GetP1()[i];

                AddElement(new KGMeshRectangle(p0, p1, p2, p3));

                p0 = p3;
                p0[i] = box->GetP0()[i];
            }
            p0[index1] = p1[index1];
            p0[index2] = box->GetP0()[index2];
        }
    }
}
}  // namespace KGeoBag
