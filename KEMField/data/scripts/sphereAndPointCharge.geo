scale = .5;

x = 0;
y = 0;
z = 0;
r = 1;

  p1 = newp; Point(p1) = {x,  y,  z,  scale} ;
  p2 = newp; Point(p2) = {x+r,y,  z,   scale} ;
  p3 = newp; Point(p3) = {x,  y+r,z,   scale} ;
  p4 = newp; Point(p4) = {x,  y,  z+r, scale} ;
  p5 = newp; Point(p5) = {x-r,y,  z,   scale} ;
  p6 = newp; Point(p6) = {x,  y-r,z,   scale} ;
  p7 = newp; Point(p7) = {x,  y,  z-r, scale} ;
 
  c1 = newreg; Circle(c1) = {p2,p1,p7};
  c2 = newreg; Circle(c2) = {p7,p1,p5};
  c3 = newreg; Circle(c3) = {p5,p1,p4};
  c4 = newreg; Circle(c4) = {p4,p1,p2};
  c5 = newreg; Circle(c5) = {p2,p1,p3};
  c6 = newreg; Circle(c6) = {p3,p1,p5};
  c7 = newreg; Circle(c7) = {p5,p1,p6};
  c8 = newreg; Circle(c8) = {p6,p1,p2};
  c9 = newreg; Circle(c9) = {p7,p1,p3};
  c10 = newreg; Circle(c10) = {p3,p1,p4};
  c11 = newreg; Circle(c11) = {p4,p1,p6};
  c12 = newreg; Circle(c12) = {p6,p1,p7};
 
 
  l1 = newreg; Line Loop(l1) = {c5,c10,c4};
  l2 = newreg; Line Loop(l2) = {c9,-c5,c1};
  l3 = newreg; Line Loop(l3) = {c12,-c8,-c1};
  l4 = newreg; Line Loop(l4) = {c8,-c4,c11};
  l5 = newreg; Line Loop(l5) = {-c10,c6,c3};
  l6 = newreg; Line Loop(l6) = {-c11,-c3,c7};
  l7 = newreg; Line Loop(l7) = {-c2,-c7,-c12};
  l8 = newreg; Line Loop(l8) = {-c6,-c9,c2};
 
 
  s1 = newreg; Ruled Surface(newreg) = {l1};
  s2 = newreg; Ruled Surface(newreg) = {l2};
  s3 = newreg; Ruled Surface(newreg) = {l3};
  s4 = newreg; Ruled Surface(newreg) = {l4};
  s5 = newreg; Ruled Surface(newreg) = {l5};
  s6 = newreg; Ruled Surface(newreg) = {l6};
  s7 = newreg; Ruled Surface(newreg) = {l7};
  s8 = newreg; Ruled Surface(newreg) = {l8};
 
x = 0;
y = 0;
z = 4;
r = .01;
scale2 = scale/100.;

  p8 = newp; Point(p8) = {x,  y,  z,  scale2} ;
  p9 = newp; Point(p9) = {x+r,y,  z,   scale2} ;
  p10 = newp; Point(p10) = {x,  y+r,z,   scale2} ;
  p11 = newp; Point(p11) = {x,  y,  z+r, scale2} ;
  p12 = newp; Point(p12) = {x-r,y,  z,   scale2} ;
  p13 = newp; Point(p13) = {x,  y-r,z,   scale2} ;
  p14 = newp; Point(p14) = {x,  y,  z-r, scale2} ;
 
  c13 = newreg; Circle(c13) = {p9,p8,p14};
  c14 = newreg; Circle(c14) = {p14,p8,p12};
  c15 = newreg; Circle(c15) = {p12,p8,p11};
  c16 = newreg; Circle(c16) = {p11,p8,p9};
  c17 = newreg; Circle(c17) = {p9,p8,p10};
  c18 = newreg; Circle(c18) = {p10,p8,p12};
  c19 = newreg; Circle(c19) = {p12,p8,p13};
  c20 = newreg; Circle(c20) = {p13,p8,p9};
  c21 = newreg; Circle(c21) = {p14,p8,p10};
  c22 = newreg; Circle(c22) = {p10,p8,p11};
  c23 = newreg; Circle(c23) = {p11,p8,p13};
  c24 = newreg; Circle(c24) = {p13,p8,p14};
 
 
  l9 = newreg; Line Loop(l9) = {c17,c22,c16};
  l10 = newreg; Line Loop(l10) = {c21,-c17,c13};
  l11 = newreg; Line Loop(l11) = {c24,-c20,-c13};
  l12 = newreg; Line Loop(l12) = {c20,-c16,c23};
  l13 = newreg; Line Loop(l13) = {-c22,c18,c15};
  l14 = newreg; Line Loop(l14) = {-c23,-c15,c19};
  l15 = newreg; Line Loop(l15) = {-c14,-c19,-c24};
  l16 = newreg; Line Loop(l16) = {-c18,-c21,c14};
 
 
  s9 = newreg; Ruled Surface(newreg) = {l9};
  s10 = newreg; Ruled Surface(newreg) = {l10};
  s11 = newreg; Ruled Surface(newreg) = {l11};
  s12 = newreg; Ruled Surface(newreg) = {l12};
  s13 = newreg; Ruled Surface(newreg) = {l13};
  s14 = newreg; Ruled Surface(newreg) = {l14};
  s15 = newreg; Ruled Surface(newreg) = {l15};
  s16 = newreg; Ruled Surface(newreg) = {l16};
 
