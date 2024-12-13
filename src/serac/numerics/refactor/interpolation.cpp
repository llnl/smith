#include <iostream>

namespace femto {

void GaussLegendreNodes(int n, double * output) {

  // clang-format off
  switch (n) {
    case 1:
      output[0] = 0.5;
      break;
    case 2:
      output[0] = 0.2113248654051871;
      output[1] = 0.7886751345948129;
      break;
    case 3:
      output[0] = 0.1127016653792583;
      output[1] = 0.500000000000000;
      output[2] = 0.887298334620742;
      break;
    case 4:
      output[0] = 0.0694318442029737;
      output[1] = 0.330009478207572 ;
      output[2] = 0.669990521792428 ;
      output[3] = 0.930568155797026 ;
      break;
    case 5:
      output[0] = 0.04691007703066800;
      output[1] = 0.2307653449471585 ;
      output[2] = 0.5000000000000000 ;
      output[3] = 0.7692346550528415 ;
      output[4] = 0.9530899229693320 ;
      break;
    case 6:
      output[0] = 0.03376524289842399;
      output[1] = 0.1693953067668677 ;
      output[2] = 0.3806904069584015 ;
      output[3] = 0.6193095930415985 ;
      output[4] = 0.8306046932331323 ;
      output[5] = 0.9662347571015760 ;
      break;
  }
  // clang-format on

}

void GaussLegendreInterpolation(double x, int n, double * output) {

  if (n == 1) { 
    output[0] = 1; 
  }

  if (n == 2) { 
    output[0] = 1.3660254037844386467637231708 - 1.732050807568877293527446342*x;
    output[1] = -0.3660254037844386467637231708 + 1.7320508075688772935274463415*x;
  }

  if (n == 3) { 
    output[0] = 1.4788305577012361475298776 + x*(-4.624327782069138961726422 + 3.3333333333333333333333333*x);
    output[1] = -0.666666666666666666666666667 + (6.66666666666666666666666667 - 6.666666666666666666666666667*x)*x;
    output[2] = 0.1878361089654305191367891 + x*(-2.0423388845975277049402449 + 3.3333333333333333333333333*x);
  } 

  if (n == 4) { 
    output[0] = 1.526788125457266786984328 + x*(-8.54602360787219876597302 + (14.32585835417188815296662 - 7.42054006803894610520064*x)*x);
    output[1] = -0.8136324494869272605619 + x*(13.8071669256895770661587 + x*(-31.3882223634460602120582 + 18.7954494075550608112617*x));
    output[2] = 0.400761520311650404800281777 + x*(-7.41707042146263907582738061 + (24.9981258592191222217269164 - 18.79544940755506081126171563*x)*x);
    output[3] = -0.11391719628198993122271197 + x*(2.1559271036452607756417044 + x*(-7.935761849944950162635307 + 7.420540068038946105200642*x));
  } 

  if (n == 5) { 
    output[0] = 1.551408049094313012813028 + x*(-13.47028450119487106120462 + x*(38.6444990553441957009803 + x*(-44.9889850558789977671881 + 18.33972111443117301508323*x)));
    output[1] = -0.8931583920000717373262 + x*(22.924333555723729737768 + x*(-88.22281082816288605026 + (117.8634151266470135556 - 51.939721114431173015083*x)*x));
    output[2] = 0.5333333333333333333333 + x*(-14.933333333333333333333 + x*(82.13333333333333333333 + x*(-134.4 + 67.2*x)));
    output[3] = -0.26794165222338750930410993 + x*(7.6899271783856937562943889 + x*(-46.270892134808883473969833 + (89.895469331077678504736602 - 51.9397211144311730150832251*x)*x));
    output[4] = 0.07635866179581290048392539 + x*(-2.210642899581219099524808 + x*(13.71587057429424048991553 + x*(-28.36989940184569429314485 + 18.33972111443117301508323*x)));
  } 

  if (n == 6) { 
    output[0] = 1.565673200151071933093717 + x*(-19.38889969575614186464859 + x*(83.3561716652066047719407 + x*(-161.6334485633571811708389 + (144.8933610434784341266087 - 48.8475703740520537090491*x)*x)));
    output[1] = -0.94046284317634892902 + x*(33.94755689005745838881 + x*(-194.5900409203250530156 + x*(431.2442105751191477314 + x*(-416.6718961318097255735 + 147.20243244417318935282*x))));
    output[2] = 0.616930055430488708617 + x*(-24.29050650593736015819 + x*(195.3041651669510511719 + x*(-523.416260790836176187 + (568.4164873451100029584 - 217.0100429728326202484*x)*x)));
    output[3] = -0.37922770211461375461734918 + x*(15.315224028266875783526055 + x*(-134.54612286322366212220967 + x*(419.85074113872236683695564 + x*(-516.63372751905309828340431 + 217.010042972832620248366597*x))));
    output[4] = 0.1918000140386679548202 + x*(-7.82468446839184002159 + x*(71.1355384559059302654 + x*(-236.5809504896121389654 + (319.3402660890562211905 - 147.2024324441731893528*x)*x)));
    output[5] = -0.05471272432926591289350948 + x*(2.241309751761007872094777 + x*(-20.65971150451487107141517 + x*(70.5357081299639817548955 + x*(-99.344490826781834418637 + 48.84757037405205370904914*x))));
  } 

}

void GaussLegendreInterpolationDerivative01(double x, int n, double * output) {

  if (n == 1) { 
    output[0] = 0.0;
  }

  if (n == 2) { 
    output[0] = -1.7320508075688772935274463415;
    output[1] = 1.7320508075688772935274463415;
  }

  if (n == 3) { 
    output[0] = -4.6243277820691389617264218 + 6.6666666666666666666666667*x;
    output[1] = 6.66666666666666666666666667 - 13.3333333333333333333333333*x;
    output[2] = -2.04233888459752770494024487 + 6.6666666666666666666666667*x;
  }

  if (n == 4) { 
    output[0] = -8.546023607872198765973 + (28.6517167083437763059332 - 22.2616202041168383156019*x)*x;
    output[1] = 13.80716692568958 + x*(-62.7764447268921 + 56.3863482226652*x);
    output[2] = -7.417070421462639075827381 + (49.99625171843824444345383 - 56.38634822266518243378515*x)*x;
    output[3] = 2.155927103645260775641704 + x*(-15.87152369988990032527061 + 22.26162020411683831560193*x);
  }

  if (n == 5) { 
    output[0] = -13.4702845011948710612046 + x*(77.288998110688391401961 + x*(-134.966955167636993301564 + 73.358884457724692060333*x));
    output[1] = 22.92433355572373 + x*(-176.4456216563258 + (353.590245379941 - 207.7588844577247*x)*x);
    output[2] = -14.93333333333333 + x*(164.2666666666667 + x*(-403.2 + 268.8*x));
    output[3] = 7.689927178385693756294389 + x*(-92.54178426961776694793967 + (269.6864079932330355142098 - 207.7588844577246920603329*x)*x);
    output[4] = -2.21064289958121909952481 + x*(27.4317411485884809798311 + x*(-85.1096982055370828794345 + 73.3588844577246920603329*x));
  }

  if (n == 6) { 
    output[0] = -19.3888996957561418646486 + x*(166.712343330413209543881 + x*(-484.90034569007154351252 + (579.57344417391373650643 - 244.23785187026026854525*x)*x));
    output[1] = 33.94755689005746 + x*(-389.18008184065 + x*(1293.732631725357 + x*(-1666.687584527239 + 736.012162220866*x)));
    output[2] = -24.29050650593736 + x*(390.6083303339021 + x*(-1570.248782372509 + (2273.66594938044 - 1085.050214864163*x)*x));
    output[3] = 15.31522402826687578352605 + x*(-269.0922457264473242444193 + x*(1259.552223416167100510867 + x*(-2066.534910076212393133617 + 1085.050214864163101241833*x)));
    output[4] = -7.82468446839184 + x*(142.2710769118119 + x*(-709.742851468836 + (1277.361064356225 - 736.012162220866*x)*x));
    output[5] = 2.24130975176100787209478 + x*(-41.3194230090297421428303 + x*(211.607124389891945264686 + x*(-397.377963307127337674548 + 244.237851870260268545246*x)));
  }

}

void GaussLobattoNodes(int n, double * output) {
  if (n == 1) {
    output[0] = 0.5;
    return;
  }
  if (n == 2) {
    output[0] = 0.0;
    output[1] = 1.0;
    return;
  }
  if (n == 3) {
    output[0] = 0.0;
    output[1] = 0.5;
    output[2] = 1.0;
    return;
  }
  if (n == 4) {
    output[0] = 0.0;
    output[1] = 0.2763932022500210;
    output[2] = 0.7236067977499790;
    output[3] = 1.0;
    return;
  }
}

void GaussLobattoInterpolation(double x, int n, double * output) {
  if (n == 1) {
    output[0] = 1.0;
    return;
  }
  if (n == 2) {
    output[0] = 1.0 - x;
    output[1] = x;
    return;
  }
  if (n == 3) {
    output[0] = (-1.0 + x) * (-1.0 + 2.0 * x);
    output[1] = -4.0 * (-1.0 + x) * x;
    output[2] = x * (-1.0 + 2.0 * x);
    return;
  }
  if (n == 4) {
    constexpr double sqrt5 = 2.23606797749978981;
    output[0] = -(-1.0 + x) * (1.0 + 5.0 * (-1.0 + x) * x);
    output[1] = -0.5 * sqrt5 * (5.0 + sqrt5 - 10.0 * x) * (-1.0 + x) * x;
    output[2] = -0.5 * sqrt5 * (-1.0 + x) * x * (-5.0 + sqrt5 + 10.0 * x);
    output[3] = x * (1.0 + 5.0 * (-1.0 + x) * x);
    return;
  }

  std::cout << "error: invalid polynomial order in GaussLobattoInterpolation" << std::endl;
}

void GaussLobattoInterpolationDerivative(double x, int n, double * output) {
  if (n == 1) {
    output[0] = 0.0;
    return;
  }
  if (n == 2) {
    output[0] = -1.0;
    output[1] = 1.0;
    return;
  }
  if (n == 3) {
    output[0] = -3.0 + 4.0 * x;
    output[1] =  4.0 - 8.0 * x;
    output[2] = -1.0 + 4.0 * x;
    return;
  }
  if (n == 4) {
    constexpr double sqrt5 = 2.23606797749978981;
    output[0] = -6.0 + 5.0 * (4.0 - 3.0 * x) * x;
    output[1] =  2.5 * (1.0 + sqrt5 + 2.0 * x * (-1.0 - 3.0 * sqrt5 + 3.0 * sqrt5 * x));
    output[2] = -2.5 * (-1.0 + sqrt5 + 2.0 * x * (1.0 - 3.0 * sqrt5 + 3.0 * sqrt5 * x));
    output[3] =  1.0 + 5.0 * x * (-2.0 + 3.0 * x);
    return;
  }

  std::cout << "error: invalid polynomial order in GaussLobattoInterpolationDerivative" << std::endl;
}

#if 1
constexpr double GaussLobattoInterpolation(double x, uint32_t n, uint32_t i) {
  if (n == 1) { return 1.0; }
  if (n == 2) { return (i == 0) ? 1.0 - x : x; }
  if (n == 3) {
    if (i == 0) return (-1.0 + x) * (-1.0 + 2.0 * x);
    if (i == 1) return -4.0 * (-1.0 + x) * x;
    if (i == 2) return x * (-1.0 + 2.0 * x);
  }
  if (n == 4) {
    constexpr double sqrt5 = 2.23606797749978981;
    if (i == 0) return -(-1.0 + x) * (1.0 + 5.0 * (-1.0 + x) * x);
    if (i == 1) return -0.5 * sqrt5 * (5.0 + sqrt5 - 10.0 * x) * (-1.0 + x) * x;
    if (i == 2) return -0.5 * sqrt5 * (-1.0 + x) * x * (-5.0 + sqrt5 + 10.0 * x);
    if (i == 3) return x * (1.0 + 5.0 * (-1.0 + x) * x);
  }
  return -1.0;
}

constexpr double GaussLobattoInterpolationDerivative(double x, uint32_t n, uint32_t i) {
  if (n == 1) { return 0.0; }
  if (n == 2) { return (i == 0) ? -1.0 : 1.0; }
  if (n == 3) {
    if (i == 0) return  -3.0 + 4.0 * x;
    if (i == 1) return  4.0 - 8.0 * x;
    if (i == 2) return  -1.0 + 4.0 * x;
  }
  if (n == 4) {
    constexpr double sqrt5 = 2.23606797749978981;
    if (i == 0) return -6.0 + 5.0 * (4.0 - 3.0 * x) * x;
    if (i == 1) return  2.5 * (1.0 + sqrt5 + 2.0 * x * (-1.0 - 3.0 * sqrt5 + 3.0 * sqrt5 * x));
    if (i == 2) return -2.5 * (-1.0 + sqrt5 + 2.0 * x * (1.0 - 3.0 * sqrt5 + 3.0 * sqrt5 * x));
    if (i == 3) return  1.0 + 5.0 * x * (-2.0 + 3.0 * x);
  }
  return -1.0;
}
#endif

void GaussLobattoInterpolationTriangle(const double * xi, int p, double * output) {
  if (p == 0) {
    output[0] = 1.0;
    return;
  }
  if (p == 1) {
    output[0] = 1.0 - xi[0] - xi[1];
    output[1] = xi[0];
    output[2] = xi[1];
    return;
  }
  if (p == 2) {
    output[0] = (-1+xi[0]+xi[1])*(-1+2*xi[0]+2*xi[1]);
    output[1] = -4*xi[0]*(-1+xi[0]+xi[1]);
    output[2] = xi[0]*(-1+2*xi[0]);
    output[3] = -4*xi[1]*(-1+xi[0]+xi[1]);
    output[4] = 4*xi[0]*xi[1];
    output[5] = xi[1]*(-1+2*xi[1]);
    return;
  }
  if (p == 3) {
    double sqrt5 = 2.23606797749978981;
    output[0] = -((-1+xi[0]+xi[1])*(1+5*xi[0]*xi[0]+5*(-1+xi[1])*xi[1]+xi[0]*(-5+11*xi[1])));
    output[1] = (5*xi[0]*(-1+xi[0]+xi[1])*(-1-sqrt5+2*sqrt5*xi[0]+(3+sqrt5)*xi[1]))/2.0;
    output[2] = (-5*xi[0]*(-1+xi[0]+xi[1])*(1-sqrt5+2*sqrt5*xi[0]+(-3+sqrt5)*xi[1]))/2.0;
    output[3] = xi[0]*(1+5*xi[0]*xi[0]+xi[1]-xi[1]*xi[1]-xi[0]*(5+xi[1]));
    output[4] = (5*xi[1]*(-1+xi[0]+xi[1])*(-1-sqrt5+(3+sqrt5)*xi[0]+2*sqrt5*xi[1]))/2.0;
    output[5] = -27*xi[0]*xi[1]*(-1+xi[0]+xi[1]);
    output[6] = (5*xi[0]*xi[1]*(-2+(3+sqrt5)*xi[0]-(-3+sqrt5)*xi[1]))/2.;
    output[7] = (5*xi[1]*(-1+xi[0]+xi[1])*(5-3*sqrt5+2*(-5+2*sqrt5)*xi[0]+5*(-1+sqrt5)*xi[1]))/(-5+sqrt5);
    output[8] = (-5*xi[0]*xi[1]*(2+(-3+sqrt5)*xi[0]-(3+sqrt5)*xi[1]))/2.;
    output[9] = xi[1]*(1+xi[0]-xi[0]*xi[0]-xi[0]*xi[1]+5*(-1+xi[1])*xi[1]);
    return;
  }
}

void GaussLobattoInterpolationDerivativeTriangle(const double * xi, int p, double * output) {
  if (p == 0) {
    output[0] = 0.0;
    output[1] = 0.0;
    return;
  }
  if (p == 1) {
    output[0] = -1;
    output[1] = -1;
    output[2] = 1;
    output[3] = 0;
    output[4] = 0;
    output[5] = 1;
    return;
  }
  if (p == 2) {
    output[ 0] = -3+4*xi[0]+4*xi[1];
    output[ 1] = -3+4*xi[0]+4*xi[1];
    output[ 2] = -4*(-1+2*xi[0]+xi[1]);
    output[ 3] = -4*xi[0];
    output[ 4] = -1+4*xi[0];
    output[ 5] = 0;
    output[ 6] = -4*xi[1];
    output[ 7] = -4*(-1+xi[0]+2*xi[1]);
    output[ 8] = 4*xi[1];
    output[ 9] = 4*xi[0];
    output[10] = 0;
    output[11] = -1+4*xi[1];
    return;
  }
  if (p == 3) {
    double sqrt5 = 2.23606797749978981;
    output[ 0] = -6-15*xi[0]*xi[0]+4*xi[0]*(5-8*xi[1])+(21-16*xi[1])*xi[1];
    output[ 1] = -6-16*xi[0]*xi[0]+xi[0]*(21-32*xi[1])+5*(4-3*xi[1])*xi[1];
    output[ 2] = (5*(6*sqrt5*xi[0]*xi[0]+xi[0]*(-2-6*sqrt5+6*(1+sqrt5)*xi[1])+(-1+xi[1])*(-1-sqrt5+(3+sqrt5)*xi[1])))/2.;
    output[ 3] = (5*xi[0]*(-2*(2+sqrt5)+3*(1+sqrt5)*xi[0]+2*(3+sqrt5)*xi[1]))/2.;
    output[ 4] = (-5*(6*sqrt5*xi[0]*xi[0]+(-1+xi[1])*(1-sqrt5+(-3+sqrt5)*xi[1])+xi[0]*(2-6*sqrt5+6*(-1+sqrt5)*xi[1])))/2.;
    output[ 5] = (-5*xi[0]*(4-2*sqrt5+3*(-1+sqrt5)*xi[0]+2*(-3+sqrt5)*xi[1]))/2.;
    output[ 6] = 1+15*xi[0]*xi[0]+xi[1]-xi[1]*xi[1]-2*xi[0]*(5+xi[1]);
    output[ 7] = -(xi[0]*(-1+xi[0]+2*xi[1]));
    output[ 8] = (5*xi[1]*(-2*(2+sqrt5)+2*(3+sqrt5)*xi[0]+3*(1+sqrt5)*xi[1]))/2.;
    output[ 9] = (5*(1+sqrt5-2*(2+sqrt5)*xi[0]+(3+sqrt5)*xi[0]*xi[0]+6*(1+sqrt5)*xi[0]*xi[1]+2*xi[1]*(-1-3*sqrt5+3*sqrt5*xi[1])))/2.;
    output[10] = -27*xi[1]*(-1+2*xi[0]+xi[1]);
    output[11] = -27*xi[0]*(-1+xi[0]+2*xi[1]);
    output[12] = (-5*xi[1]*(2-2*(3+sqrt5)*xi[0]+(-3+sqrt5)*xi[1]))/2.;
    output[13] = (5*xi[0]*(-2+(3+sqrt5)*xi[0]-2*(-3+sqrt5)*xi[1]))/2.;
    output[14] = (-5*xi[1]*(4-2*sqrt5+2*(-3+sqrt5)*xi[0]+3*(-1+sqrt5)*xi[1]))/2.;
    output[15] = (-5*(-1+sqrt5+(-3+sqrt5)*xi[0]*xi[0]+2*xi[1]*(1-3*sqrt5+3*sqrt5*xi[1])+xi[0]*(4-2*sqrt5+6*(-1+sqrt5)*xi[1])))/2.;
    output[16] = (5*xi[1]*(-2-2*(-3+sqrt5)*xi[0]+(3+sqrt5)*xi[1]))/2.;
    output[17] = (-5*xi[0]*(2+(-3+sqrt5)*xi[0]-2*(3+sqrt5)*xi[1]))/2.;
    output[18] = -(xi[1]*(-1+2*xi[0]+xi[1]));
    output[19] = 1+xi[0]-xi[0]*xi[0]-2*(5+xi[0])*xi[1]+15*xi[1]*xi[1];
    return;
  }
}

void GaussLobattoInterpolationQuadrilateral(const double * xi, int n, double * output) {
  double * N[2] = {new double[n], new double[n]};

  GaussLobattoInterpolation(xi[0], n, N[0]);
  GaussLobattoInterpolation(xi[1], n, N[1]);

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      output[j * n + i] = N[0][i] * N[0][j];
    }
  }

  delete N[0];
  delete N[1];
}

void GaussLobattoInterpolationDerivativeQuadrilateral(const double * xi, int n, double * output) {
  double * N[2] = {new double[n], new double[n]};
  double * dN[2] = {new double[n], new double[n]};

  GaussLobattoInterpolation(xi[0], n, N[0]);
  GaussLobattoInterpolation(xi[1], n, N[1]);

  GaussLobattoInterpolation(xi[0], n, dN[0]);
  GaussLobattoInterpolation(xi[1], n, dN[1]);

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      output[2 * (j * n + i) + 0] = dN[0][i] *  N[0][j];
      output[2 * (j * n + i) + 1] =  N[0][i] * dN[0][j];
    }
  }

  delete N[0]; 
  delete N[1];
  delete dN[0]; 
  delete dN[1];
}

void GaussLobattoInterpolationTetrahedron(const double * xi, int p, double * output) {
  if (p == 0) {
    output[0] = 1.0;
    return;
  }
  if (p == 1) {
    output[0] = 1-xi[0]-xi[1]-xi[2];
    output[1] = xi[0];
    output[2] = xi[1];
    output[3] = xi[2];
    return;
  }
  if (p == 2) {
    output[0] = (-1+xi[0]+xi[1]+xi[2])*(-1+2*xi[0]+2*xi[1]+2*xi[2]);
    output[1] = -4*xi[0]*(-1+xi[0]+xi[1]+xi[2]);
    output[2] = xi[0]*(-1+2*xi[0]);
    output[3] = -4*xi[1]*(-1+xi[0]+xi[1]+xi[2]);
    output[4] = 4*xi[0]*xi[1];
    output[5] = xi[1]*(-1+2*xi[1]);
    output[6] = -4*xi[2]*(-1+xi[0]+xi[1]+xi[2]);
    output[7] = 4*xi[0]*xi[2];
    output[8] = 4*xi[1]*xi[2];
    output[9] = xi[2]*(-1+2*xi[2]);
    return;
  }
  if (p == 3) {
    double sqrt5 = 2.23606797749978981;
    output[ 0] = -((-1+xi[0]+xi[1]+xi[2])*(1+5*xi[0]*xi[0]+5*xi[1]*xi[1]+5*(-1+xi[2])*xi[2]+xi[1]*(-5+11*xi[2])+xi[0]*(-5+11*xi[1]+11*xi[2])));
    output[ 1] = (5*xi[0]*(-1+xi[0]+xi[1]+xi[2])*(-1-sqrt5+2*sqrt5*xi[0]+(3+sqrt5)*xi[1]+(3+sqrt5)*xi[2]))/2.;
    output[ 2] = (-5*xi[0]*(-1+xi[0]+xi[1]+xi[2])*(1-sqrt5+2*sqrt5*xi[0]+(-3+sqrt5)*xi[1]+(-3+sqrt5)*xi[2]))/2.;
    output[ 3] = xi[0]*(1+5*xi[0]*xi[0]+xi[1]-xi[1]*xi[1]+xi[2]-xi[1]*xi[2]-xi[2]*xi[2]-xi[0]*(5+xi[1]+xi[2]));
    output[ 4] = (5*xi[1]*(-1+xi[0]+xi[1]+xi[2])*(-1-sqrt5+(3+sqrt5)*xi[0]+2*sqrt5*xi[1]+(3+sqrt5)*xi[2]))/2.;
    output[ 5] = -27*xi[0]*xi[1]*(-1+xi[0]+xi[1]+xi[2]);
    output[ 6] = (5*xi[0]*xi[1]*(-2+(3+sqrt5)*xi[0]-(-3+sqrt5)*xi[1]))/2.;
    output[ 7] = (-5*xi[1]*(-1+xi[0]+xi[1]+xi[2])*(1-sqrt5+(-3+sqrt5)*xi[0]+2*sqrt5*xi[1]+(-3+sqrt5)*xi[2]))/2.;
    output[ 8] = (-5*xi[0]*xi[1]*(2+(-3+sqrt5)*xi[0]-(3+sqrt5)*xi[1]))/2.;
    output[ 9] = xi[1]*(1-xi[0]*xi[0]+5*xi[1]*xi[1]+xi[2]-xi[2]*xi[2]-xi[1]*(5+xi[2])-xi[0]*(-1+xi[1]+xi[2]));
    output[10] = (5*xi[2]*(-1+xi[0]+xi[1]+xi[2])*(-439204-196418*sqrt5+(710647+317811*sqrt5)*xi[0]+(710647+317811*sqrt5)*xi[1]+606965*xi[2]+271443*sqrt5*xi[2]))/(271443+121393*sqrt5);
    output[11] = -27*xi[0]*xi[2]*(-1+xi[0]+xi[1]+xi[2]);
    output[12] = (5*xi[0]*xi[2]*(-5-3*sqrt5+(15+7*sqrt5)*xi[0]+2*sqrt5*xi[2]))/(5+3*sqrt5);
    output[13] = -27*xi[1]*xi[2]*(-1+xi[0]+xi[1]+xi[2]);
    output[14] = 27*xi[0]*xi[1]*xi[2];
    output[15] = (5*xi[1]*xi[2]*(-5-3*sqrt5+(15+7*sqrt5)*xi[1]+2*sqrt5*xi[2]))/(5+3*sqrt5);
    output[16] = (5*xi[2]*(-1+xi[0]+xi[1]+xi[2])*(88555+39603*sqrt5+(54730+24476*sqrt5)*xi[0]+(54730+24476*sqrt5)*xi[1]-5*(64079+28657*sqrt5)*xi[2]))/(143285+64079*sqrt5);
    output[17] = (-5*xi[0]*xi[2]*(2+(-3+sqrt5)*xi[0]-(3+sqrt5)*xi[2]))/2.;
    output[18] = (-5*xi[1]*xi[2]*(2+(-3+sqrt5)*xi[1]-(3+sqrt5)*xi[2]))/2.;
    output[19] = -(xi[2]*(-1+xi[0]*xi[0]+xi[1]*xi[1]+xi[1]*(-1+xi[2])-5*(-1+xi[2])*xi[2]+xi[0]*(-1+xi[1]+xi[2])));
    return;
  }
}

void GaussLobattoInterpolationDerivativeTetrahedron(const double * xi, int p, double * output) {
  if (p == 0) {
    output[0] = 0.0;
    output[1] = 0.0;
    output[2] = 0.0;
    return;
  }
  if (p == 1) {
    output[ 0] = -1;
    output[ 1] = -1;
    output[ 2] = -1;
    output[ 3] = 1;
    output[ 4] = 0;
    output[ 5] = 0;
    output[ 6] = 0;
    output[ 7] = 1;
    output[ 8] = 0;
    output[ 9] = 0;
    output[10] = 0;
    output[11] = 1;
    return;
  }
  if (p == 2) {
    output[ 0] = -3+4*xi[0]+4*xi[1]+4*xi[2];
    output[ 1] = -3+4*xi[0]+4*xi[1]+4*xi[2];
    output[ 2] = -3+4*xi[0]+4*xi[1]+4*xi[2];
    output[ 3] = -4*(-1+2*xi[0]+xi[1]+xi[2]);
    output[ 4] = -4*xi[0];
    output[ 5] = -4*xi[0];
    output[ 6] = -1+4*xi[0];
    output[ 7] = 0;
    output[ 8] = 0;
    output[ 9] = -4*xi[1];
    output[10] = -4*(-1+xi[0]+2*xi[1]+xi[2]);
    output[11] = -4*xi[1];
    output[12] = 4*xi[1];
    output[13] = 4*xi[0];
    output[14] = 0;
    output[15] = 0;
    output[16] = -1+4*xi[1];
    output[17] = 0;
    output[18] = -4*xi[2];
    output[19] = -4*xi[2];
    output[20] = -4*(-1+xi[0]+xi[1]+2*xi[2]);
    output[21] = 4*xi[2];
    output[22] = 0;
    output[23] = 4*xi[0];
    output[24] = 0;
    output[25] = 4*xi[2];
    output[26] = 4*xi[1];
    output[27] = 0;
    output[28] = 0;
    output[29] = -1+4*xi[2];
    return;
  }
  if (p == 3) {
    double sqrt5 = 2.23606797749978981;
    output[ 0] = -6-15*xi[0]*xi[0]-16*xi[1]*xi[1]+xi[1]*(21-33*xi[2])+(21-16*xi[2])*xi[2]-4*xi[0]*(-5+8*xi[1]+8*xi[2]);
    output[ 1] = -6-16*xi[0]*xi[0]+20*xi[1]+xi[0]*(21-32*xi[1]-33*xi[2])+21*xi[2]-(3*xi[1]+4*xi[2])*(5*xi[1]+4*xi[2]);
    output[ 2] = -6-16*xi[0]*xi[0]+21*xi[1]+xi[0]*(21-33*xi[1]-32*xi[2])+20*xi[2]-(4*xi[1]+3*xi[2])*(4*xi[1]+5*xi[2]);
    output[ 3] = (5*(6*sqrt5*xi[0]*xi[0]+xi[0]*(-2-6*sqrt5+6*(1+sqrt5)*xi[1]+6*(1+sqrt5)*xi[2])+(-1+xi[1]+xi[2])*(-1-sqrt5+(3+sqrt5)*xi[1]+(3+sqrt5)*xi[2])))/2.;
    output[ 4] = (5*xi[0]*(-4-2*sqrt5+3*(1+sqrt5)*xi[0]+2*(3+sqrt5)*xi[1]+2*(3+sqrt5)*xi[2]))/2.;
    output[ 5] = (5*xi[0]*(-4-2*sqrt5+3*(1+sqrt5)*xi[0]+2*(3+sqrt5)*xi[1]+2*(3+sqrt5)*xi[2]))/2.;
    output[ 6] = -15*sqrt5*xi[0]*xi[0]-(5*(-1+xi[1]+xi[2])*(1-sqrt5+(-3+sqrt5)*xi[1]+(-3+sqrt5)*xi[2]))/2.-5*xi[0]*(1-3*sqrt5+3*(-1+sqrt5)*xi[1]+3*(-1+sqrt5)*xi[2]);
    output[ 7] = (-5*xi[0]*(4-2*sqrt5+3*(-1+sqrt5)*xi[0]+2*(-3+sqrt5)*xi[1]+2*(-3+sqrt5)*xi[2]))/2.;
    output[ 8] = (-5*xi[0]*(4-2*sqrt5+3*(-1+sqrt5)*xi[0]+2*(-3+sqrt5)*xi[1]+2*(-3+sqrt5)*xi[2]))/2.;
    output[ 9] = 1+15*xi[0]*xi[0]+xi[1]-xi[1]*xi[1]+xi[2]-xi[1]*xi[2]-xi[2]*xi[2]-2*xi[0]*(5+xi[1]+xi[2]);
    output[10] = -(xi[0]*(-1+xi[0]+2*xi[1]+xi[2]));
    output[11] = -(xi[0]*(-1+xi[0]+xi[1]+2*xi[2]));
    output[12] = (5*xi[1]*(-2*(2+sqrt5)+2*(3+sqrt5)*xi[0]+3*(1+sqrt5)*xi[1]+2*(3+sqrt5)*xi[2]))/2.;
    output[13] = 15*sqrt5*xi[1]*xi[1]+5*xi[1]*(-1-3*sqrt5+3*(1+sqrt5)*xi[0]+3*(1+sqrt5)*xi[2])+(5*(-1+xi[0]+xi[2])*(-1-sqrt5+(3+sqrt5)*xi[0]+(3+sqrt5)*xi[2]))/2.;
    output[14] = (5*xi[1]*(-2*(2+sqrt5)+2*(3+sqrt5)*xi[0]+3*(1+sqrt5)*xi[1]+2*(3+sqrt5)*xi[2]))/2.;
    output[15] = -27*xi[1]*(-1+2*xi[0]+xi[1]+xi[2]);
    output[16] = -27*xi[0]*(-1+xi[0]+2*xi[1]+xi[2]);
    output[17] = -27*xi[0]*xi[1];
    output[18] = (-5*xi[1]*(2-2*(3+sqrt5)*xi[0]+(-3+sqrt5)*xi[1]))/2.;
    output[19] = (5*xi[0]*(-2+(3+sqrt5)*xi[0]-2*(-3+sqrt5)*xi[1]))/2.;
    output[20] = 0;
    output[21] = (-5*xi[1]*(4-2*sqrt5+2*(-3+sqrt5)*xi[0]+3*(-1+sqrt5)*xi[1]+2*(-3+sqrt5)*xi[2]))/2.;
    output[22] = -15*sqrt5*xi[1]*xi[1]-(5*(-1+xi[0]+xi[2])*(1-sqrt5+(-3+sqrt5)*xi[0]+(-3+sqrt5)*xi[2]))/2.-5*xi[1]*(1-3*sqrt5+3*(-1+sqrt5)*xi[0]+3*(-1+sqrt5)*xi[2]);
    output[23] = (-5*xi[1]*(4-2*sqrt5+2*(-3+sqrt5)*xi[0]+3*(-1+sqrt5)*xi[1]+2*(-3+sqrt5)*xi[2]))/2.;
    output[24] = (5*xi[1]*(-2-2*(-3+sqrt5)*xi[0]+(3+sqrt5)*xi[1]))/2.;
    output[25] = (-5*xi[0]*(2+(-3+sqrt5)*xi[0]-2*(3+sqrt5)*xi[1]))/2.;
    output[26] = 0;
    output[27] = -(xi[1]*(-1+2*xi[0]+xi[1]+xi[2]));
    output[28] = 1-xi[0]*xi[0]+15*xi[1]*xi[1]+xi[2]-xi[2]*xi[2]-2*xi[1]*(5+xi[2])-xi[0]*(-1+2*xi[1]+xi[2]);
    output[29] = -(xi[1]*(-1+xi[0]+xi[1]+2*xi[2]));
    output[30] = (5*xi[2]*(-2*(2+sqrt5)+2*(3+sqrt5)*xi[0]+2*(3+sqrt5)*xi[1]+3*(1+sqrt5)*xi[2]))/2.;
    output[31] = (5*xi[2]*(-2*(2+sqrt5)+2*(3+sqrt5)*xi[0]+2*(3+sqrt5)*xi[1]+3*(1+sqrt5)*xi[2]))/2.;
    output[32] = (5*(1+sqrt5+(3+sqrt5)*xi[0]*xi[0]-2*(2+sqrt5)*xi[1]+(3+sqrt5)*xi[1]*xi[1]+6*(1+sqrt5)*xi[1]*xi[2]+2*xi[2]*(-1-3*sqrt5+3*sqrt5*xi[2])+2*xi[0]*(-2-sqrt5+(3+sqrt5)*xi[1]+3*(1+sqrt5)*xi[2])))/2.;
    output[33] = -27*xi[2]*(-1+2*xi[0]+xi[1]+xi[2]);
    output[34] = -27*xi[0]*xi[2];
    output[35] = -27*xi[0]*(-1+xi[0]+xi[1]+2*xi[2]);
    output[36] = (-5*xi[2]*(2-2*(3+sqrt5)*xi[0]+(-3+sqrt5)*xi[2]))/2.;
    output[37] = 0;
    output[38] = (5*xi[0]*(-2+(3+sqrt5)*xi[0]-2*(-3+sqrt5)*xi[2]))/2.;
    output[39] = -27*xi[1]*xi[2];
    output[40] = -27*xi[2]*(-1+xi[0]+2*xi[1]+xi[2]);
    output[41] = -27*xi[1]*(-1+xi[0]+xi[1]+2*xi[2]);
    output[42] = 27*xi[1]*xi[2];
    output[43] = 27*xi[0]*xi[2];
    output[44] = 27*xi[0]*xi[1];
    output[45] = 0;
    output[46] = (-5*xi[2]*(2-2*(3+sqrt5)*xi[1]+(-3+sqrt5)*xi[2]))/2.;
    output[47] = (5*xi[1]*(-2+(3+sqrt5)*xi[1]-2*(-3+sqrt5)*xi[2]))/2.;
    output[48] = (-5*xi[2]*(4-2*sqrt5+2*(-3+sqrt5)*xi[0]+2*(-3+sqrt5)*xi[1]+3*(-1+sqrt5)*xi[2]))/2.;
    output[49] = (-5*xi[2]*(4-2*sqrt5+2*(-3+sqrt5)*xi[0]+2*(-3+sqrt5)*xi[1]+3*(-1+sqrt5)*xi[2]))/2.;
    output[50] = (-5*(-3+sqrt5)*xi[0]*xi[0])/2.-5*xi[0]*(2-sqrt5+(-3+sqrt5)*xi[1]+3*(-1+sqrt5)*xi[2])-(5*(-1+sqrt5+(-3+sqrt5)*xi[1]*xi[1]+2*xi[2]*(1-3*sqrt5+3*sqrt5*xi[2])+xi[1]*(4-2*sqrt5+6*(-1+sqrt5)*xi[2])))/2.;
    output[51] = (5*xi[2]*(-2-2*(-3+sqrt5)*xi[0]+(3+sqrt5)*xi[2]))/2.;
    output[52] = 0;
    output[53] = (-5*xi[0]*(2+(-3+sqrt5)*xi[0]-2*(3+sqrt5)*xi[2]))/2.;
    output[54] = 0;
    output[55] = (5*xi[2]*(-2-2*(-3+sqrt5)*xi[1]+(3+sqrt5)*xi[2]))/2.;
    output[56] = (-5*xi[1]*(2+(-3+sqrt5)*xi[1]-2*(3+sqrt5)*xi[2]))/2.;
    output[57] = -(xi[2]*(-1+2*xi[0]+xi[1]+xi[2]));
    output[58] = -(xi[2]*(-1+xi[0]+2*xi[1]+xi[2]));
    output[59] = 1+xi[0]-xi[0]*xi[0]+xi[1]-xi[0]*xi[1]-xi[1]*xi[1]-2*(5+xi[0]+xi[1])*xi[2]+15*xi[2]*xi[2];
    return;
  }
}

void GaussLobattoInterpolationHexahedron(const double * xi, int n, double * output) {
  double * N[3] = {new double[n], new double[n], new double[n]};

  GaussLobattoInterpolation(xi[0], n, N[0]);
  GaussLobattoInterpolation(xi[1], n, N[1]);
  GaussLobattoInterpolation(xi[2], n, N[2]);

  for (int k = 0; k < n; k++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        output[(k * n + j) * n + i] = N[0][i] * N[1][j] * N[2][k];
      }
    }
  }

  delete N[0]; delete N[1]; delete N[2];
}

void GaussLobattoInterpolationDerivativeHexahedron(const double * xi, int n, double * output) {
  double * N[3] = {new double[n], new double[n], new double[n]};
  double * dN[3] = {new double[n], new double[n], new double[n]};

  GaussLobattoInterpolation(xi[0], n, N[0]);
  GaussLobattoInterpolation(xi[1], n, N[1]);
  GaussLobattoInterpolation(xi[2], n, N[2]);

  GaussLobattoInterpolation(xi[0], n, dN[0]);
  GaussLobattoInterpolation(xi[1], n, dN[1]);
  GaussLobattoInterpolation(xi[2], n, dN[2]);

  for (int k = 0; k < n; k++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        output[3 * ((k * n + j) * n + i) + 0] = dN[0][i] *  N[1][j] *  N[2][k];
        output[3 * ((k * n + j) * n + i) + 1] =  N[0][i] * dN[1][j] *  N[2][k];
        output[3 * ((k * n + j) * n + i) + 2] =  N[0][i] *  N[1][j] * dN[2][k];
      }
    }
  }

  delete N[0]; delete N[1]; delete N[2];
  delete dN[0]; delete dN[1]; delete dN[2];
}

} // namespace femto
