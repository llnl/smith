#pragma once

#include "refactor/connection.hpp" 

namespace refactor {

template <>
struct FiniteElement<Geometry::Hexahedron, Family::H1> {

  using value_type = vec1;
  using derivative_type = vec3;

  using source_type = vec1;
  using flux_type = vec3;

  __host__ __device__ constexpr uint32_t num_nodes() const { return (p + 1) * (p + 1) * (p + 1); }

  void nodes(nd::view< double, 2 > xi) const {
    double lobatto_nodes[4];
    GaussLobattoNodes(p + 1, lobatto_nodes);

    uint32_t count = 0;
    for (uint32_t k = 0; k < (p+1); k++) {
      for (uint32_t j = 0; j < (p+1); j++) {
        for (uint32_t i = 0; i < (p+1); i++) {
          xi(count, 0) = lobatto_nodes[i];
          xi(count, 1) = lobatto_nodes[j];
          xi(count, 2) = lobatto_nodes[k];
          count++;
        }
      }
    }
  } 

  __host__ __device__ uint32_t num_interior_nodes() const { return (p == 1) ? 0 : (p - 1) * (p - 1) * (p - 1); }

  void interior_nodes(nd::view< double, 2 > xi) const {
    if (p > 1) {
      double lobatto_nodes[4];
      GaussLobattoNodes(p + 1, lobatto_nodes);

      uint32_t count = 0;
      for (uint32_t k = 1; k < p; k++) {
        for (uint32_t j = 1; j < p; j++) {
          for (uint32_t i = 1; i < p; i++) {
            xi(count, 0) = lobatto_nodes[i];
            xi(count, 1) = lobatto_nodes[j];
            xi(count, 2) = lobatto_nodes[k];
            count++;
          }
        }
      }

    }
  }

  __host__ __device__ inline void reorient(Connection c, uint32_t & i0, uint32_t & i1, uint32_t & i2, uint32_t & i3) const {

    uint8_t o = c.orientation();

    uint32_t copy[4] = {i0, i1, i2, i3};

    if (c.sign() == Sign::Negative) {
      if (o == 0) { i0 = copy[0]; i1 = copy[2]; i2 = copy[1]; i3 = copy[3]; } // ok
      if (o == 1) { i0 = copy[1]; i1 = copy[0]; i2 = copy[3]; i3 = copy[2]; } // ok
      if (o == 2) { i0 = copy[3]; i1 = copy[1]; i2 = copy[2]; i3 = copy[0]; } // ok
      if (o == 3) { i0 = copy[2]; i1 = copy[3]; i2 = copy[0]; i3 = copy[1]; } // ok
    } else {
      if (o == 0) { /* nothing to do */ }
      if (o == 1) { /* should never occur */ }
      if (o == 2) { /* should never occur */ }
      if (o == 3) { /* should never occur */ }
    }
  }

  __host__ __device__ void indices(const GeometryInfo & offsets, 
                                   const Connection * connections, 
                                   uint32_t * ids) const {
    
    const Connection * vertex = connections;
    const Connection * edge = connections + Hexahedron::edge_offset;
    const Connection * quad = connections + Hexahedron::quad_offset;
    const Connection * cell = connections + Hexahedron::cell_offset;

    if (p == 1) {
      ids[0] = offsets.vert + vertex[0].index;
      ids[1] = offsets.vert + vertex[1].index;
      ids[2] = offsets.vert + vertex[3].index;
      ids[3] = offsets.vert + vertex[2].index;
      ids[4] = offsets.vert + vertex[4].index;
      ids[5] = offsets.vert + vertex[5].index;
      ids[6] = offsets.vert + vertex[7].index;
      ids[7] = offsets.vert + vertex[6].index;
      return;
    }

    if (p == 2) {
      #define VERTEX_ID(i) offsets.vert + vertex[i].index
      #define EDGE_ID(i) offsets.edge + edge[i].index
      #define QUAD_ID(i) offsets.quad + quad[i].index
      #define HEX_ID offsets.hex + cell->index

      ids[ 6] = VERTEX_ID(3); ids[ 7] = EDGE_ID(2); ids[ 8] = VERTEX_ID(2);
      ids[ 3] =   EDGE_ID(3); ids[ 4] = QUAD_ID(0); ids[ 5] =   EDGE_ID(1);
      ids[ 0] = VERTEX_ID(0); ids[ 1] = EDGE_ID(0); ids[ 2] = VERTEX_ID(1);

      ids[15] = EDGE_ID(7); ids[16] = QUAD_ID(3); ids[17] = EDGE_ID(6);
      ids[12] = QUAD_ID(4); ids[13] = HEX_ID;     ids[14] = QUAD_ID(2);
      ids[ 9] = EDGE_ID(4); ids[10] = QUAD_ID(1); ids[11] = EDGE_ID(5);

      ids[24] = VERTEX_ID(7);  ids[25] = EDGE_ID(10); ids[26] = VERTEX_ID(6);
      ids[21] =   EDGE_ID(11); ids[22] = QUAD_ID(5);  ids[23] =   EDGE_ID(9);
      ids[18] = VERTEX_ID(4);  ids[19] = EDGE_ID(8);  ids[20] = VERTEX_ID(5);

      #undef VERTEX_ID
      #undef EDGE_ID
      #undef QUAD_ID
      #undef HEX_ID
      return;
    }

    if (p == 3) {
      #define VERTEX_ID(i) offsets.vert + vertex[i].index
      #define EDGE_ID(i, j) offsets.edge + 2 * edge[i].index + j
      #define QUAD_ID(i, j) offsets.quad + 4 * quad[i].index + j
      #define HEX_ID(j) offsets.hex + 8 * cell->index + j

      ids[12] = VERTEX_ID(3); ids[13] = EDGE_ID(2,0); ids[14] = EDGE_ID(2,1); ids[15] = VERTEX_ID(2);
      ids[ 8] = EDGE_ID(3,1); ids[ 9] = QUAD_ID(0,3); ids[10] = QUAD_ID(0,2); ids[11] = EDGE_ID(1,1);
      ids[ 4] = EDGE_ID(3,0); ids[ 5] = QUAD_ID(0,1); ids[ 6] = QUAD_ID(0,0); ids[ 7] = EDGE_ID(1,0);
      ids[ 0] = VERTEX_ID(0); ids[ 1] = EDGE_ID(0,0); ids[ 2] = EDGE_ID(0,1); ids[ 3] = VERTEX_ID(1);

      ids[28] = EDGE_ID(7,0); ids[29] = QUAD_ID(3,1); ids[30] = QUAD_ID(3,0); ids[31] = EDGE_ID(6,0);
      ids[24] = QUAD_ID(4,0); ids[25] = HEX_ID(2);    ids[26] = HEX_ID(3);    ids[27] = QUAD_ID(2,1);
      ids[20] = QUAD_ID(4,1); ids[21] = HEX_ID(0);    ids[22] = HEX_ID(1);    ids[23] = QUAD_ID(2,0);
      ids[16] = EDGE_ID(4,0); ids[17] = QUAD_ID(1,0); ids[18] = QUAD_ID(1,1); ids[19] = EDGE_ID(5,0);

      ids[44] = EDGE_ID(7,1); ids[45] = QUAD_ID(3,3); ids[46] = QUAD_ID(3,2); ids[47] = EDGE_ID(6,1);
      ids[40] = QUAD_ID(4,2); ids[41] = HEX_ID(6);    ids[42] = HEX_ID(7);    ids[43] = QUAD_ID(2,3);
      ids[36] = QUAD_ID(4,3); ids[37] = HEX_ID(4);    ids[38] = HEX_ID(5);    ids[39] = QUAD_ID(2,2);
      ids[32] = EDGE_ID(4,1); ids[33] = QUAD_ID(1,2); ids[34] = QUAD_ID(1,3); ids[35] = EDGE_ID(5,1);

      ids[60] = VERTEX_ID(7);  ids[61] = EDGE_ID(10,0); ids[62] = EDGE_ID(10,1); ids[63] = VERTEX_ID(6);  
      ids[56] = EDGE_ID(11,1); ids[57] = QUAD_ID(5,2);  ids[58] = QUAD_ID(5,3);  ids[59] = EDGE_ID(9,1);
      ids[52] = EDGE_ID(11,0); ids[53] = QUAD_ID(5,0);  ids[54] = QUAD_ID(5,1);  ids[55] = EDGE_ID(9,0);
      ids[48] = VERTEX_ID(4);  ids[49] = EDGE_ID(8,0);  ids[50] = EDGE_ID(8,1);  ids[51] = VERTEX_ID(5);  

      if (edge[ 0].sign() == Sign::Negative) { fm::swap(ids[ 1], ids[ 2]); }
      if (edge[ 1].sign() == Sign::Negative) { fm::swap(ids[ 7], ids[11]); }
      if (edge[ 2].sign() == Sign::Negative) { fm::swap(ids[14], ids[13]); }
      if (edge[ 3].sign() == Sign::Negative) { fm::swap(ids[ 8], ids[ 4]); }
      if (edge[ 4].sign() == Sign::Negative) { fm::swap(ids[16], ids[32]); }
      if (edge[ 5].sign() == Sign::Negative) { fm::swap(ids[19], ids[35]); }
      if (edge[ 6].sign() == Sign::Negative) { fm::swap(ids[31], ids[47]); }
      if (edge[ 7].sign() == Sign::Negative) { fm::swap(ids[28], ids[44]); }
      if (edge[ 8].sign() == Sign::Negative) { fm::swap(ids[49], ids[50]); }
      if (edge[ 9].sign() == Sign::Negative) { fm::swap(ids[55], ids[59]); }
      if (edge[10].sign() == Sign::Negative) { fm::swap(ids[62], ids[61]); }
      if (edge[11].sign() == Sign::Negative) { fm::swap(ids[56], ids[52]); }

      reorient(quad[0], ids[ 9], ids[10], ids[ 5], ids[ 6]);
      reorient(quad[1], ids[17], ids[18], ids[33], ids[34]);
      reorient(quad[2], ids[23], ids[27], ids[39], ids[43]);
      reorient(quad[3], ids[30], ids[29], ids[46], ids[45]);
      reorient(quad[4], ids[24], ids[20], ids[40], ids[36]);
      reorient(quad[5], ids[53], ids[54], ids[57], ids[58]);

      #undef VERTEX_ID
      #undef EDGE_ID
      #undef QUAD_ID
      #undef HEX_ID

      return;
 
    }
  
  }

  constexpr double phi_1D(double xi, uint32_t i) const {
    return GaussLobattoInterpolation(xi, p + 1, i); 
  }

  constexpr double dphi_1D(double xi, uint32_t i) const {
    return GaussLobattoInterpolationDerivative(xi, p + 1, i); 
  }

  constexpr double shape_function(vec3 xi, uint32_t i) const {
    uint32_t ix = i % (p + 1);
    uint32_t iy = (i % ((p + 1) * (p + 1))) / (p + 1) ;
    uint32_t iz = i / ((p + 1) * (p + 1));
    return phi_1D(xi[0], ix) * phi_1D(xi[1], iy) * phi_1D(xi[2], iz);
  }

  constexpr vec3 shape_function_gradient(vec3 xi, uint32_t i) const {
    uint32_t ix = i % (p + 1);
    uint32_t iy = (i % ((p + 1) * (p + 1))) / (p + 1) ;
    uint32_t iz = i / ((p + 1) * (p + 1));
    return {
      dphi_1D(xi[0], ix) *  phi_1D(xi[1], iy) *  phi_1D(xi[2], iz),
       phi_1D(xi[0], ix) * dphi_1D(xi[1], iy) *  phi_1D(xi[2], iz),
       phi_1D(xi[0], ix) *  phi_1D(xi[1], iy) * dphi_1D(xi[2], iz)
    };
  }

  vec3 shape_function_derivative(vec3 xi, uint32_t i) const {
    return shape_function_gradient(xi, i);
  }

  double interpolate(vec3 xi, double * values) const {
    uint32_t count = 0;
    double output = 0.0;
    for (int iz = 0; iz < (p+1); iz++) {
      double phi_z = phi_1D(xi[2], iz);
      for (int iy = 0; iy < (p+1); iy++) {
        double phi_y = phi_1D(xi[1], iy);
        for (int ix = 0; ix < (p+1); ix++) {
          double phi_x = phi_1D(xi[0], ix);
          output += phi_x * phi_y * phi_z * values[count];
          count++;
        }
      }
    }
    return output;
  }

  vec3 gradient(vec3 xi, double * values) const {
    vec3 output{};
    uint32_t count = 0;
    for (int iz = 0; iz < (p+1); iz++) {
      double phi_z  = phi_1D(xi[2], iz);
      double dphi_z = dphi_1D(xi[2], iz);
      for (int iy = 0; iy < (p+1); iy++) {
        double phi_y  = phi_1D(xi[1], iy);
        double dphi_y = dphi_1D(xi[1], iy);
        for (int ix = 0; ix < (p+1); ix++) {
          double phi_x  = phi_1D(xi[0], ix);
          double dphi_x = dphi_1D(xi[0], ix);
          output[0] += dphi_x *  phi_y *  phi_z * values[count];
          output[1] +=  phi_x * dphi_y *  phi_z * values[count];
          output[2] +=  phi_x *  phi_y * dphi_z * values[count];
          count++;
        }
      }
    }
    return output;
  }

  __host__ __device__ uint32_t batch_interpolation_scratch_space(nd::view<const double,2> xi) const {
    uint32_t n = p + 1;
    uint32_t q = xi.shape[0];
    return q * n * (n + q);
  }

  // note: the return value here includes the shape function evaluations as well as
  // a buffer space to hold intermediate values in the `interpolation` function
  nd::array< double, 3 > evaluate_shape_functions(nd::view<const double,2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 3> buffer({1, q, p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p+1, &buffer(0, i, 0));
    }
    return buffer;
  }

  __host__ __device__ void interpolate(nd::view<value_type> values_q, nd::view<const double> values_e, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[1];

    nd::view<double, 2> phi(&shape_fns(0,0,0), {q, n});
    nd::view<double, 3> A1(buffer, {q, n, n});
    nd::view<double, 3> A2(buffer + q * n * n, {q, q, n});

    nd::view<const double, 3> values_e_3D = nd::reshape<3>(values_e, {n, n, n});
    nd::view<double, 3> values_q_3D((double*)&values_q[0], {q, q, q});
    contract(values_e_3D, phi, A1);
    contract(         A1, phi, A2);
    contract(         A2, phi, values_q_3D);
  }

  // note: the return value here includes the shape function evaluations as well as
  // a buffer space to hold intermediate values in the `interpolation` function
  nd::array< double, 3 > evaluate_shape_function_gradients(nd::view<const double,2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 3> shape_fn_grads({2, q, p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p+1, &shape_fn_grads(0, i, 0));
      GaussLobattoInterpolationDerivative(xi(i, 0), p+1, &shape_fn_grads(1, i, 0));
    }
    return shape_fn_grads;
  }

  __host__ __device__ void gradient(nd::view<derivative_type> gradients_q, nd::view<const double, 1> values_e, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[1];

    nd::view<double, 2> B(&shape_fns(0,0,0), {q, n});
    nd::view<double, 2> G(&shape_fns(1,0,0), {q, n});
    nd::view<double, 3> A1(buffer, {q, n, n});
    nd::view<double, 3> A2(buffer + q * n * n, {q, q, n});

    nd::range All{0u, gradients_q.shape[0]};

    nd::view<const double, 3> values_e_3D = nd::reshape<3>(values_e, {n, n, n});
    nd::view<double, 3> gradients_q_3D((double*)&gradients_q[0], {q, q, q}, {3*q*q, 3*q, 3});

    // du_dxi
    contract(values_e_3D, G, A1);
    contract(         A1, B, A2);
    contract(         A2, B, gradients_q_3D);
    gradients_q_3D.values++;

    // du_deta
    contract(values_e_3D, B, A1);
    contract(         A1, G, A2);
    contract(         A2, B, gradients_q_3D);
    gradients_q_3D.values++;

    // du_dzeta
    contract(A1, B, A2);
    contract(A2, G, gradients_q_3D);
    gradients_q_3D.values++;

  }

  // note: the return value here includes the shape function evaluations as well as
  // a buffer space to hold intermediate values in the `interpolation` function
  nd::array< double, 2 > evaluate_weighted_shape_functions(nd::view<const double,2> xi,
                                                           nd::view<const double,1> weights) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 2> buffer({p+1, q});
    nd::array<double, 1> tmp({p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p+1, &tmp(0));
      for (int j = 0; j < p+1; j++) {
        buffer(j, i) = tmp(j) * weights(i);
      }
    }
    return buffer;
  }

  __host__ __device__ void integrate_source(nd::view<double, 1> residual_e, nd::view<const source_type, 1> source_q, nd::view<double, 2> shape_fn, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fn.shape[1];

    nd::view<double, 2> BT = shape_fn;
    nd::view<double, 3> A1(buffer, {n, q, q});
    nd::view<double, 3> A2(buffer + n*q*q, {n, n, q});

    nd::view<const double, 3> s3D((double*)&source_q[0], {q, q, q});
    nd::view<double, 3> r3D = nd::reshape<3>(residual_e, {n, n, n});

    //   r(iz, iy, ix) = s(qz, qy, qx) * BT(ix, qx) * BT(iy, qy) * BT(iz, qz)
    // 
    //  A1(ix, qz, qy) = BT(ix, qx) *  s(qz, qy, qx)
    //  A2(iy, ix, qz) = BT(iy, qy) * A1(ix, qz, qy)
    //   r(iz, iy, ix) = BT(iz, qz) * A2(iy, ix, qz)
    contract(s3D, BT, A1);
    contract( A1, BT, A2);
    contract( A2, BT, r3D);
  }

  // note: the return value here includes the shape function evaluations as well as
  // a buffer space to hold intermediate values in the `interpolation` function
  nd::array< double, 3 > evaluate_weighted_shape_function_gradients(nd::view<const double,2> xi,
                                                                    nd::view<const double,1> weights) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 3> buffer({2, p+1, q});
    nd::array<double, 1> tmp({p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p+1, &tmp(0));
      for (int j = 0; j < p+1; j++) {
        buffer(0, j, i) = tmp(j) * weights(i);
      }

      GaussLobattoInterpolationDerivative(xi(i, 0), p+1, &tmp(0));
      for (int j = 0; j < p+1; j++) {
        buffer(1, j, i) = tmp(j) * weights(i);
      }
    }
    return buffer;
  }

  void integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[2];

    nd::view<double, 2> BT = shape_fns(0);
    nd::view<double, 2> GT = shape_fns(1);
    nd::view<double, 3> A1(buffer, {n, q, q});
    nd::view<double, 3> A2(buffer+n*q*q, {n, n, q});

    nd::range All{0u, flux_q.shape[0]};

    uint32_t s = flux_q.stride[0] * 3;
    nd::view<const double, 3> f3D(nullptr, {q, q, q}, {s*q*q, s*q, s});
    nd::view<double, 3> r3D = nd::reshape<3>(residual_e, {n, n, n});

    f3D.values = &flux_q(0)[0];
    contract(f3D, GT, A1);
    contract( A1, BT, A2);
    contract( A2, BT, r3D, /* accumulate = */ false);

    f3D.values = &flux_q(0)[1];
    contract(f3D, BT, A1);
    contract( A1, GT, A2);
    contract( A2, BT, r3D, /* accumulate = */ true);

    f3D.values = &flux_q(0)[2];
    contract(f3D, BT, A1);
    contract( A1, BT, A2);
    contract( A2, GT, r3D, /* accumulate = */ true);
  }

  #ifdef __CUDACC__ 
  __device__ void cuda_gradient(nd::view<vec3> gradients_q, nd::view<double, 1> values_e, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[1];

    nd::view<double, 2> B(&shape_fns(0,0,0), {q, n});
    nd::view<double, 2> G(&shape_fns(1,0,0), {q, n});
    nd::view<double, 3> A1(buffer, {q, n, n});
    nd::view<double, 3> A2(buffer + q * n * n, {q, q, n});

    nd::range All{0u, gradients_q.shape[0]};

    nd::view<double, 3> values_e_3D = nd::reshape<3>(values_e, {n, n, n});
    nd::view<double, 3> gradients_q_3D((double*)&gradients_q[0], {q, q, q}, {3*q*q, 3*q, 3});

    threadid tid; 
    tid.x = threadIdx.x % q;
    tid.y = (threadIdx.x % (q * q)) / q;
    tid.z = threadIdx.x / (q * q);
    tid.stride = q;

    // du_dxi
    cuda_contract(tid, values_e_3D, G, A1);
    cuda_contract(tid,          A1, B, A2);
    cuda_contract(tid,          A2, B, gradients_q_3D);
    gradients_q_3D.values++;

    // du_deta
    cuda_contract(tid, values_e_3D, B, A1);
    cuda_contract(tid,          A1, G, A2);
    cuda_contract(tid,          A2, B, gradients_q_3D);
    gradients_q_3D.values++;

    // du_dzeta
    cuda_contract(tid, A1, B, A2);
    cuda_contract(tid, A2, G, gradients_q_3D);
    gradients_q_3D.values++;

  }

  __device__ void cuda_integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[2];

    nd::view<double, 2> BT = shape_fns(0);
    nd::view<double, 2> GT = shape_fns(1);
    nd::view<double, 3> A1(buffer, {n, q, q});
    nd::view<double, 3> A2(buffer+n*q*q, {n, n, q});

    uint32_t s = flux_q.stride[0] * 3;
    nd::view<double, 3> f3D(nullptr, {q, q, q}, {s*q*q, s*q, s});
    nd::view<double, 3> r3D = nd::reshape<3>(residual_e, {n, n, n});

    threadid tid; 
    tid.x = threadIdx.x % n;
    tid.y = (threadIdx.x % (n * n)) / n;
    tid.z = threadIdx.x / (n * n);
    tid.stride = n;

    f3D.values = &flux_q(0)[0];
    cuda_contract(tid, f3D, GT, A1);
    cuda_contract(tid,  A1, BT, A2);
    cuda_contract(tid,  A2, BT, r3D, /* accumulate = */ false);

    f3D.values = &flux_q(0)[1];
    cuda_contract(tid, f3D, BT, A1);
    cuda_contract(tid,  A1, GT, A2);
    cuda_contract(tid,  A2, BT, r3D, /* accumulate = */ true);

    f3D.values = &flux_q(0)[2];
    cuda_contract(tid, f3D, BT, A1);
    cuda_contract(tid,  A1, BT, A2);
    cuda_contract(tid,  A2, GT, r3D, /* accumulate = */ true);
  }
  #endif

  uint32_t p;

};

} // namespace refactor
